"""
DeepSeek-backed Q_proxy family generation for CAWOT V3.

V3 expects a list of proxy distributions, not one pooled query blob. This
module generates several named query families so ``cawot.scoring`` can compute
per-family MMD scores, mean relevance (abar), and proxy disagreement (u).

The OpenAI SDK import is intentionally lazy so the rest of the package remains
importable on machines that only run selection from pre-generated proxy files.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

logger = logging.getLogger("qproxy_v3")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-pro"

DEFAULT_FAMILY_NAMES = ("anomaly", "normal", "hard_cases", "appearance")
HYBRID_FAMILY_NAMES = (
    "Q1_templates",
    "Q2_deepseek_paraphrases",
    "Q3_deepseek_anomaly",
    "Q4_deepseek_hard_cases",
    "Q5_deepseek_normal",
    "Q6_deepseek_appearance",
)
HYBRID_GENERATED_FAMILY_NAMES = HYBRID_FAMILY_NAMES[1:]

SYSTEM_PROMPT = """You generate proxy query distributions for CAWOT V3, a text-to-image person anomaly retrieval coreset selector.

Output only valid JSON. Do not include markdown, comments, explanations, or numbering.
Every query must be a single English sentence that could be used to retrieve an image of a person.
Avoid identities, private information, camera IDs, dataset names, and references to test data."""

DEFAULT_USER_PROMPT_TEMPLATE = """Generate exactly {n_per_family} unique queries for each proxy family.

Families:
- anomaly: accidental or anomalous person behavior, such as falling, slipping, tripping, colliding, dropping objects, awkward landing, near-miss, loss of balance, or getting hit.
- normal: normal daily person behavior, such as walking, standing, sitting, riding, playing, carrying objects, waiting, talking, or performing.
- hard_cases: ambiguous, borderline, rare, or visually difficult person behavior where normal and anomalous interpretations may be confused.
- appearance: person-focused queries where clothing, accessories, pose, age group, and scene are emphasized more than the event label.

Every query must include:
- a person subject;
- clothing or appearance detail;
- an action or pose;
- a scene or location.

Length: 8 to 36 words per query.
Style mix: surveillance description, natural search phrase, and short conversational description.
Batch seed for diversity: {batch_idx}

Return this exact JSON shape:
{{
  "families": {{
    "anomaly": ["..."],
    "normal": ["..."],
    "hard_cases": ["..."],
    "appearance": ["..."]
  }}
}}"""

HYBRID_USER_PROMPT_TEMPLATE = """Generate exactly {n_per_family} unique queries for each DeepSeek proxy family.

Q1_templates already exists and was built from training annotations. Use these Q1 examples as semantic anchors, but do not copy them verbatim:
{seed_queries}

Families to output:
- Q2_deepseek_paraphrases: paraphrases and stylistic rewrites of the Q1 training-attribute query ideas, preserving training-side semantics while varying wording.
- Q3_deepseek_anomaly: anomaly-heavy person search queries with accidents, falls, slips, collisions, dropped objects, near-misses, awkward landings, and loss of balance.
- Q4_deepseek_hard_cases: ambiguous or visually hard person search queries where normal and anomalous behavior could be confused.
- Q5_deepseek_normal: normal daily person behavior, such as walking, standing, sitting, riding, playing, carrying objects, waiting, talking, or performing.
- Q6_deepseek_appearance: person-focused retrieval queries where clothing, accessories, age group, pose, body position, and scene are emphasized more than event labels.

Every query must include:
- a person subject;
- clothing or appearance detail;
- an action or pose;
- a scene or location.

Length: 8 to 35 words per query.
Style mix: surveillance description, natural search phrase, and short conversational description.
Batch seed for diversity: {batch_idx}

Return this exact JSON shape:
{{
  "families": {{
    "Q2_deepseek_paraphrases": ["..."],
    "Q3_deepseek_anomaly": ["..."],
    "Q4_deepseek_hard_cases": ["..."],
    "Q5_deepseek_normal": ["..."],
    "Q6_deepseek_appearance": ["..."]
  }}
}}"""


@dataclass
class CallStats:
    batch_idx: int
    mode: str
    success: bool
    query_counts: dict[str, int] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    error: str = ""


@dataclass
class GenerationStats:
    mode: str
    model: str
    requested_calls: int
    n_per_family: int
    successful_calls: int = 0
    failed_calls: int = 0
    resumed_calls: int = 0
    topup_calls: int = 0
    target_total_queries: int = 0
    raw_query_count: int = 0
    unique_query_count: int = 0
    duplicate_rate: float = 0.0
    duration_seconds: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    family_preset: str = "default"
    per_family_raw: dict[str, int] = field(default_factory=dict)
    per_family_unique: dict[str, int] = field(default_factory=dict)
    postprocess: dict[str, int] = field(default_factory=dict)
    token_usage: dict[str, int] = field(default_factory=dict)
    calls: list[CallStats] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = asdict(self)
        return out


@dataclass(frozen=True)
class ProgressEvent:
    mode: str
    completed_batches: int
    total_batches: int
    completed_queries: int
    target_queries: int
    resumed_calls: int = 0


def load_api_key(
    api_key_file: str | os.PathLike | None = None,
    *,
    direct_key: str | None = None,
    env_var: str = "DEEPSEEK_API_KEY",
) -> str:
    """Load a DeepSeek API key from CLI, file, environment, or deepseek_api.txt."""
    if direct_key:
        return direct_key.strip()

    if api_key_file:
        path = Path(api_key_file)
        if path.exists():
            return path.read_text(encoding="utf-8").strip()

    env_value = os.environ.get(env_var)
    if env_value:
        return env_value.strip()

    for candidate in (Path("deepseek_api.txt"), Path("../deepseek_api.txt")):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8").strip()

    raise RuntimeError(
        "DeepSeek API key not found. Pass --api-key, pass --api-key-file, "
        "set DEEPSEEK_API_KEY, or place deepseek_api.txt in the working directory."
    )


def make_client(api_key: str, base_url: str = DEEPSEEK_BASE_URL, timeout: float = 90.0):
    """Create an OpenAI-compatible DeepSeek client."""
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Install the OpenAI SDK first: pip install openai") from exc
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _normalise_query(text: str) -> str:
    return " ".join(text.strip().strip('"').split())


def _query_key(text: str) -> str:
    return " ".join(text.lower().split())


def family_names_for_preset(family_preset: str) -> tuple[str, ...]:
    if family_preset == "default":
        return DEFAULT_FAMILY_NAMES
    if family_preset == "hybrid":
        return HYBRID_FAMILY_NAMES
    raise ValueError("family_preset must be 'default' or 'hybrid'")


def generated_family_names_for_preset(family_preset: str) -> tuple[str, ...]:
    if family_preset == "default":
        return DEFAULT_FAMILY_NAMES
    if family_preset == "hybrid":
        return HYBRID_GENERATED_FAMILY_NAMES
    raise ValueError("family_preset must be 'default' or 'hybrid'")


def _empty_family_dict(family_names: Iterable[str]) -> dict[str, list[str]]:
    return {name: [] for name in family_names}


def parse_family_response(
    raw_text: str,
    family_names: Iterable[str] = DEFAULT_FAMILY_NAMES,
) -> dict[str, list[str]]:
    """Parse a model response into the V3 family dictionary."""
    family_names = tuple(family_names)
    out = _empty_family_dict(family_names)
    if not raw_text:
        return out

    text = _JSON_FENCE_RE.sub("", raw_text).strip()
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(text)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None

    if not isinstance(parsed, dict):
        return out

    families = parsed.get("families", parsed)
    if not isinstance(families, dict):
        return out

    for name in family_names:
        items = families.get(name, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, str):
                query = _normalise_query(item)
                if len(query.split()) >= 3:
                    out[name].append(query)
    return out


def dedupe_families(
    families: dict[str, list[str]],
    family_names: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    """Dedupe within each family while preserving family boundaries."""
    family_names = tuple(family_names or families.keys())
    deduped = _empty_family_dict(family_names)
    for name in family_names:
        seen: set[str] = set()
        for query in families.get(name, []):
            key = " ".join(query.lower().split())
            if key in seen:
                continue
            seen.add(key)
            deduped[name].append(query)
    return deduped


def postprocess_qproxy_families(
    families: dict[str, list[str]],
    family_names: Iterable[str] | None = None,
    *,
    min_query_words: int = 8,
    cross_family_dedupe: bool = True,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Filter weak queries and dedupe while preserving family priority order."""
    family_names = tuple(family_names or families.keys())
    cleaned = _empty_family_dict(family_names)
    seen_global: set[str] = set()
    report = {
        "input_query_count": 0,
        "kept_query_count": 0,
        "short_query_count": 0,
        "within_family_duplicate_count": 0,
        "cross_family_duplicate_count": 0,
        "malformed_query_count": 0,
        "artifact_query_count": 0,
    }
    artifact_re = re.compile(
        r"\b(doing birthday|at a snow|at a stairs|in a snow scene|"
        r"in a stairs scene|building in|building at|building near|"
        r"passing through at|moving around at|doing an activity involving break dancing)\b",
        re.IGNORECASE,
    )

    for name in family_names:
        seen_local: set[str] = set()
        for item in families.get(name, []):
            report["input_query_count"] += 1
            if not isinstance(item, str):
                report["malformed_query_count"] += 1
                continue
            query = _normalise_query(item)
            if len(query.split()) < min_query_words:
                report["short_query_count"] += 1
                continue
            if artifact_re.search(query):
                report["artifact_query_count"] += 1
                continue
            key = _query_key(query)
            if key in seen_local:
                report["within_family_duplicate_count"] += 1
                continue
            if cross_family_dedupe and key in seen_global:
                report["cross_family_duplicate_count"] += 1
                continue
            seen_local.add(key)
            seen_global.add(key)
            cleaned[name].append(query)
            report["kept_query_count"] += 1

    return cleaned, report


def trim_qproxy_families_to_total(
    families: dict[str, list[str]],
    family_names: Iterable[str],
    target_total: int,
    *,
    protected_family_names: Iterable[str] = (),
) -> tuple[dict[str, list[str]], int]:
    """Trim extra queries from the largest unprotected families to hit a target."""
    family_names = tuple(family_names)
    protected = set(protected_family_names)
    trimmed = {name: list(families.get(name, [])) for name in family_names}
    current = sum(len(trimmed[name]) for name in family_names)
    removed = 0
    while current > target_total:
        candidates = [name for name in family_names if name not in protected and trimmed[name]]
        if not candidates:
            candidates = [name for name in family_names if trimmed[name]]
        if not candidates:
            break
        name = max(candidates, key=lambda family: len(trimmed[family]))
        trimmed[name].pop()
        current -= 1
        removed += 1
    return trimmed, removed


def cap_families(
    families: dict[str, list[str]],
    family_names: Iterable[str],
    limit: int,
) -> dict[str, list[str]]:
    """Cap each named family to at most ``limit`` queries."""
    family_names = tuple(family_names)
    return {name: list(families.get(name, []))[:limit] for name in family_names}


def make_checkpoint_context(
    *,
    mode: str,
    model: str,
    family_preset: str,
    n_per_family: int,
    seed: int,
    generated_family_names: Iterable[str],
) -> dict:
    """Build the metadata used to reject stale batch checkpoints."""
    return {
        "schema_version": 1,
        "mode": mode,
        "model": model,
        "family_preset": family_preset,
        "n_per_family": n_per_family,
        "seed": seed,
        "generated_family_names": list(generated_family_names),
    }


def _checkpoint_path(checkpoint_dir: str | os.PathLike, batch_idx: int) -> Path:
    return Path(checkpoint_dir) / f"batch_{batch_idx:05d}.json"


def save_batch_checkpoint(
    checkpoint_dir: str | os.PathLike,
    *,
    context: dict,
    batch_idx: int,
    families: dict[str, list[str]],
    call: CallStats,
) -> None:
    """Persist one successful generated batch for later resume."""
    path = _checkpoint_path(checkpoint_dir, batch_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "context": context,
        "batch_idx": batch_idx,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "families": families,
        "call": asdict(call),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _call_stats_from_dict(data: dict) -> CallStats | None:
    try:
        return CallStats(
            batch_idx=int(data.get("batch_idx", -1)),
            mode=str(data.get("mode", "")),
            success=bool(data.get("success", False)),
            query_counts=dict(data.get("query_counts", {}) or {}),
            input_tokens=int(data.get("input_tokens", 0) or 0),
            output_tokens=int(data.get("output_tokens", 0) or 0),
            total_tokens=int(data.get("total_tokens", 0) or 0),
            error=str(data.get("error", "") or ""),
        )
    except (TypeError, ValueError):
        return None


def load_batch_checkpoint(
    checkpoint_dir: str | os.PathLike,
    *,
    context: dict,
    batch_idx: int,
    family_names: Iterable[str],
) -> tuple[dict[str, list[str]], CallStats] | None:
    """Load a successful batch checkpoint if it matches the current run."""
    path = _checkpoint_path(checkpoint_dir, batch_idx)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Ignoring unreadable checkpoint: %s", path)
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("context") != context or payload.get("batch_idx") != batch_idx:
        return None

    call_payload = payload.get("call")
    if not isinstance(call_payload, dict):
        return None
    call = _call_stats_from_dict(call_payload)
    if call is None or not call.success:
        return None
    if call.batch_idx != batch_idx or call.mode != context.get("mode"):
        return None

    raw_families = payload.get("families")
    if not isinstance(raw_families, dict):
        return None

    family_names = tuple(family_names)
    families = _empty_family_dict(family_names)
    for name in family_names:
        items = raw_families.get(name, [])
        if not isinstance(items, list):
            return None
        for item in items:
            if isinstance(item, str):
                query = _normalise_query(item)
                if len(query.split()) >= 3:
                    families[name].append(query)

    return families, call


def _response_usage(resp) -> dict[str, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _thinking_body(mode: str) -> dict:
    if mode == "thinking":
        return {"thinking": {"type": "enabled"}}
    if mode == "nonthinking":
        return {"thinking": {"type": "disabled"}}
    raise ValueError("mode must be 'thinking' or 'nonthinking'")


def _clean_phrase(value: str) -> str | None:
    value = value.strip().lower().replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" .,:;-/")
    if not value:
        return None
    words = value.split()
    if not (1 <= len(words) <= 6):
        return None
    if len(value) > 48:
        return None
    if len(set(words)) <= max(1, len(words) // 3) and len(words) > 3:
        return None
    return value


def _iter_annotation_entries(annotation_dir: str | os.PathLike):
    root = Path(annotation_dir)
    files = sorted(root.glob("*.json"))
    if not files:
        files = sorted(root.rglob("*.json"))
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        if text.startswith("["):
            try:
                entries = json.loads(text)
            except json.JSONDecodeError:
                continue
            for entry in entries:
                if isinstance(entry, dict):
                    yield entry
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                yield entry


def extract_training_attributes(
    annotation_dir: str | os.PathLike,
    *,
    max_actions: int = 120,
    max_scenes: int = 40,
) -> tuple[list[str], list[str]]:
    """Extract compact action and scene phrases from PAB train annotations."""
    action_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    for entry in _iter_annotation_entries(annotation_dir):
        normal = entry.get("normal")
        if isinstance(normal, str):
            for part in re.split(r"[,;/|]+", normal):
                phrase = _clean_phrase(part)
                if phrase:
                    action_counts[phrase] += 1
        scene = entry.get("scene")
        if isinstance(scene, str):
            phrase = _clean_phrase(scene)
            if phrase:
                scene_counts[phrase] += 1

    actions = [p for p, _ in action_counts.most_common(max_actions)]
    scenes = [p for p, _ in scene_counts.most_common(max_scenes)]
    if not actions and not scenes:
        raise RuntimeError(f"No usable attributes found under {annotation_dir}")
    return actions, scenes


def build_q1_templates_from_annotations(
    annotation_dir: str | os.PathLike,
    *,
    max_queries: int = 400,
    seed: int = 42,
) -> list[str]:
    """Build Q1 training-attribute proxy queries from local annotations."""
    def action_label(attr: str) -> str:
        special = {
            "baseball": "playing baseball",
            "basketball": "playing basketball",
            "soccer": "playing soccer",
            "gymnastics": "performing gymnastics",
            "martial arts": "practicing martial arts",
            "yoga": "practicing yoga",
            "wedding": "attending a wedding event",
            "birthday": "attending a birthday party",
            "building": "working on a construction task",
            "carrying": "carrying an object",
            "holding": "holding an object",
            "catching": "catching an object",
            "looking": "looking around",
            "opening door": "opening a door",
            "decorating": "decorating a display",
        }
        if attr in special:
            return special[attr]
        words = attr.split()
        first = words[0] if words else attr
        last = words[-1] if words else attr
        if first.endswith("ing") or last.endswith("ing"):
            return attr
        return f"doing an activity involving {attr}"

    def scene_context(scene: str) -> str:
        special = {
            "snow": "in a snowy outdoor area",
            "ice": "on an icy surface",
            "stairs": "near a stairway",
            "staircase": "near a stairway",
            "indoor": "inside an indoor public space",
            "outdoor": "in an outdoor public area",
            "highway": "near a roadside area",
            "birthday": "at a birthday party",
            "wedding": "at a wedding venue",
            "gymnastics": "inside a gymnastics facility",
            "christmas tree": "near a decorated indoor tree",
            "couch": "near a couch in a room",
            "wall": "beside a wall",
            "race": "at a race event",
            "pool": "near a swimming pool",
        }
        if scene in special:
            return special[scene]
        article = "an" if scene[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
        return f"at {article} {scene}"

    def action_queries(attr: str) -> list[str]:
        action = action_label(attr)
        return [
            f"a person wearing casual clothing {action} in a public area",
            f"a photo of a person in dark clothing {action} in an open public place",
            f"someone wearing a light jacket {action} while standing in a community space",
            f"a surveillance image of a person with a backpack {action} near a walkway",
            f"a person in a blue jacket {action} while another person passes nearby",
            f"a person wearing jeans and sneakers {action} in a busy public setting",
            f"a person in a gray hoodie {action} with other people visible nearby",
            f"a person carrying a small bag {action} beside a pedestrian walkway",
            f"a person in a short-sleeve shirt {action} under normal lighting",
            f"a person wearing a cap {action} in a casual everyday scene",
            f"a person in athletic clothing {action} in a recreational area",
            f"a person with a backpack {action} in front of a fixed camera view",
        ]

    def scene_queries(scene: str) -> list[str]:
        context = scene_context(scene)
        return [
            f"a person wearing casual clothing standing {context}",
            f"a surveillance image of a person in dark clothing walking {context}",
            f"someone with a backpack standing or walking {context}",
            f"a person in a light jacket waiting {context}",
            f"a person wearing jeans and sneakers walking {context}",
            f"a person carrying a small bag looking around {context}",
            f"a person in athletic clothing changing posture {context}",
        ]

    actions, scenes = extract_training_attributes(
        annotation_dir,
        max_actions=180,
        max_scenes=80,
    )
    queries: list[str] = []
    for attr in actions:
        queries.extend(action_queries(attr))
    for scene in scenes:
        queries.extend(scene_queries(scene))

    deduped, _ = postprocess_qproxy_families(
        {"q1": queries},
        ("q1",),
        min_query_words=8,
        cross_family_dedupe=False,
    )
    rng = random.Random(seed)
    q1 = deduped["q1"]
    rng.shuffle(q1)
    return q1[:max_queries]


def _generate_one_batch(
    client,
    *,
    batch_idx: int,
    n_per_family: int,
    mode: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    family_names: tuple[str, ...],
    user_prompt_template: str,
    seed_queries: list[str] | None = None,
) -> tuple[dict[str, list[str]], CallStats]:
    seed_block = "\n".join(f"- {q}" for q in (seed_queries or [])[:24])
    user_prompt = user_prompt_template.format(
        n_per_family=n_per_family,
        batch_idx=batch_idx,
        seed_queries=seed_block or "- a person walking in a public place",
    )
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "extra_body": _thinking_body(mode),
    }
    if mode == "thinking":
        kwargs["reasoning_effort"] = "high"
    else:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = 0.95

    backoff = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            families = cap_families(
                parse_family_response(content, family_names),
                family_names,
                n_per_family,
            )
            counts = {name: len(families[name]) for name in family_names}
            if sum(counts.values()) > 0:
                usage = _response_usage(resp)
                return families, CallStats(
                    batch_idx=batch_idx,
                    mode=mode,
                    success=True,
                    query_counts=counts,
                    **usage,
                )
            logger.warning("Batch %d %s attempt %d parsed 0 queries", batch_idx, mode, attempt)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Batch %d %s attempt %d failed: %s", batch_idx, mode, attempt, exc)
            if attempt == max_retries:
                return _empty_family_dict(family_names), CallStats(
                    batch_idx=batch_idx,
                    mode=mode,
                    success=False,
                    error=str(exc),
                )
        if attempt < max_retries:
            time.sleep(backoff)
            backoff *= 1.5

    return _empty_family_dict(family_names), CallStats(
        batch_idx=batch_idx,
        mode=mode,
        success=False,
        error="parsed zero queries",
    )


def generate_qproxy_families(
    *,
    api_key: str,
    mode: str,
    n_calls: int = 10,
    n_per_family: int = 25,
    max_workers: int = 2,
    model: str = DEEPSEEK_DEFAULT_MODEL,
    base_url: str = DEEPSEEK_BASE_URL,
    max_tokens: int = 8192,
    temperature: float = 0.85,
    temperature_jitter: float = 0.1,
    seed: int = 42,
    max_retries: int = 4,
    family_preset: str = "default",
    annotation_dir: str | os.PathLike | None = None,
    max_q1_queries: int = 400,
    checkpoint_dir: str | os.PathLike | None = None,
    progress_callback: Callable[[ProgressEvent], None] | None = None,
    target_total_queries: int | None = None,
    max_topup_calls: int = 0,
    min_query_words: int = 8,
    cross_family_dedupe: bool = True,
) -> tuple[dict[str, list[str]], GenerationStats]:
    """Generate one V3 Q_proxy family set in thinking or nonthinking mode."""
    if mode not in {"thinking", "nonthinking"}:
        raise ValueError("mode must be 'thinking' or 'nonthinking'")
    if n_calls <= 0:
        raise ValueError("n_calls must be positive")
    if n_per_family <= 0:
        raise ValueError("n_per_family must be positive")
    if target_total_queries is not None and target_total_queries <= 0:
        raise ValueError("target_total_queries must be positive")
    if max_topup_calls < 0:
        raise ValueError("max_topup_calls cannot be negative")
    if min_query_words <= 0:
        raise ValueError("min_query_words must be positive")

    family_names = family_names_for_preset(family_preset)
    generated_family_names = generated_family_names_for_preset(family_preset)
    q1_queries: list[str] = []
    if family_preset == "hybrid":
        if annotation_dir is None:
            raise ValueError("family_preset='hybrid' requires annotation_dir")
        q1_queries = build_q1_templates_from_annotations(
            annotation_dir,
            max_queries=max_q1_queries,
            seed=seed,
        )

    started = datetime.now(timezone.utc)
    stats = GenerationStats(
        mode=mode,
        model=model,
        requested_calls=n_calls,
        n_per_family=n_per_family,
        family_preset=family_preset,
        target_total_queries=target_total_queries or 0,
        started_at=started.isoformat(timespec="seconds"),
        per_family_raw={name: 0 for name in family_names},
        per_family_unique={name: 0 for name in family_names},
        token_usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    )

    rng = random.Random(seed)
    batches = []
    for batch_idx in range(n_calls):
        temp = max(0.1, min(1.5, temperature + rng.uniform(-temperature_jitter, temperature_jitter)))
        batches.append((batch_idx, round(temp, 3)))

    raw = _empty_family_dict(family_names)
    if q1_queries:
        raw["Q1_templates"].extend(q1_queries)
    planned_query_count = len(q1_queries) + (
        n_calls * n_per_family * len(generated_family_names)
    )
    progress_target_query_count = target_total_queries or planned_query_count
    planned_total_batches = n_calls + (max_topup_calls if target_total_queries else 0)
    checkpoint_context = make_checkpoint_context(
        mode=mode,
        model=model,
        family_preset=family_preset,
        n_per_family=n_per_family,
        seed=seed,
        generated_family_names=generated_family_names,
    )

    completed_batches = 0
    client = None

    def get_client():
        nonlocal client
        if client is None:
            client = make_client(api_key, base_url=base_url)
        return client

    def count_current_output_queries() -> int:
        cleaned, _ = postprocess_qproxy_families(
            raw,
            family_names,
            min_query_words=min_query_words,
            cross_family_dedupe=cross_family_dedupe,
        )
        count = sum(len(cleaned[name]) for name in family_names)
        if target_total_queries is not None:
            return min(count, target_total_queries)
        return count

    def emit_progress() -> None:
        if progress_callback is None:
            return
        progress_callback(
            ProgressEvent(
                mode=mode,
                completed_batches=completed_batches,
                total_batches=planned_total_batches,
                completed_queries=count_current_output_queries(),
                target_queries=progress_target_query_count,
                resumed_calls=stats.resumed_calls,
            )
        )

    def record_batch(
        families: dict[str, list[str]],
        call: CallStats,
        *,
        resumed: bool = False,
    ) -> None:
        nonlocal completed_batches
        stats.calls.append(call)
        completed_batches += 1
        if call.success:
            stats.successful_calls += 1
            if resumed:
                stats.resumed_calls += 1
            for name in generated_family_names:
                raw[name].extend(families[name])
        else:
            stats.failed_calls += 1
        stats.token_usage["input_tokens"] += call.input_tokens
        stats.token_usage["output_tokens"] += call.output_tokens
        stats.token_usage["total_tokens"] += call.total_tokens
        emit_progress()
        if completed_batches % 5 == 0 or completed_batches == n_calls:
            logger.info(
                "%s progress: %d/%d batches, %d/%d qproxy queries",
                mode,
                completed_batches,
                planned_total_batches,
                count_current_output_queries(),
                progress_target_query_count,
            )

    def seed_queries_for_batch(batch_idx: int) -> list[str] | None:
        if not q1_queries:
            return None
        return random.Random(seed + batch_idx).sample(
            q1_queries,
            k=min(24, len(q1_queries)),
        )

    def run_pending_batches(pending_batches: list[tuple[int, float]]) -> None:
        if not pending_batches:
            return
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _generate_one_batch,
                    get_client(),
                    batch_idx=batch_idx,
                    n_per_family=n_per_family,
                    mode=mode,
                    model=model,
                    temperature=temp,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    family_names=generated_family_names,
                    user_prompt_template=(
                        HYBRID_USER_PROMPT_TEMPLATE
                        if family_preset == "hybrid"
                        else DEFAULT_USER_PROMPT_TEMPLATE
                    ),
                    seed_queries=seed_queries_for_batch(batch_idx),
                ): batch_idx
                for batch_idx, temp in pending_batches
            }
            for fut in as_completed(futures):
                families, call = fut.result()
                if call.success and checkpoint_dir is not None:
                    save_batch_checkpoint(
                        checkpoint_dir,
                        context=checkpoint_context,
                        batch_idx=call.batch_idx,
                        families=families,
                        call=call,
                    )
                record_batch(families, call)

    def cleaned_output() -> tuple[dict[str, list[str]], dict[str, int]]:
        return postprocess_qproxy_families(
            raw,
            family_names,
            min_query_words=min_query_words,
            cross_family_dedupe=cross_family_dedupe,
        )

    emit_progress()
    t0 = time.time()

    pending_batches = []
    for batch_idx, temp in batches:
        loaded = None
        if checkpoint_dir is not None:
            loaded = load_batch_checkpoint(
                checkpoint_dir,
                context=checkpoint_context,
                batch_idx=batch_idx,
                family_names=generated_family_names,
            )
        if loaded is not None:
            families, call = loaded
            record_batch(families, call, resumed=True)
            continue
        pending_batches.append((batch_idx, temp))

    run_pending_batches(pending_batches)

    cleaned, postprocess_report = cleaned_output()
    if target_total_queries is not None:
        current_unique = sum(len(cleaned[name]) for name in family_names)
        estimated_queries_per_batch = max(1, n_per_family * len(generated_family_names))
        while current_unique < target_total_queries and stats.topup_calls < max_topup_calls:
            remaining_calls = max_topup_calls - stats.topup_calls
            deficit = target_total_queries - current_unique
            needed_calls = max(1, (deficit + estimated_queries_per_batch - 1) // estimated_queries_per_batch)
            wave_size = min(max_workers, remaining_calls, needed_calls)
            topup_batches = []
            for _ in range(wave_size):
                batch_idx = n_calls + stats.topup_calls
                stats.topup_calls += 1
                temp = max(0.1, min(1.5, temperature + rng.uniform(-temperature_jitter, temperature_jitter)))
                loaded = None
                if checkpoint_dir is not None:
                    loaded = load_batch_checkpoint(
                        checkpoint_dir,
                        context=checkpoint_context,
                        batch_idx=batch_idx,
                        family_names=generated_family_names,
                    )
                if loaded is not None:
                    families, call = loaded
                    record_batch(families, call, resumed=True)
                    continue
                topup_batches.append((batch_idx, round(temp, 3)))
            run_pending_batches(topup_batches)
            cleaned, postprocess_report = cleaned_output()
            current_unique = sum(len(cleaned[name]) for name in family_names)

    # Keep the original batch order in the manifest even when futures finish out of order.
    stats.calls.sort(key=lambda call: call.batch_idx)

    stats.per_family_raw = {name: len(raw[name]) for name in family_names}
    stats.raw_query_count = sum(stats.per_family_raw.values())
    if target_total_queries is not None:
        cleaned, trimmed_count = trim_qproxy_families_to_total(
            cleaned,
            family_names,
            target_total_queries,
            protected_family_names=("Q1_templates",) if family_preset == "hybrid" else (),
        )
        postprocess_report["trimmed_query_count"] = trimmed_count
    else:
        postprocess_report["trimmed_query_count"] = 0
    stats.per_family_unique = {name: len(cleaned[name]) for name in family_names}
    stats.unique_query_count = sum(stats.per_family_unique.values())
    postprocess_report["removed_query_count"] = stats.raw_query_count - stats.unique_query_count
    stats.postprocess = postprocess_report
    if stats.raw_query_count:
        stats.duplicate_rate = round(1.0 - stats.unique_query_count / stats.raw_query_count, 4)
    stats.duration_seconds = round(time.time() - t0, 1)
    stats.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return cleaned, stats


def compare_mode_outputs(
    thinking: dict[str, list[str]],
    nonthinking: dict[str, list[str]],
    family_names: Iterable[str] | None = None,
) -> dict:
    """Compare family overlap between thinking and nonthinking generations."""
    family_names = tuple(family_names or thinking.keys() or nonthinking.keys())
    report = {"families": {}, "overall_overlap": 0.0}
    all_a: set[str] = set()
    all_b: set[str] = set()
    for name in family_names:
        a = {" ".join(q.lower().split()) for q in thinking.get(name, [])}
        b = {" ".join(q.lower().split()) for q in nonthinking.get(name, [])}
        all_a |= a
        all_b |= b
        union = len(a | b)
        report["families"][name] = {
            "thinking_unique": len(a),
            "nonthinking_unique": len(b),
            "overlap": len(a & b),
            "jaccard": round(len(a & b) / union, 4) if union else 0.0,
        }
    union_all = len(all_a | all_b)
    report["overall_overlap"] = round(len(all_a & all_b) / union_all, 4) if union_all else 0.0
    return report


def save_family_outputs(
    families: dict[str, list[str]],
    stats: GenerationStats,
    out_dir: str | os.PathLike,
) -> None:
    """Write one JSON list per family plus manifest metadata."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in families:
        filename = f"{name}.json" if name.startswith("Q") else f"Q_{name}.json"
        path = out / filename
        path.write_text(
            json.dumps(families.get(name, []), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    manifest = {
        "family_files": {
            name: (f"{name}.json" if name.startswith("Q") else f"Q_{name}.json")
            for name in families
        },
        "stats": stats.to_dict(),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

from __future__ import annotations

import argparse
import asyncio
import configparser
import json
import math
import random
import re
import statistics
import textwrap
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable, Iterable, Optional

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table


BENCHMARK_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are completing a deterministic benchmark.
    Follow every instruction exactly.
    Never add explanations, markdown fences, or commentary.
    If asked for JSON, return only JSON.
    """
).strip()


SAFE_TOPICS = [
    "software design",
    "incident response",
    "testing strategy",
    "release engineering",
    "system reliability",
    "developer productivity",
    "code review",
    "architecture governance",
]

SAFE_TERMS = [
    "latency",
    "ownership",
    "rollback",
    "telemetry",
    "coverage",
    "refactor",
    "backlog",
    "oncall",
    "service",
    "quality",
    "design",
    "release",
    "monitoring",
    "capacity",
    "workflow",
    "defect",
    "resilience",
    "versioning",
    "planning",
    "testing",
]

SAFE_WORDS = [
    "atlas",
    "binary",
    "cinder",
    "delta",
    "ember",
    "fable",
    "garnet",
    "harbor",
    "ion",
    "jigsaw",
    "kepler",
    "lumen",
    "metric",
    "nova",
    "orbit",
    "prism",
    "quartz",
    "radar",
    "signal",
    "tensor",
    "umbra",
    "vector",
    "willow",
    "xenon",
    "yonder",
    "zenith",
]

FORBIDDEN_CHAR_POOL = [",", "，", ";", "；", ":", "：", "(", ")", "[", "]"]
STATE_POOL = ["ready", "hold", "green", "amber", "blue", "steady", "clear", "active"]
ZONE_POOL = ["north", "south", "east", "west", "delta", "ridge", "coast", "plain"]
TAG_POOL = ["alpha", "bravo", "cobalt", "drift", "ember", "flux", "glint", "haze", "ivory", "jolt"]

DIMENSION_LABELS = {
    "format_control": "Format Control",
    "json_schema": "JSON Schema",
    "constraint_binding": "Constraint Binding",
    "context_retrieval": "Context Retrieval",
}

DIMENSION_WEIGHTS = {
    "format_control": 1.0,
    "json_schema": 1.1,
    "constraint_binding": 1.1,
    "context_retrieval": 1.3,
}

DEFAULT_CONFIG_PATH = "probe.ini"
DEFAULT_PROVIDER = "openai"
DEFAULT_TIMEOUT = 120.0
DEFAULT_CASES_PER_DIMENSION = 5
DEFAULT_CONTEXT_RECORDS = 36
DEFAULT_REQUEST_RETRIES = 2


class ProbeError(RuntimeError):
    """Raised when the remote API probe cannot complete."""


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class TimedResponse:
    url: str
    status_code: int
    header_time_ms: float
    first_byte_time_ms: float
    total_time_ms: float
    body: bytes
    json_body: Optional[dict[str, Any]]


@dataclass
class ApiCallResult:
    text: str
    usage: dict[str, Any]
    raw: dict[str, Any]
    metrics: TimedResponse


@dataclass
class CaseCheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class DegradationCaseDefinition:
    case_id: str
    dimension: str
    title: str
    prompt: str
    reference_output: str
    evaluator: Callable[[str], tuple[list[CaseCheckResult], dict[str, Any]]]
    max_tokens: int = 384
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DegradationCaseResult:
    case_id: str
    dimension: str
    title: str
    prompt: str
    reference_output: str
    raw_text: str
    checks: list[CaseCheckResult]
    observations: dict[str, Any]
    metrics: TimedResponse

    @property
    def atomic_passes(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def atomic_total(self) -> int:
        return len(self.checks)

    @property
    def score_rate(self) -> float:
        return self.atomic_passes / self.atomic_total if self.atomic_total else 0.0

    @property
    def strict_pass(self) -> bool:
        return self.atomic_passes == self.atomic_total

    @property
    def failed_checks(self) -> list[str]:
        return [check.name for check in self.checks if not check.passed]


@dataclass
class RateInterval:
    low: float
    high: float


@dataclass
class DimensionSummary:
    dimension: str
    case_count: int
    atomic_pass_rate: float
    atomic_interval: RateInterval
    strict_pass_rate: float
    strict_interval: RateInterval
    mean_case_score: float
    score_stdev: float
    weak_cases: list[str]


@dataclass
class DegradationSuiteReport:
    seed: int
    cases_per_dimension: int
    total_cases: int
    overall_atomic_pass_rate: float
    overall_atomic_interval: RateInterval
    overall_strict_pass_rate: float
    overall_strict_interval: RateInterval
    mean_case_score: float
    score_stdev: float
    risk_level: str
    risk_score: float
    verdict: str
    calibration_note: str
    dimension_summaries: list[DimensionSummary]
    case_results: list[DegradationCaseResult]


@dataclass
class BaselineComparisonRow:
    label: str
    sample_count: int
    distance: float
    top_drifts: list[str]
    provider_hints: list[str]
    model_hints: list[str]


@dataclass
class BaselineComparison:
    profile_name: str
    compared_feature_count: int
    nearest_label: str
    nearest_distance: float
    rows: list[BaselineComparisonRow]
    note: str


@dataclass
class CacheReport:
    first: ApiCallResult
    second: ApiCallResult
    payload_indicator_name: str
    payload_indicator_value: int
    payload_hit: bool
    cliff_drop: bool
    delta_ms: float
    ratio: float
    verdict: str
    notes: str


def normalize_base_url(base_url: str) -> str:
    clean = base_url.rstrip("/")
    for suffix in ("/chat/completions", "/messages", "/models"):
        if clean.endswith(suffix):
            return clean[: -len(suffix)]
    return clean


def candidate_urls(base_url: str, endpoint: str) -> list[str]:
    root = normalize_base_url(base_url)
    endpoint = endpoint.strip("/")
    urls: list[str] = []

    def add(url: str) -> None:
        if url not in urls:
            urls.append(url)

    if root.endswith(f"/{endpoint}"):
        add(root)
        return urls

    if re.search(r"/v\d+$", root):
        add(f"{root}/{endpoint}")
    else:
        add(f"{root}/v1/{endpoint}")
        add(f"{root}/{endpoint}")
    return urls


def ms_from_ns(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / 1_000_000


def truncate(text: str, limit: int = 300) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def read_optional_text(config: configparser.ConfigParser, section: str, option: str) -> Optional[str]:
    if not config.has_option(section, option):
        return None
    value = config.get(section, option).strip()
    return value or None


def read_optional_int(config: configparser.ConfigParser, section: str, option: str) -> Optional[int]:
    value = read_optional_text(config, section, option)
    return int(value) if value is not None else None


def read_optional_float(config: configparser.ConfigParser, section: str, option: str) -> Optional[float]:
    value = read_optional_text(config, section, option)
    return float(value) if value is not None else None


def read_optional_bool(config: configparser.ConfigParser, section: str, option: str) -> Optional[bool]:
    if not config.has_option(section, option):
        return None
    return config.getboolean(section, option)


def load_runtime_config(config_path_arg: Optional[str]) -> tuple[dict[str, Any], Optional[Path]]:
    config_path = Path(config_path_arg) if config_path_arg else Path(DEFAULT_CONFIG_PATH)
    if not config_path.exists():
        if config_path_arg:
            raise ProbeError(f"Config file not found: {config_path}")
        return {}, None
    parser = configparser.ConfigParser(interpolation=None)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            parser.read_file(handle)
    except (OSError, configparser.Error) as exc:
        raise ProbeError(f"Failed to read config file {config_path}: {exc}") from exc
    try:
        config_values = {
            "provider": read_optional_text(parser, "probe", "provider"),
            "base_url": read_optional_text(parser, "probe", "base_url"),
            "api_key": read_optional_text(parser, "probe", "api_key"),
            "model": read_optional_text(parser, "probe", "model"),
            "timeout": read_optional_float(parser, "probe", "timeout"),
            "cases_per_dimension": read_optional_int(parser, "probe", "cases_per_dimension"),
            "context_records": read_optional_int(parser, "probe", "context_records"),
            "seed": read_optional_int(parser, "probe", "seed"),
            "request_retries": read_optional_int(parser, "probe", "request_retries"),
            "skip_cache": read_optional_bool(parser, "probe", "skip_cache"),
            "show_raw": read_optional_bool(parser, "probe", "show_raw"),
            "report_out": read_optional_text(parser, "report", "report_out"),
            "report_label": read_optional_text(parser, "report", "report_label"),
            "baseline_profile": read_optional_text(parser, "baseline", "baseline_profile"),
        }
    except ValueError as exc:
        raise ProbeError(f"Invalid value in config file {config_path}: {exc}") from exc
    return config_values, config_path


def resolve_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    config_values, config_path = load_runtime_config(args.config)
    args.config_resolved_path = str(config_path) if config_path is not None else None
    args.provider = coalesce(args.provider, config_values.get("provider"), DEFAULT_PROVIDER)
    args.base_url = coalesce(args.base_url, config_values.get("base_url"))
    args.api_key = coalesce(args.api_key, config_values.get("api_key"))
    args.model = coalesce(args.model, config_values.get("model"))
    args.timeout = coalesce(args.timeout, config_values.get("timeout"), DEFAULT_TIMEOUT)
    args.cases_per_dimension = coalesce(
        args.cases_per_dimension,
        config_values.get("cases_per_dimension"),
        DEFAULT_CASES_PER_DIMENSION,
    )
    args.context_records = coalesce(
        args.context_records,
        config_values.get("context_records"),
        DEFAULT_CONTEXT_RECORDS,
    )
    args.seed = coalesce(args.seed, config_values.get("seed"))
    args.request_retries = coalesce(
        args.request_retries,
        config_values.get("request_retries"),
        DEFAULT_REQUEST_RETRIES,
    )
    args.skip_cache = coalesce(args.skip_cache, config_values.get("skip_cache"), False)
    args.show_raw = coalesce(args.show_raw, config_values.get("show_raw"), False)
    args.report_out = coalesce(args.report_out, config_values.get("report_out"))
    args.report_label = coalesce(args.report_label, config_values.get("report_label"))
    args.baseline_profile = coalesce(
        args.baseline_profile,
        config_values.get("baseline_profile"),
    )
    return args


def wilson_interval(successes: int, total: int, z: float = 1.96) -> RateInterval:
    if total <= 0:
        return RateInterval(0.0, 0.0)
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return RateInterval(max(0.0, center - margin), min(1.0, center + margin))


def parse_full_json_object(text: str) -> Optional[dict[str, Any]]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_json_tail(text: str) -> tuple[str, Optional[dict[str, Any]]]:
    stripped = text.strip()
    positions = [match.start() for match in re.finditer(r"\{", stripped)]
    for start in reversed(positions):
        candidate = stripped[start:].strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return stripped[:start].rstrip(), parsed
    return stripped, None


def split_paragraphs(body_text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body_text) if part.strip()]
    if len(paragraphs) <= 1:
        line_based = [line.strip() for line in body_text.splitlines() if line.strip()]
        if len(line_based) > 1:
            paragraphs = line_based
    return paragraphs


def sentence_count(paragraph: str) -> int:
    parts = [segment.strip() for segment in re.split(r"[.。]", paragraph) if segment.strip()]
    return len(parts)


def random_nonce(rng: random.Random, prefix: str) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return prefix + "".join(rng.choice(alphabet) for _ in range(6))


def choose_words(rng: random.Random, pool: list[str], size: int) -> list[str]:
    return rng.sample(pool, size)


def build_format_reference(topic: str, terms: list[str], sentences_per_paragraph: int) -> list[str]:
    template_pool = [
        "{term} improves {topic} practice.",
        "Teams protect reliability through review.",
        "{term} keeps delivery steady under load.",
        "Clear ownership reduces hidden risk.",
        "{term} helps engineers detect drift.",
        "Simple plans keep {topic} work calm.",
        "{term} gives projects stable direction.",
        "Strong habits preserve service health.",
    ]
    paragraphs: list[str] = []
    template_index = 0
    for term in terms:
        sentences: list[str] = []
        for sentence_index in range(sentences_per_paragraph):
            template = template_pool[(template_index + sentence_index) % len(template_pool)]
            sentences.append(template.format(term=term, topic=topic))
        template_index += 2
        paragraphs.append(" ".join(sentences))
    return paragraphs


def build_format_case(rng: random.Random, index: int) -> DegradationCaseDefinition:
    case_id = f"FMT-{index:02d}"
    topic = rng.choice(SAFE_TOPICS)
    paragraph_count = rng.randint(2, 4)
    sentences_per_paragraph = rng.randint(1, 3)
    terms = choose_words(rng, SAFE_TERMS, paragraph_count)
    forbidden_chars = rng.sample(FORBIDDEN_CHAR_POOL, rng.randint(2, 4))
    tail = {"case_id": case_id, "nonce": random_nonce(rng, "F"), "status": "ok"}
    reference_paragraphs = build_format_reference(topic, terms, sentences_per_paragraph)
    reference_output = "\n\n".join(reference_paragraphs) + "\n" + json.dumps(
        tail, ensure_ascii=False, separators=(",", ":")
    )
    prompt_lines = [
        f"Benchmark case: {case_id}",
        f"Write about {topic}.",
        f"The essay body must contain exactly {paragraph_count} paragraphs.",
        f"Each paragraph must contain exactly {sentences_per_paragraph} sentences.",
    ]
    for idx, term in enumerate(terms, start=1):
        prompt_lines.append(f'Paragraph {idx} must contain the exact term "{term}".')
    prompt_lines.extend(
        [
            "Do not use any of these characters anywhere in the essay body: "
            + " ".join(json.dumps(char, ensure_ascii=False) for char in forbidden_chars),
            "Use only periods or Chinese full stops as sentence-ending punctuation.",
            "After the essay body output exactly this JSON object on its own line:",
            json.dumps(tail, ensure_ascii=False, separators=(",", ":")),
            "Do not output anything else.",
        ]
    )
    prompt = "\n".join(prompt_lines)

    def evaluator(text: str) -> tuple[list[CaseCheckResult], dict[str, Any]]:
        body_text, parsed_tail = extract_json_tail(text)
        paragraphs = split_paragraphs(body_text)
        sentence_counts = [sentence_count(paragraph) for paragraph in paragraphs]
        forbidden_hits = [char for char in forbidden_chars if char in body_text]
        term_hits = [
            idx < len(paragraphs) and terms[idx] in paragraphs[idx]
            for idx in range(len(terms))
        ]
        checks = [
            CaseCheckResult(
                name="paragraph_count",
                passed=len(paragraphs) == paragraph_count,
                detail=f"Observed {len(paragraphs)} paragraph(s).",
            ),
            CaseCheckResult(
                name="sentence_count",
                passed=len(paragraphs) == paragraph_count
                and all(count == sentences_per_paragraph for count in sentence_counts),
                detail=f"Sentence counts: {sentence_counts or '[]'}.",
            ),
            CaseCheckResult(
                name="forbidden_chars",
                passed=not forbidden_hits,
                detail="No forbidden body characters detected."
                if not forbidden_hits
                else f"Detected forbidden characters: {forbidden_hits}.",
            ),
            CaseCheckResult(
                name="required_terms",
                passed=all(term_hits),
                detail="All required terms appeared in the assigned paragraphs."
                if all(term_hits)
                else f"Paragraph term hits: {term_hits}.",
            ),
            CaseCheckResult(
                name="json_tail",
                passed=parsed_tail == tail,
                detail="Trailing JSON matched exactly."
                if parsed_tail == tail
                else f"Parsed trailing JSON: {parsed_tail}.",
            ),
        ]
        return checks, {
            "paragraph_count": len(paragraphs),
            "sentence_counts": sentence_counts,
            "forbidden_hits": forbidden_hits,
            "parsed_tail": parsed_tail,
        }

    return DegradationCaseDefinition(
        case_id=case_id,
        dimension="format_control",
        title="Strict multi-constraint essay formatting",
        prompt=prompt,
        reference_output=reference_output,
        evaluator=evaluator,
        max_tokens=420,
        metadata={
            "topic": topic,
            "paragraph_count": paragraph_count,
            "sentences_per_paragraph": sentences_per_paragraph,
            "terms": terms,
            "forbidden_chars": forbidden_chars,
            "tail": tail,
        },
    )


def build_json_case(rng: random.Random, index: int) -> DegradationCaseDefinition:
    case_id = f"JSN-{index:02d}"
    nonce = random_nonce(rng, "J")
    numbers = rng.sample(range(11, 98), 5)
    words = choose_words(rng, SAFE_WORDS, 4)
    sorted_numbers = sorted(numbers)
    summary = {
        "min": min(numbers),
        "max": max(numbers),
        "sum": sum(numbers),
        "span": max(numbers) - min(numbers),
    }
    expected = {
        "case_id": case_id,
        "nonce": nonce,
        "numbers_sorted": sorted_numbers,
        "summary": summary,
        "even_count": sum(1 for number in numbers if number % 2 == 0),
        "initials": "".join(word[0].upper() for word in words),
        "words_reversed": list(reversed(words)),
    }
    prompt = textwrap.dedent(
        f"""
        Benchmark case: {case_id}
        Return exactly one JSON object and nothing else.
        Source numbers: {numbers}
        Source words: {words}
        Build an object with these exact keys:
        - case_id
        - nonce
        - numbers_sorted
        - summary
        - even_count
        - initials
        - words_reversed
        Rules:
        - case_id must be "{case_id}"
        - nonce must be "{nonce}"
        - numbers_sorted must sort the source numbers ascending
        - summary must contain min max sum span
        - even_count counts even source numbers
        - initials concatenates the uppercase first letter of each source word in the original order
        - words_reversed reverses the source words
        """
    ).strip()
    reference_output = json.dumps(expected, ensure_ascii=False, separators=(",", ":"))

    def evaluator(text: str) -> tuple[list[CaseCheckResult], dict[str, Any]]:
        parsed = parse_full_json_object(text)
        checks = [
            CaseCheckResult(
                name="json_only",
                passed=parsed is not None,
                detail="Response parsed as a single JSON object."
                if parsed is not None
                else "Response was not a clean JSON object.",
            ),
            CaseCheckResult(
                name="top_level_keys",
                passed=parsed is not None and set(parsed.keys()) == set(expected.keys()),
                detail=f"Observed keys: {sorted(parsed.keys())}" if parsed is not None else "No JSON keys available.",
            ),
            CaseCheckResult(
                name="number_transform",
                passed=parsed is not None
                and parsed.get("numbers_sorted") == expected["numbers_sorted"]
                and parsed.get("summary") == expected["summary"]
                and parsed.get("even_count") == expected["even_count"],
                detail="Numeric transforms matched."
                if parsed is not None
                and parsed.get("numbers_sorted") == expected["numbers_sorted"]
                and parsed.get("summary") == expected["summary"]
                and parsed.get("even_count") == expected["even_count"]
                else f"Observed numeric fields: {parsed if parsed is not None else None}",
            ),
            CaseCheckResult(
                name="string_transform",
                passed=parsed is not None
                and parsed.get("case_id") == expected["case_id"]
                and parsed.get("nonce") == expected["nonce"]
                and parsed.get("initials") == expected["initials"]
                and parsed.get("words_reversed") == expected["words_reversed"],
                detail="String transforms matched."
                if parsed is not None
                and parsed.get("case_id") == expected["case_id"]
                and parsed.get("nonce") == expected["nonce"]
                and parsed.get("initials") == expected["initials"]
                and parsed.get("words_reversed") == expected["words_reversed"]
                else f"Observed string fields: {parsed if parsed is not None else None}",
            ),
        ]
        return checks, {"parsed": parsed}

    return DegradationCaseDefinition(
        case_id=case_id,
        dimension="json_schema",
        title="Exact JSON schema and transforms",
        prompt=prompt,
        reference_output=reference_output,
        evaluator=evaluator,
        max_tokens=260,
        metadata={"numbers": numbers, "words": words, "expected": expected},
    )


def build_binding_case(rng: random.Random, index: int) -> DegradationCaseDefinition:
    case_id = f"BND-{index:02d}"
    nonce = random_nonce(rng, "B")
    row_count = rng.randint(4, 5)
    codes = [word.upper() for word in choose_words(rng, SAFE_WORDS, row_count)]
    states = choose_words(rng, STATE_POOL, row_count)
    values = rng.sample(range(12, 70), row_count)
    rows = []
    for slot in range(1, row_count + 1):
        rows.append((slot, codes[slot - 1], states[slot - 1], values[slot - 1]))
    checksum = sum(slot * value for slot, _, _, value in rows) + row_count
    expected_lines = [f"SLOT-{slot}|{code}|{state}|{value}" for slot, code, state, value in rows]
    checksum_line = f"CHECK|{case_id}|{nonce}|{checksum}"
    prompt_lines = [
        f"Benchmark case: {case_id}",
        f"Nonce: {nonce}",
        f"Output exactly {row_count + 1} lines and nothing else.",
        "For each registry row output one line with this exact format:",
        "SLOT-{slot}|{code}|{state}|{value}",
        "Use ascending slot order.",
        "After the registry lines output a checksum line with this exact format:",
        "CHECK|<case_id>|<nonce>|<checksum>",
        "Checksum rule: sum(slot * value) for all registry rows plus the number of registry rows.",
        "Registry:",
    ]
    prompt_lines.extend(
        f"- slot {slot} code {code} state {state} value {value}"
        for slot, code, state, value in rows
    )
    prompt = "\n".join(prompt_lines)
    reference_output = "\n".join(expected_lines + [checksum_line])

    def evaluator(text: str) -> tuple[list[CaseCheckResult], dict[str, Any]]:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        row_pattern = re.compile(r"^SLOT-(\d+)\|([A-Z]+)\|([a-z]+)\|(\d+)$")
        parsed_rows: list[tuple[int, str, str, int]] = []
        for line in lines[:-1]:
            match = row_pattern.fullmatch(line)
            if not match:
                continue
            parsed_rows.append(
                (int(match.group(1)), match.group(2), match.group(3), int(match.group(4)))
            )
        checks = [
            CaseCheckResult(
                name="line_count",
                passed=len(lines) == row_count + 1,
                detail=f"Observed {len(lines)} line(s).",
            ),
            CaseCheckResult(
                name="row_format",
                passed=len(parsed_rows) == row_count,
                detail=f"Parsed {len(parsed_rows)} registry row(s).",
            ),
            CaseCheckResult(
                name="row_binding",
                passed=[f"SLOT-{slot}|{code}|{state}|{value}" for slot, code, state, value in parsed_rows]
                == expected_lines,
                detail="Registry rows matched expected order and values."
                if [f"SLOT-{slot}|{code}|{state}|{value}" for slot, code, state, value in parsed_rows]
                == expected_lines
                else f"Observed rows: {parsed_rows}.",
            ),
            CaseCheckResult(
                name="checksum",
                passed=bool(lines) and lines[-1] == checksum_line,
                detail="Checksum line matched exactly."
                if bool(lines) and lines[-1] == checksum_line
                else f"Observed checksum line: {lines[-1] if lines else None}.",
            ),
        ]
        return checks, {"lines": lines, "parsed_rows": parsed_rows}

    return DegradationCaseDefinition(
        case_id=case_id,
        dimension="constraint_binding",
        title="Row binding and checksum control",
        prompt=prompt,
        reference_output=reference_output,
        evaluator=evaluator,
        max_tokens=220,
        metadata={"rows": rows, "checksum": checksum},
    )


def build_context_case(
    rng: random.Random,
    index: int,
    *,
    record_count: int,
) -> DegradationCaseDefinition:
    case_id = f"CTX-{index:02d}"
    nonce = random_nonce(rng, "C")
    record_ids = [f"REC-{slot:02d}" for slot in range(1, record_count + 1)]
    markers = [f"mk-{rng.randint(100, 999)}" for _ in range(record_count)]
    loads = rng.sample(range(15, 200), record_count)
    zones = [rng.choice(ZONE_POOL) for _ in range(record_count)]
    tags = [rng.choice(TAG_POOL) for _ in range(record_count)]
    tiers = [f"t{rng.randint(1, 5)}" for _ in range(record_count)]
    ledgers: list[dict[str, Any]] = []
    for idx, record_id in enumerate(record_ids):
        ledgers.append(
            {
                "id": record_id,
                "marker": markers[idx],
                "load": loads[idx],
                "zone": zones[idx],
                "tag": tags[idx],
                "tier": tiers[idx],
            }
        )
    targets = rng.sample(ledgers, 4)
    max_target = sorted(targets, key=lambda item: (-item["load"], item["id"]))[0]
    expected = {
        "case_id": case_id,
        "nonce": nonce,
        "selected_markers": [item["marker"] for item in targets],
        "total_load": sum(item["load"] for item in targets),
        "max_load_id": max_target["id"],
        "zone_signature": "".join(item["zone"][0].upper() for item in targets),
    }
    ledger_lines = [
        (
            f'{item["id"]} marker {item["marker"]} load {item["load"]} '
            f'zone {item["zone"]} tag {item["tag"]} tier {item["tier"]} '
            f'memo stable route window {idx + 11}'
        )
        for idx, item in enumerate(ledgers)
    ]
    prompt = textwrap.dedent(
        f"""
        Benchmark case: {case_id}
        Use only the ledger below.
        Return exactly one JSON object and nothing else.
        Required keys:
        - case_id
        - nonce
        - selected_markers
        - total_load
        - max_load_id
        - zone_signature
        Rules:
        - case_id must be "{case_id}"
        - nonce must be "{nonce}"
        - selected_markers must list markers for these target ids in this exact order:
          {[item["id"] for item in targets]}
        - total_load sums the loads for the target ids
        - max_load_id is the target id with the highest load
        - if there is a tie choose the lexicographically smaller id
        - zone_signature concatenates the uppercase first letter of each target zone in target order
        Ledger:
        {chr(10).join(ledger_lines)}
        """
    ).strip()
    reference_output = json.dumps(expected, ensure_ascii=False, separators=(",", ":"))

    def evaluator(text: str) -> tuple[list[CaseCheckResult], dict[str, Any]]:
        parsed = parse_full_json_object(text)
        keys_ok = parsed is not None and set(parsed.keys()) == set(expected.keys())
        selection_ok = parsed is not None and parsed.get("selected_markers") == expected["selected_markers"]
        numeric_ok = parsed is not None and parsed.get("total_load") == expected["total_load"]
        max_ok = parsed is not None and parsed.get("max_load_id") == expected["max_load_id"]
        zone_ok = parsed is not None and parsed.get("zone_signature") == expected["zone_signature"]
        id_ok = parsed is not None and parsed.get("case_id") == case_id and parsed.get("nonce") == nonce
        checks = [
            CaseCheckResult(
                name="json_only",
                passed=parsed is not None,
                detail="Response parsed as a single JSON object."
                if parsed is not None
                else "Response was not a clean JSON object.",
            ),
            CaseCheckResult(
                name="top_level_keys",
                passed=keys_ok,
                detail=f"Observed keys: {sorted(parsed.keys())}" if parsed is not None else "No JSON keys available.",
            ),
            CaseCheckResult(
                name="identity_fields",
                passed=id_ok,
                detail="case_id and nonce matched."
                if id_ok
                else f"Observed identity fields: {parsed if parsed is not None else None}",
            ),
            CaseCheckResult(
                name="marker_selection",
                passed=selection_ok,
                detail="Selected markers matched."
                if selection_ok
                else f"Observed selected_markers: {parsed.get('selected_markers') if parsed is not None else None}.",
            ),
            CaseCheckResult(
                name="aggregate_fields",
                passed=numeric_ok and max_ok and zone_ok,
                detail="Aggregates matched expected values."
                if numeric_ok and max_ok and zone_ok
                else f"Observed aggregate fields: {parsed if parsed is not None else None}",
            ),
        ]
        return checks, {"parsed": parsed}

    return DegradationCaseDefinition(
        case_id=case_id,
        dimension="context_retrieval",
        title="Long-context retrieval and aggregation",
        prompt=prompt,
        reference_output=reference_output,
        evaluator=evaluator,
        max_tokens=260,
        metadata={"targets": [item["id"] for item in targets], "expected": expected},
    )


def build_degradation_suite(
    *,
    seed: int,
    cases_per_dimension: int,
    context_record_count: int,
) -> list[DegradationCaseDefinition]:
    rng = random.Random(seed)
    suite: list[DegradationCaseDefinition] = []
    for index in range(1, cases_per_dimension + 1):
        suite.append(build_format_case(rng, index))
        suite.append(build_json_case(rng, index))
        suite.append(build_binding_case(rng, index))
        suite.append(build_context_case(rng, index, record_count=context_record_count))
    return suite


def evaluate_case_result(
    definition: DegradationCaseDefinition,
    response: ApiCallResult,
) -> DegradationCaseResult:
    checks, observations = definition.evaluator(response.text)
    return DegradationCaseResult(
        case_id=definition.case_id,
        dimension=definition.dimension,
        title=definition.title,
        prompt=definition.prompt,
        reference_output=definition.reference_output,
        raw_text=response.text,
        checks=checks,
        observations=observations,
        metrics=response.metrics,
    )


def aggregate_degradation_suite(
    *,
    seed: int,
    cases_per_dimension: int,
    case_results: list[DegradationCaseResult],
) -> DegradationSuiteReport:
    atomic_successes = sum(result.atomic_passes for result in case_results)
    atomic_total = sum(result.atomic_total for result in case_results)
    strict_successes = sum(1 for result in case_results if result.strict_pass)
    case_scores = [result.score_rate for result in case_results]
    dimension_summaries: list[DimensionSummary] = []
    for dimension in DIMENSION_LABELS:
        items = [result for result in case_results if result.dimension == dimension]
        dim_atomic_successes = sum(result.atomic_passes for result in items)
        dim_atomic_total = sum(result.atomic_total for result in items)
        dim_strict_successes = sum(1 for result in items if result.strict_pass)
        dim_scores = [result.score_rate for result in items]
        weak_cases = [result.case_id for result in items if result.score_rate < 0.75]
        dimension_summaries.append(
            DimensionSummary(
                dimension=dimension,
                case_count=len(items),
                atomic_pass_rate=dim_atomic_successes / dim_atomic_total if dim_atomic_total else 0.0,
                atomic_interval=wilson_interval(dim_atomic_successes, dim_atomic_total),
                strict_pass_rate=dim_strict_successes / len(items) if items else 0.0,
                strict_interval=wilson_interval(dim_strict_successes, len(items)),
                mean_case_score=statistics.mean(dim_scores) if dim_scores else 0.0,
                score_stdev=statistics.pstdev(dim_scores) if len(dim_scores) > 1 else 0.0,
                weak_cases=weak_cases,
            )
        )

    weighted_score = 0.0
    total_weight = 0.0
    for summary in dimension_summaries:
        weight = DIMENSION_WEIGHTS.get(summary.dimension, 1.0)
        weighted_score += summary.mean_case_score * weight
        total_weight += weight
    weighted_score = weighted_score / total_weight if total_weight else 0.0
    mean_case_score = statistics.mean(case_scores) if case_scores else 0.0
    score_stdev = statistics.pstdev(case_scores) if len(case_scores) > 1 else 0.0
    overall_atomic_pass_rate = atomic_successes / atomic_total if atomic_total else 0.0
    overall_strict_pass_rate = strict_successes / len(case_results) if case_results else 0.0

    stability_penalty = min(0.15, score_stdev * 0.4)
    strict_penalty = (1.0 - overall_strict_pass_rate) * 0.18
    weakest_dimension_penalty = max(
        0.0,
        0.12 - min(summary.mean_case_score for summary in dimension_summaries)
        if dimension_summaries
        else 0.0,
    )
    risk_score = max(0.0, min(1.0, weighted_score - stability_penalty - strict_penalty - weakest_dimension_penalty))

    if risk_score >= 0.95:
        risk_level = "Minimal"
    elif risk_score >= 0.88:
        risk_level = "Low"
    elif risk_score >= 0.75:
        risk_level = "Moderate"
    elif risk_score >= 0.60:
        risk_level = "High"
    else:
        risk_level = "Critical"

    weakest = sorted(dimension_summaries, key=lambda item: item.mean_case_score)[:2]
    if dimension_summaries and (
        max(summary.mean_case_score for summary in dimension_summaries)
        - min(summary.mean_case_score for summary in dimension_summaries)
        < 0.02
    ):
        weakness_text = "none, all dimension scores are effectively tied"
    else:
        weakness_text = ", ".join(DIMENSION_LABELS[item.dimension] for item in weakest) if weakest else "none"
    verdict = (
        f"Heuristic risk level {risk_level}. "
        f"Composite score {risk_score:.3f}. "
        f"Weakest dimensions: {weakness_text}."
    )
    calibration_note = (
        "Risk level is a reference-free heuristic. Calibrate against known flagship and small-model baselines "
        "before treating it as a probabilistic substitution claim."
    )
    return DegradationSuiteReport(
        seed=seed,
        cases_per_dimension=cases_per_dimension,
        total_cases=len(case_results),
        overall_atomic_pass_rate=overall_atomic_pass_rate,
        overall_atomic_interval=wilson_interval(atomic_successes, atomic_total),
        overall_strict_pass_rate=overall_strict_pass_rate,
        overall_strict_interval=wilson_interval(strict_successes, len(case_results)),
        mean_case_score=mean_case_score,
        score_stdev=score_stdev,
        risk_level=risk_level,
        risk_score=risk_score,
        verdict=verdict,
        calibration_note=calibration_note,
        dimension_summaries=dimension_summaries,
        case_results=case_results,
    )


def serialize_timed_response(metrics: TimedResponse) -> dict[str, Any]:
    return {
        "url": metrics.url,
        "status_code": metrics.status_code,
        "header_time_ms": metrics.header_time_ms,
        "first_byte_time_ms": metrics.first_byte_time_ms,
        "total_time_ms": metrics.total_time_ms,
    }


def serialize_case_result(result: DegradationCaseResult) -> dict[str, Any]:
    return {
        "case_id": result.case_id,
        "dimension": result.dimension,
        "title": result.title,
        "prompt": result.prompt,
        "reference_output": result.reference_output,
        "raw_text": result.raw_text,
        "score_rate": result.score_rate,
        "strict_pass": result.strict_pass,
        "checks": [
            {"name": check.name, "passed": check.passed, "detail": check.detail}
            for check in result.checks
        ],
        "observations": result.observations,
        "metrics": serialize_timed_response(result.metrics),
    }


def serialize_dimension_summary(summary: DimensionSummary) -> dict[str, Any]:
    return {
        "dimension": summary.dimension,
        "case_count": summary.case_count,
        "atomic_pass_rate": summary.atomic_pass_rate,
        "atomic_interval": {"low": summary.atomic_interval.low, "high": summary.atomic_interval.high},
        "strict_pass_rate": summary.strict_pass_rate,
        "strict_interval": {"low": summary.strict_interval.low, "high": summary.strict_interval.high},
        "mean_case_score": summary.mean_case_score,
        "score_stdev": summary.score_stdev,
        "weak_cases": summary.weak_cases,
    }


def serialize_degradation_report(report: DegradationSuiteReport) -> dict[str, Any]:
    return {
        "seed": report.seed,
        "cases_per_dimension": report.cases_per_dimension,
        "total_cases": report.total_cases,
        "overall_atomic_pass_rate": report.overall_atomic_pass_rate,
        "overall_atomic_interval": {
            "low": report.overall_atomic_interval.low,
            "high": report.overall_atomic_interval.high,
        },
        "overall_strict_pass_rate": report.overall_strict_pass_rate,
        "overall_strict_interval": {
            "low": report.overall_strict_interval.low,
            "high": report.overall_strict_interval.high,
        },
        "mean_case_score": report.mean_case_score,
        "score_stdev": report.score_stdev,
        "risk_level": report.risk_level,
        "risk_score": report.risk_score,
        "verdict": report.verdict,
        "calibration_note": report.calibration_note,
        "dimension_summaries": [
            serialize_dimension_summary(summary) for summary in report.dimension_summaries
        ],
        "case_results": [serialize_case_result(result) for result in report.case_results],
    }


def serialize_api_call_result(result: ApiCallResult) -> dict[str, Any]:
    return {
        "text": result.text,
        "usage": result.usage,
        "raw": result.raw,
        "metrics": serialize_timed_response(result.metrics),
    }


def serialize_cache_report(report: CacheReport) -> dict[str, Any]:
    return {
        "first": serialize_api_call_result(report.first),
        "second": serialize_api_call_result(report.second),
        "payload_indicator_name": report.payload_indicator_name,
        "payload_indicator_value": report.payload_indicator_value,
        "payload_hit": report.payload_hit,
        "cliff_drop": report.cliff_drop,
        "delta_ms": report.delta_ms,
        "ratio": report.ratio,
        "verdict": report.verdict,
        "notes": report.notes,
    }


def serialize_baseline_comparison(comparison: BaselineComparison) -> dict[str, Any]:
    return {
        "profile_name": comparison.profile_name,
        "compared_feature_count": comparison.compared_feature_count,
        "nearest_label": comparison.nearest_label,
        "nearest_distance": comparison.nearest_distance,
        "rows": [
            {
                "label": row.label,
                "sample_count": row.sample_count,
                "distance": row.distance,
                "top_drifts": row.top_drifts,
                "provider_hints": row.provider_hints,
                "model_hints": row.model_hints,
            }
            for row in comparison.rows
        ],
        "note": comparison.note,
    }


def build_run_artifact(
    *,
    args: argparse.Namespace,
    provider: Provider,
    model: str,
    degradation_report: DegradationSuiteReport,
    cache_report: Optional[CacheReport],
    baseline_comparison: Optional[BaselineComparison],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "transit-probe-run-report",
        "created_at": timestamp_utc(),
        "report_label": args.report_label,
        "provider": provider.value,
        "model": model,
        "base_url": normalize_base_url(args.base_url),
        "suite_config": {
            "cases_per_dimension": args.cases_per_dimension,
            "context_records": args.context_records,
            "request_retries": args.request_retries,
            "skip_cache": args.skip_cache,
            "seed": degradation_report.seed,
        },
        "degradation_report": serialize_degradation_report(degradation_report),
        "cache_report": serialize_cache_report(cache_report) if cache_report is not None else None,
        "baseline_comparison": serialize_baseline_comparison(baseline_comparison)
        if baseline_comparison is not None
        else None,
    }


def write_json_file(path_value: str, payload: dict[str, Any]) -> None:
    path = Path(path_value)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def feature_floor(feature_key: str) -> float:
    if feature_key.endswith("score_stdev"):
        return 0.05
    if feature_key.endswith("risk_score"):
        return 0.03
    return 0.04


def feature_label(feature_key: str) -> str:
    if feature_key.startswith("dim."):
        _, dimension, metric = feature_key.split(".", 2)
        metric_label = metric.replace("_", " ")
        return f"{DIMENSION_LABELS.get(dimension, dimension)} {metric_label}"
    return feature_key.replace("_", " ")


def feature_map_from_report_dict(report_data: dict[str, Any]) -> dict[str, float]:
    features = {
        "overall_atomic_pass_rate": safe_float(report_data.get("overall_atomic_pass_rate")),
        "overall_strict_pass_rate": safe_float(report_data.get("overall_strict_pass_rate")),
        "mean_case_score": safe_float(report_data.get("mean_case_score")),
        "score_stdev": safe_float(report_data.get("score_stdev")),
        "risk_score": safe_float(report_data.get("risk_score")),
    }
    summaries = {
        item.get("dimension"): item
        for item in report_data.get("dimension_summaries", [])
        if isinstance(item, dict) and item.get("dimension")
    }
    for dimension in DIMENSION_LABELS:
        summary = summaries.get(dimension, {})
        features[f"dim.{dimension}.mean_case_score"] = safe_float(summary.get("mean_case_score"))
        features[f"dim.{dimension}.strict_pass_rate"] = safe_float(summary.get("strict_pass_rate"))
        features[f"dim.{dimension}.atomic_pass_rate"] = safe_float(summary.get("atomic_pass_rate"))
    return features


def feature_map_from_report(report: DegradationSuiteReport) -> dict[str, float]:
    return feature_map_from_report_dict(serialize_degradation_report(report))


def collect_report_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(candidate for candidate in path.rglob("*.json") if candidate.is_file()))
        elif path.is_file():
            paths.append(path)
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(resolved)
    return unique_paths


def build_baseline_profile_from_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    profile_name: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    feature_keys: Optional[list[str]] = None
    for artifact in artifacts:
        degradation_data = artifact.get("degradation_report")
        if not isinstance(degradation_data, dict):
            continue
        label = artifact.get("report_label") or f"{artifact.get('provider')}:{artifact.get('model')}"
        feature_map = feature_map_from_report_dict(degradation_data)
        if feature_keys is None:
            feature_keys = sorted(feature_map.keys())
        grouped.setdefault(label, []).append(
            {
                "features": feature_map,
                "provider": str(artifact.get("provider")),
                "model": str(artifact.get("model")),
                "created_at": artifact.get("created_at"),
            }
        )
    if not grouped:
        raise ProbeError("No valid run reports were found to build a baseline profile.")
    feature_keys = feature_keys or []
    groups = []
    for label, items in sorted(grouped.items()):
        means: dict[str, float] = {}
        stdevs: dict[str, float] = {}
        for key in feature_keys:
            values = [safe_float(item["features"].get(key)) for item in items]
            means[key] = statistics.mean(values) if values else 0.0
            stdevs[key] = statistics.pstdev(values) if len(values) > 1 else 0.0
        groups.append(
            {
                "label": label,
                "sample_count": len(items),
                "feature_means": means,
                "feature_stdevs": stdevs,
                "provider_hints": sorted({item["provider"] for item in items if item["provider"]}),
                "model_hints": sorted({item["model"] for item in items if item["model"]}),
            }
        )
    return {
        "schema_version": 1,
        "kind": "transit-probe-baseline-profile",
        "created_at": timestamp_utc(),
        "profile_name": profile_name,
        "feature_keys": feature_keys,
        "report_count": sum(group["sample_count"] for group in groups),
        "groups": groups,
    }


def compare_report_to_profile(
    report: DegradationSuiteReport,
    profile: dict[str, Any],
) -> BaselineComparison:
    feature_map = feature_map_from_report(report)
    groups = profile.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ProbeError("Baseline profile does not contain any groups.")
    feature_keys = [
        key for key in profile.get("feature_keys", []) if key in feature_map
    ] or sorted(feature_map.keys())
    rows: list[BaselineComparisonRow] = []
    for group in groups:
        means = group.get("feature_means", {})
        stdevs = group.get("feature_stdevs", {})
        drifts: list[tuple[float, str]] = []
        squared: list[float] = []
        for key in feature_keys:
            current = feature_map[key]
            mean_value = safe_float(means.get(key))
            scale = max(safe_float(stdevs.get(key)), feature_floor(key))
            z_value = (current - mean_value) / scale if scale else 0.0
            squared.append(z_value * z_value)
            drifts.append(
                (
                    abs(z_value),
                    f"{feature_label(key)} z={z_value:+.2f} "
                    f"(current {current:.3f} vs baseline {mean_value:.3f})",
                )
            )
        distance = math.sqrt(sum(squared) / len(squared)) if squared else float("inf")
        rows.append(
            BaselineComparisonRow(
                label=str(group.get("label")),
                sample_count=safe_int(group.get("sample_count")),
                distance=distance,
                top_drifts=[item[1] for item in sorted(drifts, reverse=True)[:3]],
                provider_hints=[str(item) for item in group.get("provider_hints", [])],
                model_hints=[str(item) for item in group.get("model_hints", [])],
            )
        )
    rows.sort(key=lambda item: item.distance)
    nearest = rows[0]
    return BaselineComparison(
        profile_name=str(profile.get("profile_name", "unnamed")),
        compared_feature_count=len(feature_keys),
        nearest_label=nearest.label,
        nearest_distance=nearest.distance,
        rows=rows,
        note="Lower normalized distance means the current run is closer to that labeled baseline cluster.",
    )


def build_failure_summary(result: DegradationCaseResult) -> str:
    if not result.failed_checks:
        return "perfect"
    return ", ".join(result.failed_checks[:3])


def render_degradation(console: Console, report: DegradationSuiteReport, show_raw: bool) -> None:
    summary = Panel.fit(
        (
            f"Seed: [bold]{report.seed}[/bold]\n"
            f"Cases: [bold]{report.total_cases}[/bold] "
            f"({report.cases_per_dimension} per dimension)\n"
            f"Atomic pass rate: [bold]{report.overall_atomic_pass_rate:.1%}[/bold] "
            f"(95% CI {report.overall_atomic_interval.low:.1%} - {report.overall_atomic_interval.high:.1%})\n"
            f"Strict case pass rate: [bold]{report.overall_strict_pass_rate:.1%}[/bold] "
            f"(95% CI {report.overall_strict_interval.low:.1%} - {report.overall_strict_interval.high:.1%})\n"
            f"Mean case score: [bold]{report.mean_case_score:.1%}[/bold]\n"
            f"Case score stdev: [bold]{report.score_stdev:.3f}[/bold]\n"
            f"Risk: [bold]{report.risk_level}[/bold] "
            f"(composite {report.risk_score:.3f})\n"
            f"Verdict: {report.verdict}\n"
            f"Calibration: {report.calibration_note}"
        ),
        title="Degradation Suite",
        border_style="magenta",
    )
    console.print(summary)

    dim_table = Table(title="Dimension Breakdown")
    dim_table.add_column("Dimension", style="bold")
    dim_table.add_column("Cases", justify="right")
    dim_table.add_column("Mean Score", justify="right")
    dim_table.add_column("Strict Pass", justify="right")
    dim_table.add_column("Atomic 95% CI", justify="right")
    dim_table.add_column("Weak Cases")
    for summary in report.dimension_summaries:
        dim_table.add_row(
            DIMENSION_LABELS[summary.dimension],
            str(summary.case_count),
            f"{summary.mean_case_score:.1%}",
            f"{summary.strict_pass_rate:.1%}",
            f"{summary.atomic_interval.low:.1%} - {summary.atomic_interval.high:.1%}",
            ", ".join(summary.weak_cases[:4]) if summary.weak_cases else "-",
        )
    console.print(dim_table)

    failure_table = Table(title="Worst Cases")
    failure_table.add_column("Case", style="bold")
    failure_table.add_column("Dimension")
    failure_table.add_column("Score", justify="right")
    failure_table.add_column("TTFT", justify="right")
    failure_table.add_column("Failures")
    worst_cases = sorted(report.case_results, key=lambda item: (item.score_rate, item.metrics.first_byte_time_ms))
    for result in worst_cases[: min(8, len(worst_cases))]:
        failure_table.add_row(
            result.case_id,
            DIMENSION_LABELS[result.dimension],
            f"{result.score_rate:.0%}",
            f"{result.metrics.first_byte_time_ms:.0f} ms",
            build_failure_summary(result),
        )
    console.print(failure_table)

    if show_raw:
        failing_cases = [result for result in worst_cases if not result.strict_pass][:2]
        for result in failing_cases:
            console.print(
                Panel(
                    f"Prompt:\n{truncate(result.prompt, 900)}\n\n"
                    f"Expected:\n{truncate(result.reference_output, 700)}\n\n"
                    f"Observed:\n{truncate(result.raw_text, 700)}",
                    title=f"{result.case_id} Detail",
                    border_style="bright_black",
                )
            )


def render_baseline_comparison(console: Console, comparison: BaselineComparison) -> None:
    console.print(
        Panel.fit(
            f"Profile: [bold]{comparison.profile_name}[/bold]\n"
            f"Nearest label: [bold]{comparison.nearest_label}[/bold]\n"
            f"Normalized distance: [bold]{comparison.nearest_distance:.3f}[/bold]\n"
            f"Compared features: [bold]{comparison.compared_feature_count}[/bold]\n"
            f"Note: {comparison.note}",
            title="Baseline Comparison",
            border_style="cyan",
        )
    )
    table = Table(title="Baseline Distance Ranking")
    table.add_column("Label", style="bold")
    table.add_column("Samples", justify="right")
    table.add_column("Distance", justify="right")
    table.add_column("Models")
    table.add_column("Top Drift Signals")
    for row in comparison.rows[: min(6, len(comparison.rows))]:
        table.add_row(
            row.label,
            str(row.sample_count),
            f"{row.distance:.3f}",
            ", ".join(row.model_hints[:3]) if row.model_hints else "-",
            " | ".join(row.top_drifts),
        )
    console.print(table)


def render_baseline_profile(console: Console, profile: dict[str, Any], output_path: str) -> None:
    console.print(
        Panel.fit(
            f"Profile: [bold]{profile.get('profile_name', 'unnamed')}[/bold]\n"
            f"Reports: [bold]{profile.get('report_count', 0)}[/bold]\n"
            f"Groups: [bold]{len(profile.get('groups', []))}[/bold]\n"
            f"Saved to: [bold]{output_path}[/bold]",
            title="Baseline Profile",
            border_style="green",
        )
    )
    table = Table(title="Baseline Groups")
    table.add_column("Label", style="bold")
    table.add_column("Samples", justify="right")
    table.add_column("Providers")
    table.add_column("Models")
    for group in profile.get("groups", []):
        table.add_row(
            str(group.get("label")),
            str(group.get("sample_count")),
            ", ".join(str(item) for item in group.get("provider_hints", [])[:3]) or "-",
            ", ".join(str(item) for item in group.get("model_hints", [])[:3]) or "-",
        )
    console.print(table)


def build_cache_probe_text(run_id: str, target_chars: int = 32000) -> str:
    lines: list[str] = [
        f"cache probe run id {run_id}.",
        "This prefix is intentionally large and stable.",
        "It exists only to measure prompt caching behavior.",
    ]
    index = 0
    while len("\n".join(lines)) < target_chars:
        lines.append(
            f"section {index:04d}. "
            f"stable technical context for latency measurement {index:04d}. "
            f"deterministic prefix block for cache inspection {index:04d}."
        )
        index += 1
    return "\n".join(lines)


def extract_openai_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "output_text"} and item.get("text"):
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
        return "\n".join(part.strip() for part in parts if part.strip())
    return str(content).strip()


def extract_anthropic_text(payload: dict[str, Any]) -> str:
    content = payload.get("content") or []
    if isinstance(content, str):
        return content.strip()
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
            parts.append(str(block["text"]).strip())
    return "\n".join(part for part in parts if part)


def extract_usage(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage")
    return usage if isinstance(usage, dict) else {}


def extract_error_detail(payload: Optional[dict[str, Any]], fallback: bytes) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            if error.get("message"):
                return str(error["message"])
            return json.dumps(error, ensure_ascii=False)
        if payload.get("detail"):
            return str(payload["detail"])
        if payload.get("message"):
            return str(payload["message"])
    decoded = fallback.decode("utf-8", errors="replace").strip()
    return truncate(decoded, 400) or "No error payload returned."


def is_openai_cache_hint_error(detail: str) -> bool:
    lowered = detail.lower()
    return "prompt_cache" in lowered or "unknown parameter" in lowered


class TransitProbeClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        provider: Provider,
        timeout: float,
        model: Optional[str],
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.timeout = timeout
        self.model = model
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    def headers(self) -> dict[str, str]:
        if self.provider is Provider.OPENAI:
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def completion_endpoint(self) -> str:
        return "chat/completions" if self.provider is Provider.OPENAI else "messages"

    async def list_models(self) -> list[str]:
        for url in candidate_urls(self.base_url, "models"):
            try:
                response = await self._client.get(url, headers=self.headers())
            except httpx.HTTPError:
                continue
            if response.status_code == 404:
                continue
            try:
                payload = response.json()
            except json.JSONDecodeError:
                payload = {}
            if response.is_success:
                data = payload.get("data") or []
                models = [
                    str(item.get("id"))
                    for item in data
                    if isinstance(item, dict) and item.get("id")
                ]
                if models:
                    return models
        return []

    def choose_model(self, models: Iterable[str]) -> str:
        available = list(models)
        if self.model:
            return self.model
        preferred = (
            [
                "gpt-5.4",
                "gpt-5.2",
                "gpt-5.1",
                "gpt-5",
                "gpt-4.1",
                "gpt-4o",
                "o4",
                "claude-opus-4",
                "claude-sonnet-4",
                "claude-3-7-sonnet",
                "claude-3-5-sonnet",
            ]
            if self.provider is Provider.OPENAI
            else [
                "claude-opus-4-7",
                "claude-opus-4",
                "claude-sonnet-4-5",
                "claude-sonnet-4",
                "claude-3-7-sonnet",
                "claude-3-5-sonnet",
                "claude-3-5-haiku",
            ]
        )
        lowered_pairs = [(item.lower(), item) for item in available]
        for pattern in preferred:
            for lowered, original in lowered_pairs:
                if pattern in lowered:
                    return original
        if available:
            return sorted(available)[0]
        if self.provider is Provider.OPENAI:
            return "gpt-4o"
        return "claude-3-5-sonnet-latest"

    async def resolve_model(self) -> str:
        models = await self.list_models()
        self.model = self.choose_model(models)
        return self.model

    async def timed_post(self, payload: dict[str, Any]) -> TimedResponse:
        last_error: Optional[Exception] = None
        for url in candidate_urls(self.base_url, self.completion_endpoint()):
            start_ns = perf_counter_ns()
            try:
                async with self._client.stream(
                    "POST",
                    url,
                    headers=self.headers(),
                    json=payload,
                ) as response:
                    header_ns = perf_counter_ns()
                    chunks: list[bytes] = []
                    first_byte_ns: Optional[int] = None
                    async for chunk in response.aiter_bytes():
                        if chunk and first_byte_ns is None:
                            first_byte_ns = perf_counter_ns()
                        chunks.append(chunk)
                    if first_byte_ns is None:
                        first_byte_ns = perf_counter_ns()
                    end_ns = perf_counter_ns()
            except httpx.HTTPError as exc:
                last_error = exc
                continue
            body = b"".join(chunks)
            try:
                parsed = json.loads(body.decode("utf-8", errors="replace"))
                json_body = parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                json_body = None
            if response.status_code in {404, 405}:
                last_error = ProbeError(f"{url} returned HTTP {response.status_code}")
                continue
            return TimedResponse(
                url=url,
                status_code=response.status_code,
                header_time_ms=ms_from_ns(start_ns, header_ns),
                first_byte_time_ms=ms_from_ns(start_ns, first_byte_ns),
                total_time_ms=ms_from_ns(start_ns, end_ns),
                body=body,
                json_body=json_body,
            )
        if last_error is not None:
            raise ProbeError(f"Request failed on all candidate endpoints: {last_error}") from last_error
        raise ProbeError("Request failed before reaching any candidate endpoint.")

    async def invoke(
        self,
        payload: dict[str, Any],
        *,
        retry_without_openai_cache_hints: bool = False,
    ) -> ApiCallResult:
        timed = await self.timed_post(payload)
        if timed.status_code >= 400:
            detail = extract_error_detail(timed.json_body, timed.body)
            if (
                retry_without_openai_cache_hints
                and self.provider is Provider.OPENAI
                and is_openai_cache_hint_error(detail)
            ):
                slimmed = dict(payload)
                slimmed.pop("prompt_cache_key", None)
                slimmed.pop("prompt_cache_retention", None)
                timed = await self.timed_post(slimmed)
                if timed.status_code >= 400:
                    retry_detail = extract_error_detail(timed.json_body, timed.body)
                    raise ProbeError(
                        f"{timed.url} returned HTTP {timed.status_code}: {retry_detail}"
                    )
            else:
                raise ProbeError(f"{timed.url} returned HTTP {timed.status_code}: {detail}")
        if timed.json_body is None:
            body_preview = truncate(timed.body.decode("utf-8", errors="replace"), 400)
            raise ProbeError(
                f"{timed.url} returned a non-JSON body. Preview: {body_preview or '<empty>'}"
            )
        raw = timed.json_body
        text = (
            extract_openai_text(raw)
            if self.provider is Provider.OPENAI
            else extract_anthropic_text(raw)
        )
        return ApiCallResult(text=text, usage=extract_usage(raw), raw=raw, metrics=timed)

    def build_benchmark_payload(self, model: str, definition: DegradationCaseDefinition) -> dict[str, Any]:
        if self.provider is Provider.OPENAI:
            return {
                "model": model,
                "temperature": 0,
                "max_tokens": definition.max_tokens,
                "messages": [
                    {"role": "system", "content": BENCHMARK_SYSTEM_PROMPT},
                    {"role": "user", "content": definition.prompt},
                ],
            }
        return {
            "model": model,
            "temperature": 0,
            "max_tokens": definition.max_tokens,
            "system": BENCHMARK_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": definition.prompt}],
        }

    def build_cache_payload(self, model: str, run_id: str) -> dict[str, Any]:
        large_prefix = build_cache_probe_text(run_id)
        followup = "Reply with the single word READY."
        if self.provider is Provider.OPENAI:
            return {
                "model": model,
                "temperature": 0,
                "max_tokens": 32,
                "prompt_cache_key": f"transit-probe-{run_id}",
                "prompt_cache_retention": "in_memory",
                "messages": [
                    {"role": "system", "content": large_prefix},
                    {"role": "user", "content": followup},
                ],
            }
        return {
            "model": model,
            "temperature": 0,
            "max_tokens": 32,
            "system": [
                {
                    "type": "text",
                    "text": large_prefix,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": followup}],
                }
            ],
        }


async def invoke_with_retries(
    client: TransitProbeClient,
    payload: dict[str, Any],
    *,
    retries: int,
    retry_without_openai_cache_hints: bool = False,
) -> ApiCallResult:
    last_exc: Optional[ProbeError] = None
    for attempt in range(1, retries + 1):
        try:
            return await client.invoke(
                payload,
                retry_without_openai_cache_hints=retry_without_openai_cache_hints,
            )
        except ProbeError as exc:
            last_exc = exc
            if attempt == retries:
                break
            await asyncio.sleep(min(4.0, 0.8 * attempt))
    if last_exc is not None:
        raise last_exc
    raise ProbeError("Request retry loop exited unexpectedly.")


def extract_cache_indicator(provider: Provider, usage: dict[str, Any]) -> tuple[str, int]:
    if provider is Provider.OPENAI:
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached_tokens = safe_int(details.get("cached_tokens"))
            if cached_tokens > 0:
                return "usage.prompt_tokens_details.cached_tokens", cached_tokens
        top_level = safe_int(usage.get("cached_tokens"))
        if top_level > 0:
            return "usage.cached_tokens", top_level
        response_style = usage.get("input_tokens_details")
        if isinstance(response_style, dict):
            cached_tokens = safe_int(response_style.get("cached_tokens"))
            if cached_tokens > 0:
                return "usage.input_tokens_details.cached_tokens", cached_tokens
        return "usage.prompt_tokens_details.cached_tokens", 0
    return "usage.cache_read_input_tokens", safe_int(usage.get("cache_read_input_tokens"))


def analyze_cache(
    provider: Provider,
    first: ApiCallResult,
    second: ApiCallResult,
) -> CacheReport:
    indicator_name, indicator_value = extract_cache_indicator(provider, second.usage)
    payload_hit = indicator_value > 0
    delta_ms = first.metrics.first_byte_time_ms - second.metrics.first_byte_time_ms
    ratio = (
        second.metrics.first_byte_time_ms / first.metrics.first_byte_time_ms
        if first.metrics.first_byte_time_ms > 0
        else 1.0
    )
    cliff_drop = (
        first.metrics.first_byte_time_ms >= 1000
        and delta_ms >= 800
        and ratio <= 0.35
    )
    if payload_hit and cliff_drop:
        verdict = "Real cache detected."
        notes = "Usage metrics show cached input and TTFT dropped sharply on the second request."
    elif payload_hit and not cliff_drop:
        verdict = "Cache flag present but latency did not improve enough."
        notes = "This looks like a forged or ineffective cache signal."
    elif not payload_hit and cliff_drop:
        verdict = "Latency dropped but no cache usage field was reported."
        notes = "This may be network jitter or a non-standard gateway that hides usage details."
    else:
        verdict = "No convincing cache evidence."
        notes = "The second request neither exposed a cache hit field nor showed a cliff-like TTFT drop."
    return CacheReport(
        first=first,
        second=second,
        payload_indicator_name=indicator_name,
        payload_indicator_value=indicator_value,
        payload_hit=payload_hit,
        cliff_drop=cliff_drop,
        delta_ms=delta_ms,
        ratio=ratio,
        verdict=verdict,
        notes=notes,
    )


def render_cache(console: Console, report: CacheReport, provider: Provider, show_raw: bool) -> None:
    table = Table(title="Prompt Caching Test")
    table.add_column("Metric", style="bold")
    table.add_column("Request 1", justify="right")
    table.add_column("Request 2", justify="right")
    table.add_column("Delta", justify="right")
    table.add_row(
        "TTFT",
        f"{report.first.metrics.first_byte_time_ms:.2f} ms",
        f"{report.second.metrics.first_byte_time_ms:.2f} ms",
        f"{report.delta_ms:.2f} ms",
    )
    table.add_row(
        "Headers received",
        f"{report.first.metrics.header_time_ms:.2f} ms",
        f"{report.second.metrics.header_time_ms:.2f} ms",
        "-",
    )
    table.add_row(
        "Total request time",
        f"{report.first.metrics.total_time_ms:.2f} ms",
        f"{report.second.metrics.total_time_ms:.2f} ms",
        "-",
    )
    table.add_row(
        report.payload_indicator_name,
        "-",
        str(report.payload_indicator_value),
        "-",
    )
    if provider is Provider.ANTHROPIC:
        table.add_row(
            "usage.cache_creation_input_tokens",
            str(safe_int(report.first.usage.get("cache_creation_input_tokens"))),
            str(safe_int(report.second.usage.get("cache_creation_input_tokens"))),
            "-",
        )
    console.print(table)
    console.print(
        Panel.fit(
            f"TTFT ratio: [bold]{report.ratio:.2f}x[/bold]\n"
            f"Payload cache hit: [bold]{'yes' if report.payload_hit else 'no'}[/bold]\n"
            f"Cliff drop detected: [bold]{'yes' if report.cliff_drop else 'no'}[/bold]\n"
            f"Verdict: {report.verdict}\n"
            f"Notes: {report.notes}",
            border_style="green" if report.payload_hit and report.cliff_drop else "yellow",
        )
    )
    if show_raw:
        console.print(
            Panel(
                truncate(json.dumps(report.second.raw, ensure_ascii=False, indent=2), 1600),
                title="Second Response JSON Preview",
                border_style="bright_black",
            )
        )


def render_intro(
    console: Console,
    provider: Provider,
    model: str,
    base_url: str,
    *,
    config_path: Optional[str] = None,
) -> None:
    lines = [
        f"Provider: [bold]{provider.value}[/bold]",
        f"Model: [bold]{model}[/bold]",
        f"Base URL: [bold]{normalize_base_url(base_url)}[/bold]",
    ]
    if config_path:
        lines.append(f"Config: [bold]{config_path}[/bold]")
    console.print(
        Panel.fit(
            "\n".join(lines),
            title="Transit Probe",
            border_style="cyan",
        )
    )


async def run_degradation_suite(
    console: Console,
    client: TransitProbeClient,
    model: str,
    args: argparse.Namespace,
) -> DegradationSuiteReport:
    seed = args.seed if args.seed is not None else random.SystemRandom().randint(10_000, 999_999_999)
    suite = build_degradation_suite(
        seed=seed,
        cases_per_dimension=args.cases_per_dimension,
        context_record_count=args.context_records,
    )
    case_results: list[DegradationCaseResult] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("Running degradation suite", total=len(suite))
        for definition in suite:
            payload = client.build_benchmark_payload(model, definition)
            response = await invoke_with_retries(client, payload, retries=args.request_retries)
            case_results.append(evaluate_case_result(definition, response))
            progress.advance(task_id)
    return aggregate_degradation_suite(
        seed=seed,
        cases_per_dimension=args.cases_per_dimension,
        case_results=case_results,
    )


def build_baseline_mode(args: argparse.Namespace) -> int:
    console = Console()
    report_paths = collect_report_paths(args.build_baseline)
    if not report_paths:
        console.print(
            Panel.fit(
                "No JSON run reports were found in the provided paths.",
                title="Baseline Error",
                border_style="red",
            )
        )
        return 1
    artifacts: list[dict[str, Any]] = []
    invalid_paths: list[str] = []
    for path in report_paths:
        try:
            payload = read_json_file(path)
        except (OSError, json.JSONDecodeError):
            invalid_paths.append(str(path))
            continue
        if payload.get("kind") != "transit-probe-run-report":
            invalid_paths.append(str(path))
            continue
        artifacts.append(payload)
    if not artifacts:
        console.print(
            Panel.fit(
                "None of the provided JSON files matched the transit probe run report format.",
                title="Baseline Error",
                border_style="red",
            )
        )
        return 1
    try:
        profile = build_baseline_profile_from_artifacts(
            artifacts,
            profile_name=args.baseline_name or "default",
        )
        write_json_file(args.baseline_out, profile)
    except (ProbeError, OSError) as exc:
        console.print(Panel.fit(str(exc), title="Baseline Error", border_style="red"))
        return 1
    render_baseline_profile(console, profile, args.baseline_out)
    if invalid_paths:
        console.print(
            Panel.fit(
                "Skipped files:\n" + "\n".join(invalid_paths[:12]),
                title="Ignored Inputs",
                border_style="yellow",
            )
        )
    return 0


async def run_probe(args: argparse.Namespace) -> int:
    console = Console()
    client = TransitProbeClient(
        base_url=args.base_url,
        api_key=args.api_key,
        provider=Provider(args.provider),
        timeout=args.timeout,
        model=args.model,
    )
    try:
        with console.status("Resolving model...", spinner="dots"):
            model = await client.resolve_model()
        render_intro(
            console,
            client.provider,
            model,
            args.base_url,
            config_path=args.config_resolved_path,
        )

        degradation_report = await run_degradation_suite(console, client, model, args)
        render_degradation(console, degradation_report, args.show_raw)
        baseline_comparison: Optional[BaselineComparison] = None
        if args.baseline_profile:
            try:
                baseline_payload = read_json_file(Path(args.baseline_profile))
            except (OSError, json.JSONDecodeError) as exc:
                raise ProbeError(f"Failed to read baseline profile: {exc}") from exc
            if baseline_payload.get("kind") != "transit-probe-baseline-profile":
                raise ProbeError("The baseline profile file is not a valid transit probe baseline profile.")
            baseline_comparison = compare_report_to_profile(degradation_report, baseline_payload)
            render_baseline_comparison(console, baseline_comparison)

        cache_report: Optional[CacheReport] = None
        if not args.skip_cache:
            run_id = uuid.uuid4().hex[:12]
            cache_payload = client.build_cache_payload(model, run_id)
            with console.status("Running cache probe request 1...", spinner="dots"):
                first = await invoke_with_retries(
                    client,
                    cache_payload,
                    retries=args.request_retries,
                    retry_without_openai_cache_hints=True,
                )
            await asyncio.sleep(2)
            with console.status("Running cache probe request 2...", spinner="dots"):
                second = await invoke_with_retries(
                    client,
                    cache_payload,
                    retries=args.request_retries,
                    retry_without_openai_cache_hints=True,
                )
            cache_report = analyze_cache(client.provider, first, second)
            render_cache(console, cache_report, client.provider, args.show_raw)

        if args.report_out:
            artifact = build_run_artifact(
                args=args,
                provider=client.provider,
                model=model,
                degradation_report=degradation_report,
                cache_report=cache_report,
                baseline_comparison=baseline_comparison,
            )
            try:
                write_json_file(args.report_out, artifact)
            except OSError as exc:
                raise ProbeError(f"Failed to write run report: {exc}") from exc
            console.print(
                Panel.fit(
                    f"Saved run report to [bold]{args.report_out}[/bold]",
                    title="Report Export",
                    border_style="green",
                )
            )

        console.print(Panel.fit("The probe completed successfully.", border_style="blue"))
        return 0
    except ProbeError as exc:
        console.print(Panel.fit(str(exc), title="Probe Error", border_style="red"))
        return 1
    finally:
        await client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="API quality probe for model degradation and native prompt caching.",
    )
    parser.add_argument(
        "--config",
        help=f"Optional config file path. If omitted the tool will auto-load {DEFAULT_CONFIG_PATH} when present.",
    )
    parser.add_argument("--base-url", help="Transit API base URL.")
    parser.add_argument("--api-key", help="Transit API key.")
    parser.add_argument(
        "--provider",
        choices=[provider.value for provider in Provider],
        default=None,
        help=f"Request protocol to use. Default: {DEFAULT_PROVIDER}.",
    )
    parser.add_argument(
        "--model",
        help="Optional model name. If omitted the tool will try /models and auto-select one.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=f"HTTP timeout in seconds. Default: {DEFAULT_TIMEOUT}.",
    )
    parser.add_argument(
        "--cases-per-dimension",
        type=int,
        default=None,
        help=f"How many randomized degradation cases to run per dimension. Default: {DEFAULT_CASES_PER_DIMENSION}.",
    )
    parser.add_argument(
        "--context-records",
        type=int,
        default=None,
        help=f"Record count for each long-context retrieval case. Default: {DEFAULT_CONTEXT_RECORDS}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for reproducible degradation suites.",
    )
    parser.add_argument(
        "--request-retries",
        type=int,
        default=None,
        help=f"How many times to retry a failed API call. Default: {DEFAULT_REQUEST_RETRIES}.",
    )
    parser.add_argument(
        "--skip-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run only the degradation suite and skip the prompt cache probe.",
    )
    parser.add_argument(
        "--show-raw",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show truncated raw model outputs and response JSON previews.",
    )
    parser.add_argument(
        "--report-out",
        help="Write a structured run report JSON to this path after a successful probe.",
    )
    parser.add_argument(
        "--report-label",
        help="Optional label stored in exported run reports for later baseline grouping.",
    )
    parser.add_argument(
        "--baseline-profile",
        help="Optional baseline profile JSON to compare against the current run.",
    )
    parser.add_argument(
        "--build-baseline",
        nargs="+",
        help="Offline mode. Build a baseline profile from one or more exported run report JSON files or directories.",
    )
    parser.add_argument(
        "--baseline-out",
        help="Output path for --build-baseline mode.",
    )
    parser.add_argument(
        "--baseline-name",
        help="Optional name embedded in a generated baseline profile.",
    )
    return parser


async def async_main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.build_baseline:
        if not args.baseline_out:
            parser.error("--baseline-out is required with --build-baseline")
        return build_baseline_mode(args)
    try:
        args = resolve_runtime_args(args)
    except ProbeError as exc:
        parser.error(str(exc))
    if not args.base_url:
        parser.error("--base-url is required unless --build-baseline is used")
    if not args.api_key:
        parser.error("--api-key is required unless --build-baseline is used")
    if args.provider not in {provider.value for provider in Provider}:
        parser.error(f"--provider must be one of: {', '.join(provider.value for provider in Provider)}")
    if args.cases_per_dimension < 1:
        parser.error("--cases-per-dimension must be >= 1")
    if args.context_records < 8:
        parser.error("--context-records must be >= 8")
    if args.request_retries < 1:
        parser.error("--request-retries must be >= 1")
    return await run_probe(args)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))

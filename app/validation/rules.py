"""Rule-based validation: 8 fast checks (~50ms, no LLM required)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from loguru import logger


# ── Bloom verb banks (for bloom_verb_presence check) ──────────────────────────

BLOOM_VERBS_BY_LEVEL: dict[str, List[str]] = {
    "REMEMBER":   ["define", "list", "recall", "name", "identify", "state", "memorise", "memorize"],
    "UNDERSTAND": ["explain", "describe", "summarise", "summarize", "paraphrase", "interpret", "classify", "give examples"],
    "APPLY":      ["solve", "use", "demonstrate", "calculate", "apply", "carry out", "execute", "implement"],
    "ANALYZE":    ["analyse", "analyze", "compare", "contrast", "distinguish", "examine", "break down", "differentiate"],
    "EVALUATE":   ["evaluate", "justify", "assess", "critique", "judge", "argue", "defend"],
    "CREATE":     ["design", "create", "compose", "construct", "develop", "produce", "plan"],
}

# ── Profanity / explicit content blocklist (non-exhaustive) ───────────────────

EXPLICIT_BLOCKLIST = [
    r"\bfuck\b", r"\bshit\b", r"\bporn\b", r"\bsex\b(?!ual\s+health|\s+education)",
    r"\bpenis\b(?!\s+(?:development|anatomy|health))", r"\bvagina\b(?!\s+(?:anatomy|health))",
    r"\bnude\b", r"\bnaked\b",
]

# ── Cultural flag: Western names that dilute local authenticity ────────────────

WESTERN_NAMES = {
    "john", "james", "peter", "michael", "david", "william", "robert", "richard",
    "thomas", "charles", "george", "edward", "henry", "joseph", "paul",
    "mary", "jennifer", "jessica", "emily", "sarah", "elizabeth", "lisa",
    "susan", "karen", "nancy", "betty", "helen", "sandra", "donna",
    "matthew", "andrew", "daniel", "christopher", "mark", "joshua", "ryan",
    "kevin", "brian", "gary", "timothy", "jason", "jeff", "frank",
}

LOCAL_NAMES = {
    "chukwuemeka", "adaeze", "aminu", "tunde", "kofi", "ama", "fatima",
    "emeka", "ngozi", "kwame", "abena", "bola", "sola", "kemi", "biodun",
    "seun", "yetunde", "taiwo", "kehinde", "femi", "toyin", "chioma",
    "nkechi", "uchenna", "obiora", "chiamaka", "obinna", "chidi", "aisha",
    "musa", "ibrahim", "halima", "zainab", "binta", "garba", "yusuf",
    "efua", "akosua", "adjoa", "esi", "yaw", "kojo", "ama", "akua",
    "nana", "afia", "ekua", "mensah", "asante", "owusu", "boateng",
}

# ── Curriculum alignment keywords (per board) ─────────────────────────────────

CURRICULUM_KEYWORDS = {
    "NERDC":  ["objective", "activity", "evaluation", "term", "week", "topic", "learning", "assessment"],
    "WAEC":   ["objective", "question", "mark", "answer", "examination", "candidate", "section"],
    "NECO":   ["objective", "section", "question", "answer", "mark", "examination"],
    "NABTEB": ["objective", "practical", "trade", "skill", "competency", "assessment"],
    "UBEC":   ["objective", "activity", "term", "week", "learning", "lesson"],
    "GES_GH": ["objective", "activity", "term", "week", "learning", "lesson", "indicator"],
    "DEFAULT":["objective", "topic", "learning", "assessment", "activity"],
}

# ── Date hallucination check ───────────────────────────────────────────────────

YEAR_PATTERN = re.compile(r"\b(1[0-7]\d{2}|20[3-9]\d|2[1-9]\d{2})\b")  # outside [1800-2030]


@dataclass
class RuleResult:
    rule_name: str
    passed: bool
    message: str = ""
    is_hard_fail: bool = False   # hard fails auto-fail regardless of judge


@dataclass
class ValidationRulesReport:
    passed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    hard_failed: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return not self.failed and not self.hard_failed

    @property
    def has_hard_fail(self) -> bool:
        return bool(self.hard_failed)


# ── Individual rule functions ──────────────────────────────────────────────────

def check_length(content: str, max_tokens: int) -> RuleResult:
    """Content must be ≥ 25% and ≤ 600% of max_tokens (in characters)."""
    char_count = len(content)
    lower = max_tokens * 0.25 * 4  # approx chars
    upper = max_tokens * 6.0 * 4
    if char_count < lower:
        return RuleResult("length_check", False, f"Content too short ({char_count} chars; min ~{int(lower)})")
    if char_count > upper:
        return RuleResult("length_check", False, f"Content too long ({char_count} chars; max ~{int(upper)})")
    return RuleResult("length_check", True)


def check_language_detection(
    content: str,
    expected_lang: str,
    threshold: float = 0.85,
) -> RuleResult:
    """Detect language and fail if confidence > threshold for wrong language."""
    try:
        from lingua import Language as LinguaLang, LanguageDetectorBuilder  # type: ignore
        detector = (
            LanguageDetectorBuilder.from_all_languages()
            .with_low_accuracy_mode()
            .build()
        )
        result = detector.detect_language_of(content[:1000])
        if result is None:
            return RuleResult("language_detection", True, "Language could not be determined; skipping")
        detected = result.name.lower()
        lang_map = {"en": "english", "yo": "yoruba", "ha": "hausa", "ig": "igbo", "pcm": "english"}
        expected_name = lang_map.get(expected_lang, expected_lang.lower())

        if expected_name not in detected and detected not in expected_name:
            conf_result = detector.compute_language_confidence(content[:1000], result)
            if conf_result > threshold:
                return RuleResult(
                    "language_detection", False,
                    f"Language mismatch: expected '{expected_lang}', detected '{detected}' (conf {conf_result:.2f})"
                )
    except ImportError:
        # Fall back to langdetect
        try:
            from langdetect import detect  # type: ignore
            detected = detect(content[:1000])
            lang_map = {"en": "en", "yo": "yo", "ha": "ha", "ig": "ig", "pcm": "en"}
            expected_code = lang_map.get(expected_lang, expected_lang.split("-")[0])
            if detected != expected_code and not expected_lang.startswith("en-"):
                return RuleResult(
                    "language_detection", False,
                    f"Language mismatch: expected '{expected_lang}', detected '{detected}'"
                )
        except Exception as exc:
            logger.warning(f"Language detection skipped: {exc}")

    return RuleResult("language_detection", True)


def check_bloom_verbs(
    content: str,
    bloom_level: str,
) -> RuleResult:
    """At least one Bloom-level verb must appear in the content."""
    verbs = BLOOM_VERBS_BY_LEVEL.get(bloom_level.upper(), [])
    if not verbs:
        return RuleResult("bloom_verb_presence", True, "No verb list for bloom level; skipping")
    lower = content.lower()
    found = [v for v in verbs if v in lower]
    if not found:
        return RuleResult(
            "bloom_verb_presence", False,
            f"No {bloom_level} Bloom verbs found. Expected one of: {', '.join(verbs[:5])}"
        )
    return RuleResult("bloom_verb_presence", True, f"Bloom verbs found: {', '.join(found[:3])}")


def check_cultural_flags(
    content: str,
    use_local_names: bool = True,
    western_ratio_threshold: float = 0.6,
) -> RuleResult:
    """Flag if Western name ratio > threshold when local names expected."""
    if not use_local_names:
        return RuleResult("cultural_flag_check", True, "Local names not required")

    words = re.findall(r"\b[A-Z][a-z]+\b", content)
    names = [w for w in words if w.lower() in WESTERN_NAMES or w.lower() in LOCAL_NAMES]

    if not names:
        return RuleResult("cultural_flag_check", True, "No recognisable names found; skipping")

    western_count = sum(1 for n in names if n.lower() in WESTERN_NAMES)
    ratio = western_count / len(names)

    if ratio > western_ratio_threshold:
        return RuleResult(
            "cultural_flag_check", False,
            f"Western name ratio {ratio:.0%} exceeds {western_ratio_threshold:.0%} threshold. "
            f"Found Western names: {list(set(n for n in names if n.lower() in WESTERN_NAMES))[:5]}"
        )
    return RuleResult("cultural_flag_check", True, f"Name diversity OK (Western ratio: {ratio:.0%})")


def check_format_compliance(
    content: str,
    content_type: str,
    num_questions: Optional[int] = None,
) -> RuleResult:
    """Check structural compliance based on content type."""
    ct = content_type.upper()

    if ct in {"QUIZ", "EXAM_QUESTIONS", "QUESTION_BANK", "DIAGNOSTIC_TEST"}:
        # Should have question marks or numbered questions
        question_marks = content.count("?")
        numbered_q = len(re.findall(r"^\s*Q?\d+[\.\)]\s", content, re.MULTILINE))
        total_q = question_marks + numbered_q
        if num_questions and total_q < max(1, num_questions // 2):
            return RuleResult(
                "format_compliance", False,
                f"Assessment has only {total_q} questions; expected ~{num_questions}"
            )

    elif ct == "LESSON_PLAN":
        required_sections = ["objective", "activity", "assessment"]
        lower = content.lower()
        missing = [s for s in required_sections if s not in lower]
        if missing:
            return RuleResult(
                "format_compliance", False,
                f"Lesson plan missing sections: {', '.join(missing)}"
            )

    elif ct in {"SCHEME_OF_WORK", "TERM_PLAN", "SCOPE_AND_SEQUENCE"}:
        # Should reference weeks or terms
        if not re.search(r"week\s*\d+|term\s*\d+|w\d+\s*[:\-]", content, re.IGNORECASE):
            return RuleResult(
                "format_compliance", False,
                "Curriculum plan lacks week/term structure"
            )

    return RuleResult("format_compliance", True)


def check_no_hallucinated_dates(content: str) -> RuleResult:
    """Flag years outside [1800–2030] as potential hallucinations."""
    bad_years = YEAR_PATTERN.findall(content)
    if bad_years:
        return RuleResult(
            "no_hallucinated_dates", False,
            f"Suspicious years found: {bad_years[:5]}",
        )
    return RuleResult("no_hallucinated_dates", True)


def check_no_explicit_content(content: str) -> RuleResult:
    """Hard-fail if explicit/profane content is detected."""
    lower = content.lower()
    for pattern in EXPLICIT_BLOCKLIST:
        if re.search(pattern, lower):
            return RuleResult(
                "no_explicit_content", False,
                f"Explicit content detected (pattern: {pattern})",
                is_hard_fail=True,
            )
    return RuleResult("no_explicit_content", True)


def check_curriculum_alignment(
    content: str,
    curriculum_board: str = "NERDC",
    min_keywords: int = 3,
) -> RuleResult:
    """At least min_keywords curriculum terms must appear."""
    keywords = CURRICULUM_KEYWORDS.get(curriculum_board.upper(), CURRICULUM_KEYWORDS["DEFAULT"])
    lower = content.lower()
    found = [kw for kw in keywords if kw in lower]
    if len(found) < min_keywords:
        return RuleResult(
            "curriculum_alignment", False,
            f"Only {len(found)}/{min_keywords} curriculum terms found: {found}"
        )
    return RuleResult("curriculum_alignment", True, f"Curriculum terms found: {found[:5]}")


# ── Master runner ──────────────────────────────────────────────────────────────

def run_all_rules(
    content: str,
    *,
    max_tokens: int = 1024,
    expected_language: str = "en",
    bloom_level: str = "UNDERSTAND",
    use_local_names: bool = True,
    content_type: str = "LESSON_PLAN",
    curriculum_board: str = "NERDC",
    num_questions: Optional[int] = None,
) -> ValidationRulesReport:
    """Run all 8 validation rules and return a consolidated report."""
    report = ValidationRulesReport()

    checks = [
        check_length(content, max_tokens),
        check_language_detection(content, expected_language),
        check_bloom_verbs(content, bloom_level),
        check_cultural_flags(content, use_local_names),
        check_format_compliance(content, content_type, num_questions),
        check_no_hallucinated_dates(content),
        check_no_explicit_content(content),
        check_curriculum_alignment(content, curriculum_board),
    ]

    for result in checks:
        if result.passed:
            report.passed.append(result.rule_name)
        else:
            if result.is_hard_fail:
                report.hard_failed.append(result.rule_name)
            else:
                report.failed.append(result.rule_name)
            if result.message:
                report.notes.append(f"{result.rule_name}: {result.message}")

    logger.info(
        f"Rules: {len(report.passed)} passed, {len(report.failed)} failed, "
        f"{len(report.hard_failed)} hard-failed"
    )
    return report

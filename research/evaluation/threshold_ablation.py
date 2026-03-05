"""
Cultural flag threshold ablation: test 40%, 50%, 60%, 70%, 80% thresholds
against the golden set to find the empirically optimal value.

Also audits Bloom verb lists against Anderson & Krathwohl (2001) revised
taxonomy and reports any additions or gaps.

Run:
    python research/evaluation/threshold_ablation.py
    python research/evaluation/threshold_ablation.py --golden-dir research/evaluation/golden_set_v2
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Anderson & Krathwohl (2001) canonical verb lists ──────────────────────────
# Source: Anderson, L.W. & Krathwohl, D.R. (2001). A Taxonomy for Learning,
#         Teaching and Assessing. Longman.

AK_VERBS: dict[str, list[str]] = {
    "REMEMBER": [
        "define", "duplicate", "list", "memorise", "memorize", "recall",
        "repeat", "reproduce", "state", "recognise", "recognize",
        "identify", "name", "label", "match", "select", "locate",
    ],
    "UNDERSTAND": [
        "classify", "describe", "discuss", "explain", "identify",
        "locate", "recognise", "recognize", "report", "select",
        "translate", "paraphrase", "summarise", "summarize",
        "interpret", "exemplify", "instantiate", "infer",
        "compare", "explain", "give examples", "illustrate",
    ],
    "APPLY": [
        "choose", "demonstrate", "dramatise", "dramatize",
        "employ", "illustrate", "interpret", "operate",
        "schedule", "sketch", "solve", "use", "write",
        "carry out", "execute", "implement", "apply",
        "calculate", "complete", "show", "practise", "practice", "perform",
    ],
    "ANALYZE": [
        "appraise", "compare", "contrast", "criticise", "criticize",
        "differentiate", "discriminate", "distinguish", "examine",
        "experiment", "question", "test", "analyse", "analyze",
        "break down", "categorise", "categorize", "separate",
        "order", "attribute", "organise", "organize", "deconstruct",
        "investigate", "relate", "select", "infer",
    ],
    "EVALUATE": [
        "appraise", "argue", "defend", "judge", "select",
        "support", "value", "evaluate", "critique", "assess",
        "justify", "recommend", "rate", "rank", "measure",
        "decide", "review", "weigh", "conclude", "prioritise", "prioritize",
    ],
    "CREATE": [
        "assemble", "construct", "create", "design", "develop",
        "formulate", "write", "plan", "produce", "generate",
        "invent", "make", "build", "compose", "hypothesise",
        "hypothesize", "propose", "combine", "compile", "devise",
    ],
}

# ── Current production verb lists (from rules.py) ────────────────────────────

PRODUCTION_VERBS: dict[str, list[str]] = {
    "REMEMBER":   ["define", "list", "recall", "name", "identify", "state", "memorise", "memorize",
                   "know", "recognise", "recognize", "repeat", "match", "label", "select"],
    "UNDERSTAND": ["explain", "describe", "summarise", "summarize", "paraphrase", "interpret", "classify", "give examples",
                   "understand", "discuss", "express", "locate", "report", "review", "tell"],
    "APPLY":      ["solve", "use", "demonstrate", "calculate", "apply", "carry out", "execute", "implement",
                   "practise", "practice", "show", "complete", "model", "perform", "present"],
    "ANALYZE":    ["analyse", "analyze", "compare", "contrast", "distinguish", "examine", "break down", "differentiate",
                   "investigate", "question", "test", "categorise", "categorize", "separate", "order"],
    "EVALUATE":   ["evaluate", "justify", "assess", "critique", "judge", "argue", "defend",
                   "review", "measure", "recommend", "rank", "select", "decide", "rate"],
    "CREATE":     ["design", "create", "compose", "construct", "develop", "produce", "plan",
                   "write", "make", "build", "generate", "formulate", "invent", "propose"],
}

# ── Name sets (mirror of rules.py) ───────────────────────────────────────────

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
    "efua", "akosua", "adjoa", "esi", "yaw", "kojo", "akua",
    "nana", "afia", "ekua", "mensah", "asante", "owusu", "boateng",
    "okonkwo", "ezinne", "ifeanyi", "oluwaseun", "olumide", "adebayo",
}


def _western_ratio(text: str) -> Optional[float]:
    words = re.findall(r"\b[A-Z][a-z]+\b", text)
    names = [w for w in words if w.lower() in WESTERN_NAMES or w.lower() in LOCAL_NAMES]
    if not names:
        return None
    western = sum(1 for n in names if n.lower() in WESTERN_NAMES)
    return western / len(names)


def _bloom_match(text: str, expected_level: str) -> bool:
    verbs = PRODUCTION_VERBS.get(expected_level.upper(), [])
    lower = text.lower()
    return any(v in lower for v in verbs)


def load_golden(golden_dir: Path) -> list[dict]:
    examples = []
    for f in sorted(golden_dir.glob("*.json")):
        with open(f) as fh:
            ex = json.load(fh)
        # Support both old golden set schema and new v2 schema
        content = (ex.get("generated")
                   or ex.get("content")
                   or ex.get("text", ""))
        bloom   = (ex.get("expected_bloom")
                   or ex.get("expected", {}).get("bloom_level")
                   or ex.get("bloom_level")
                   or ex.get("metadata", {}).get("bloom_level", ""))
        if content and bloom:
            examples.append({"content": content, "bloom_level": bloom.upper(), "file": f.name})
    return examples


# ── Threshold ablation ────────────────────────────────────────────────────────

def run_threshold_ablation(examples: list[dict]) -> dict:
    thresholds = [0.40, 0.50, 0.60, 0.70, 0.80]
    results = {}

    for thresh in thresholds:
        pass_count = 0
        applicable = 0
        for ex in examples:
            ratio = _western_ratio(ex["content"])
            if ratio is None:
                continue  # no names detected, skip
            applicable += 1
            if ratio <= thresh:
                pass_count += 1

        pct = pass_count / applicable * 100 if applicable else 0
        results[thresh] = {
            "threshold":   thresh,
            "pass_count":  pass_count,
            "applicable":  applicable,
            "pass_rate":   round(pct, 1),
        }

    return results


# ── Bloom verb audit ──────────────────────────────────────────────────────────

def run_bloom_audit() -> dict:
    audit = {}
    for level in AK_VERBS:
        ak_set   = set(AK_VERBS[level])
        prod_set = set(PRODUCTION_VERBS.get(level, []))

        in_both      = ak_set & prod_set
        only_in_ak   = ak_set - prod_set       # in A&K but missing from production
        only_in_prod = prod_set - ak_set        # added beyond A&K (Nigerian extensions or custom)

        audit[level] = {
            "ak_count":      len(ak_set),
            "prod_count":    len(prod_set),
            "shared":        sorted(in_both),
            "missing_from_production": sorted(only_in_ak),
            "extensions_beyond_ak":   sorted(only_in_prod),
        }
    return audit


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--golden-dir", default="research/evaluation/golden_set")
    p.add_argument("--output",     default="research/evaluation/results/threshold_ablation.json")
    args = p.parse_args()

    golden_dir = PROJECT_ROOT / args.golden_dir
    examples   = load_golden(golden_dir)
    print(f"Loaded {len(examples)} examples from {golden_dir}\n")

    # --- Cultural threshold ablation ---
    print("=== Cultural Flag Threshold Ablation ===")
    thresh_results = run_threshold_ablation(examples)

    print(f"{'Threshold':>12}  {'Pass Rate':>10}  {'Pass/Applicable':>16}")
    print("-" * 44)
    for t, r in thresh_results.items():
        marker = "  ← current" if t == 0.60 else ""
        print(f"{t:>12.0%}  {r['pass_rate']:>9.1f}%  {r['pass_count']:>6}/{r['applicable']:<6}{marker}")

    # Find optimal threshold (highest pass rate without being too permissive)
    # "optimal" = highest pass rate at threshold <= 0.70
    optimal = max(
        (t for t in thresh_results if t <= 0.70),
        key=lambda t: thresh_results[t]["pass_rate"],
    )
    print(f"\nRecommended threshold: {optimal:.0%} (highest pass rate ≤ 70%)")

    # --- Bloom verb audit ---
    print("\n=== Bloom Verb Audit vs Anderson & Krathwohl (2001) ===")
    bloom_audit = run_bloom_audit()

    for level, info in bloom_audit.items():
        print(f"\n{level}:")
        print(f"  A&K canonical: {info['ak_count']} verbs | Production: {info['prod_count']} verbs | Shared: {len(info['shared'])}")
        if info["missing_from_production"]:
            print(f"  Missing from production (add?): {info['missing_from_production']}")
        if info["extensions_beyond_ak"]:
            print(f"  Production extensions beyond A&K: {info['extensions_beyond_ak']}")

    # Save results
    output = {
        "threshold_ablation": thresh_results,
        "recommended_threshold": optimal,
        "bloom_verb_audit": bloom_audit,
        "citation": "Anderson, L.W. & Krathwohl, D.R. (2001). A Taxonomy for Learning, Teaching and Assessing. Longman.",
    }
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()

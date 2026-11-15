"""
Prompt-based (zero-shot) category classification with an LLM.

The model is shown the list of valid categories with short descriptions and asked to
return JSON: {"category": ..., "explanation": ..., "confidence": "low|medium|high"}.

Robustness rules:
  * The output is parsed defensively (code fences, extra prose, single quotes).
  * The category must be one of the valid labels (exact match after trimming;
    case-insensitive match is accepted because codes are unambiguous).
  * If JSON parsing fails but exactly one valid label occurs in the text, it is
    accepted and marked parse_method='label_search'. Anything else is INVALID and is
    scored as a wrong prediction - never silently replaced by a guess.
  * 'confidence' is the model's own verbal statement. It is not a calibrated
    probability and is reported only as a qualitative label.
"""

from __future__ import annotations

import json
import re

from src.llm_client import LLMClient

CATEGORY_DESCRIPTIONS = {
    "astro-ph.GA": "Astrophysics of Galaxies",
    "astro-ph.SR": "Solar and Stellar Astrophysics",
    "astro-ph.HE": "High Energy Astrophysical Phenomena",
    "astro-ph.CO": "Cosmology and Nongalactic Astrophysics",
    "cond-mat.mes-hall": "Mesoscale and Nanoscale Physics",
    "cond-mat.mtrl-sci": "Materials Science",
    "cond-mat.str-el": "Strongly Correlated Electrons",
    "cs.CL": "Computation and Language (NLP)",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.LG": "Machine Learning",
    "cs.IT": "Information Theory",
    "gr-qc": "General Relativity and Quantum Cosmology",
    "hep-ph": "High Energy Physics - Phenomenology",
    "hep-th": "High Energy Physics - Theory",
    "math.AP": "Analysis of PDEs",
    "math.CO": "Combinatorics",
    "math.OC": "Optimization and Control",
    "math.PR": "Probability",
    "quant-ph": "Quantum Physics",
    "stat.ML": "Statistics - Machine Learning",
    # computer-science categories that dominate the Mendeley dataset
    "cs.AI": "Artificial Intelligence",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.RO": "Robotics",
    "cs.CR": "Cryptography and Security",
    "cs.IR": "Information Retrieval",
    "cs.SE": "Software Engineering",
    "cs.CY": "Computers and Society",
    "cs.HC": "Human-Computer Interaction",
    "cs.SD": "Sound",
    "cs.DC": "Distributed, Parallel, and Cluster Computing",
    "cs.MA": "Multiagent Systems",
    "eess.IV": "Image and Video Processing",
    "eess.AS": "Audio and Speech Processing",
    "eess.SP": "Signal Processing",
    "stat.ME": "Statistics - Methodology",
}
VALID_CONFIDENCE = {"low", "medium", "high"}
MAX_ABSTRACT_WORDS = 350  # keeps prompts short and cost predictable

SYSTEM_PROMPT = ("You are an expert arXiv moderator who assigns each paper to its single "
                 "primary arXiv category. You answer only with valid JSON.")


def build_messages(abstract: str, labels: list[str]) -> list[dict]:
    words = abstract.split()
    abstract = " ".join(words[:MAX_ABSTRACT_WORDS]) + (" ..." if len(words) > MAX_ABSTRACT_WORDS else "")
    options = "\n".join(f"- {c}: {CATEGORY_DESCRIPTIONS.get(c, c)}" for c in labels)
    user = (
        "Classify the abstract into exactly ONE of these arXiv categories:\n"
        f"{options}\n\n"
        f"Abstract:\n\"\"\"{abstract}\"\"\"\n\n"
        "Return JSON with keys:\n"
        '  "category": one category code copied exactly from the list,\n'
        '  "explanation": one sentence citing evidence from the abstract,\n'
        '  "confidence": "low", "medium" or "high".\n'
        "Output the JSON object only."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


LOCAL_MAX_ABSTRACT_WORDS = 250   # local model has a 512-token input limit


def build_messages_local(abstract: str, labels: list[str]) -> list[dict]:
    """
    Prompt for small instruction-tuned seq2seq models (FLAN-T5), which were trained on the
    'OPTIONS:' multiple-choice format and do not produce JSON reliably. Same task and label
    set as `build_messages`; only the output format differs.
    """
    words = abstract.split()
    abstract = " ".join(words[:LOCAL_MAX_ABSTRACT_WORDS])
    options = "\n".join(f"- {c} ({CATEGORY_DESCRIPTIONS.get(c, c)})" for c in labels)
    return [{"role": "user", "content":
             f"Abstract: {abstract}\n\nWhich arXiv category is the primary subject of this paper?\n"
             f"OPTIONS:\n{options}\n\nAnswer with the category code only."}]


def _match_label(value: str, labels: list[str]) -> str | None:
    value = (value or "").strip().strip(".\"'` ")
    if value in labels:
        return value
    lower = {l.lower(): l for l in labels}
    return lower.get(value.lower())


def _extract_json_object(raw: str) -> dict | None:
    text = re.sub(r"```(?:json)?", "", raw or "", flags=re.IGNORECASE)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    blob = match.group(0)
    for candidate in (blob, blob.replace("'", '"')):
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def parse_llm_json(raw: str | None, labels: list[str]) -> dict:
    """
    Parse an LLM reply into {category, explanation, confidence, valid, parse_method, raw}.
    `valid` is False when no unambiguous valid category could be read.
    """
    result = {"category": None, "explanation": "", "confidence": None, "valid": False,
              "parse_method": "failed", "raw": raw}
    if not raw or not raw.strip():
        result["parse_method"] = "empty"
        return result
    obj = _extract_json_object(raw)
    if obj is not None:
        cat = _match_label(str(obj.get("category", "")), labels)
        conf = str(obj.get("confidence", "")).strip().lower()
        result.update(explanation=str(obj.get("explanation", "")).strip()[:500],
                      confidence=conf if conf in VALID_CONFIDENCE else None)
        if cat:
            result.update(category=cat, valid=True, parse_method="json")
            return result
        result["parse_method"] = "json_invalid_label"
        return result
    # Fallback: accept only if exactly one valid label is mentioned.
    found = {l for l in labels if re.search(rf"(?<![\w.-]){re.escape(l)}(?![\w-])", raw)}
    if len(found) == 1:
        result.update(category=found.pop(), valid=True, parse_method="label_search")
        return result
    if not found:
        # Small models sometimes answer with the category NAME instead of the code.
        low = raw.lower()
        by_name = {l for l in labels if l in CATEGORY_DESCRIPTIONS
                   and CATEGORY_DESCRIPTIONS[l].lower().split(" (")[0] in low}
        name = lambda l: CATEGORY_DESCRIPTIONS[l].lower().split(" (")[0]  # noqa: E731
        # Drop a name that only matched as part of a longer matched name.
        by_name = {l for l in by_name if not any(o != l and name(l) in name(o) for o in by_name)}
        if len(by_name) == 1:
            result.update(category=by_name.pop(), valid=True, parse_method="description_search")
    return result


def classify_with_llm(abstract: str, client: LLMClient, labels: list[str]) -> dict:
    """Classify one abstract. Returns the parsed dict plus 'error' and 'latency_s'."""
    messages = (build_messages_local(abstract, labels) if client.backend == "local"
                else build_messages(abstract, labels))
    raw = client.chat(messages, max_tokens=200)
    out = parse_llm_json(raw, labels)
    out["error"] = client.last_error
    out["latency_s"] = client.cached_latency(messages, max_tokens=200)
    return out

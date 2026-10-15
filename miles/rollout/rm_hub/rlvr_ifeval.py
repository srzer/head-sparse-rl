from __future__ import annotations

import json
import logging
import re
import string
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


JsonDict = dict[str, Any]


def _as_text(response: str | None) -> str:
    return str(response or "").strip()


def _target_int(gt: JsonDict, key: str = "N") -> int | None:
    value = gt.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compare_count(count: int, target: int | None, quantifier: str | None = None) -> bool:
    if target is None:
        return False
    quantifier = (quantifier or "").strip().lower()
    if quantifier == "at least":
        return count >= target
    if quantifier == "at most":
        return count <= target
    if quantifier == "around":
        return abs(count - target) <= max(1, round(target * 0.1))
    return count == target


def _words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)


def validate_lowercase(response: str, gt: JsonDict) -> bool:
    del gt
    return not any("A" <= char <= "Z" for char in _as_text(response))


def validate_uppercase(response: str, gt: JsonDict) -> bool:
    del gt
    return not any("a" <= char <= "z" for char in _as_text(response))


def validate_no_commas(response: str, gt: JsonDict) -> bool:
    del gt
    return "," not in _as_text(response)


def validate_quotation(response: str, gt: JsonDict) -> bool:
    del gt
    text = _as_text(response)
    return len(text) >= 2 and text.startswith('"') and text.endswith('"')


def validate_json_format(response: str, gt: JsonDict) -> bool:
    del gt
    try:
        json.loads(_as_text(response))
    except Exception:
        return False
    return True


def validate_end(response: str, gt: JsonDict) -> bool:
    phrase = gt.get("end_phrase")
    return bool(phrase) and _as_text(response).endswith(str(phrase))


def verify_keyword_frequency(response: str, gt: JsonDict) -> bool:
    word = gt.get("word")
    if not word:
        return False
    pattern = rf"\b{re.escape(str(word))}\b"
    count = len(re.findall(pattern, _as_text(response), flags=re.IGNORECASE))
    return _compare_count(count, _target_int(gt), gt.get("quantifier"))


def verify_keywords(response: str, gt: JsonDict) -> bool:
    text = _as_text(response).lower()
    keywords = gt.get("keyword_list") or []
    return all(str(keyword).lower() in text for keyword in keywords)


def validate_forbidden_words(response: str, gt: JsonDict) -> bool:
    text = _as_text(response).lower()
    forbidden_words = gt.get("forbidden_words") or []
    for word in forbidden_words:
        pattern = rf"\b{re.escape(str(word).lower())}\b"
        if re.search(pattern, text):
            return False
    return True


def verify_letter_frequency(response: str, gt: JsonDict) -> bool:
    letter = gt.get("letter")
    if not letter:
        return False
    count = _as_text(response).lower().count(str(letter).lower())
    return _compare_count(count, _target_int(gt), gt.get("quantifier"))


def validate_frequency_capital_words(response: str, gt: JsonDict) -> bool:
    count = sum(1 for word in re.findall(r"\b[A-Z]{2,}\b", _as_text(response)))
    return _compare_count(count, _target_int(gt), gt.get("quantifier"))


def validate_word_constraint(response: str, gt: JsonDict) -> bool:
    return _compare_count(len(_words(_as_text(response))), _target_int(gt), gt.get("quantifier"))


def verify_sentence_constraint(response: str, gt: JsonDict) -> bool:
    text = _as_text(response)
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if not sentences and text:
        sentences = [text]
    return _compare_count(len(sentences), _target_int(gt), gt.get("quantifier"))


def _paragraphs_by_blank_lines(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]


def _paragraphs_by_markdown_divider(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n?\s*\*\s*\*\s*\*\s*\n?", text.strip()) if part.strip()]


def verify_paragraph_count(response: str, gt: JsonDict) -> bool:
    text = _as_text(response)
    target = _target_int(gt)
    if target is None:
        return False
    counts = {
        len(_paragraphs_by_blank_lines(text)),
        len(_paragraphs_by_markdown_divider(text)),
    }
    return target in counts


def validate_paragraphs(response: str, gt: JsonDict) -> bool:
    paragraphs = _paragraphs_by_blank_lines(_as_text(response))
    target = _target_int(gt)
    if target is not None and len(paragraphs) != target:
        return False
    index = _target_int(gt, "i")
    first_word = gt.get("first_word")
    if index is not None and first_word:
        if index < 1 or index > len(paragraphs):
            return False
        return paragraphs[index - 1].lstrip().lower().startswith(str(first_word).lower())
    return True


def validate_sections(response: str, gt: JsonDict) -> bool:
    splitter = str(gt.get("section_splitter") or "Section:")
    count = len(re.findall(rf"(^|\n)\s*{re.escape(splitter)}\s*\S+", _as_text(response)))
    return _compare_count(count, _target_int(gt), gt.get("quantifier"))


def validate_highlighted_sections(response: str, gt: JsonDict) -> bool:
    count = len(re.findall(r"(?<!\*)\*[^*\n][^*\n]*\*(?!\*)", _as_text(response)))
    target = _target_int(gt)
    return target is not None and count >= target


def verify_bullet_points(response: str, gt: JsonDict) -> bool:
    count = len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+\S+", _as_text(response)))
    return _compare_count(count, _target_int(gt), gt.get("quantifier"))


def validate_placeholders(response: str, gt: JsonDict) -> bool:
    count = len(re.findall(r"\[[^\[\]\n]+\]", _as_text(response)))
    target = _target_int(gt)
    return target is not None and count >= target


def validate_title(response: str, gt: JsonDict) -> bool:
    del gt
    return re.search(r"<<[^<>\n]+>>", _as_text(response)) is not None


def verify_postscript(response: str, gt: JsonDict) -> bool:
    marker = gt.get("postscript_marker")
    if not marker:
        return False
    text = _as_text(response)
    last_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(last_lines) and last_lines[-1].startswith(str(marker))


def validate_choice(response: str, gt: JsonDict) -> bool:
    options = gt.get("options")
    if not options:
        return False
    if isinstance(options, str):
        raw_options = [part.strip() for part in re.split(r",|\bor\b", options) if part.strip()]
    else:
        raw_options = [str(option).strip() for option in options]
    text = _as_text(response).strip().lower().strip(string.punctuation + " ")
    normalized_options = [option.lower().strip(string.punctuation + " ") for option in raw_options]
    return any(text == option or text.startswith(option + " ") for option in normalized_options)


def validate_repeat_prompt(response: str, gt: JsonDict) -> bool:
    original_prompt = gt.get("original_prompt")
    if not original_prompt:
        return False
    return _as_text(response).startswith(str(original_prompt).strip())


def validate_two_responses(response: str, gt: JsonDict) -> bool:
    del gt
    parts = [part.strip() for part in _as_text(response).split("******")]
    return len(parts) == 2 and all(parts) and parts[0] != parts[1]


REGISTRY: dict[str, Callable[[str, JsonDict], bool]] = {
    "validate_lowercase": validate_lowercase,
    "validate_uppercase": validate_uppercase,
    "validate_no_commas": validate_no_commas,
    "validate_quotation": validate_quotation,
    "validate_json_format": validate_json_format,
    "validate_end": validate_end,
    "verify_keyword_frequency": verify_keyword_frequency,
    "verify_keywords": verify_keywords,
    "validate_forbidden_words": validate_forbidden_words,
    "verify_letter_frequency": verify_letter_frequency,
    "validate_frequency_capital_words": validate_frequency_capital_words,
    "validate_word_constraint": validate_word_constraint,
    "verify_sentence_constraint": verify_sentence_constraint,
    "verify_paragraph_count": verify_paragraph_count,
    "validate_paragraphs": validate_paragraphs,
    "validate_sections": validate_sections,
    "validate_highlighted_sections": validate_highlighted_sections,
    "verify_bullet_points": verify_bullet_points,
    "validate_placeholders": validate_placeholders,
    "validate_title": validate_title,
    "verify_postscript": verify_postscript,
    "validate_choice": validate_choice,
    "validate_repeat_prompt": validate_repeat_prompt,
    "validate_two_responses": validate_two_responses,
}


def compute_rlvr_ifeval_reward(response: str, label: Any, metadata: JsonDict | None = None) -> float:
    metadata = metadata if isinstance(metadata, dict) else {}
    ground_truth = metadata.get("ground_truth") or label
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except Exception:
            logger.warning("Invalid RLVR-IFeval ground_truth JSON: %r", ground_truth)
            return 0.0
    if not isinstance(ground_truth, dict):
        logger.warning("Missing RLVR-IFeval ground_truth: %r", ground_truth)
        return 0.0

    func_name = str(ground_truth.get("func_name") or "")
    verifier = REGISTRY.get(func_name)
    if verifier is None:
        logger.warning("Unsupported RLVR-IFeval verifier: %s", func_name)
        return 0.0
    try:
        return 1.0 if verifier(response, ground_truth) else 0.0
    except Exception as exc:
        logger.warning("RLVR-IFeval verifier %s failed: %s", func_name, exc)
        return 0.0

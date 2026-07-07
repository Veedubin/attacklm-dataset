"""Carlini prefix-completion extraction probe.

Strategy 1 from Carlini et al. (2021): extract memorized training data
by feeding the first N tokens of a training record as a prompt, then
generating K completions with temperature sampling. Score each completion
against the original assistant turn using exact match, LCS, and BLEU-4.

Reference:
    Carlini, N., et al. "Extracting Training Data from Large Language
    Models." USENIX Security Symposium, 2021.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Result of a single prefix-completion probe."""

    record_index: int
    source: str
    license_id: str
    prompt_hash: str
    reconstruction_hash: str
    best_exact_match: bool
    best_lcs_length: int
    best_bleu4: float
    original_length: int
    best_completion_length: int
    num_completions: int


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace."""
    import re

    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_prefix(record: dict, tokenizer, num_tokens: int = 50) -> str:
    """Extract the first num_tokens of the user turn as a prompt prefix.

    Falls back to the first message's content if no 'user' role is found.
    """
    messages = record.get("messages", [])
    user_content = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
            break
    if not user_content and messages:
        user_content = messages[0].get("content", "")

    if tokenizer is not None:
        tokens = tokenizer.encode(user_content, add_special_tokens=False)
        prefix_tokens = tokens[:num_tokens]
        return tokenizer.decode(prefix_tokens, skip_special_tokens=True)
    # Fallback: character-level prefix (approximate)
    return user_content[: num_tokens * 4]


def get_target_text(record: dict) -> str:
    """Extract the expected assistant response from a training record."""
    messages = record.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    # Fallback: last message
    if messages:
        return messages[-1].get("content", "")
    return ""


def generate_completions(
    model,
    tokenizer,
    prefix: str,
    num_completions: int = 20,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.96,
) -> list[str]:
    """Generate K completions from a prefix prompt using the model.

    Args:
        model: HuggingFace model (must support .generate()).
        tokenizer: HuggingFace tokenizer.
        prefix: The prompt prefix text.
        num_completions: Number of completions to generate (K).
        max_new_tokens: Maximum tokens per completion.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.

    Returns:
        List of K generated text strings.
    """
    if tokenizer is None:
        raise ValueError("Tokenizer is required for prefix-completion probes")

    import torch

    inputs = tokenizer(prefix, return_tensors="pt")
    # Move inputs to same device as model
    if hasattr(model, "device") and model.device.type != "cpu":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    completions = []
    for i in range(num_completions):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
            )
        # Decode only the generated part (skip the prefix)
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completions.append(completion)

    return completions


def exact_match(generated: str, reference: str) -> bool:
    """Check if normalized generated text exactly matches reference."""
    return normalize_text(generated) == normalize_text(reference)


def lcs_length(s1: str, s2: str) -> int:
    """Compute the length of the longest common subsequence."""
    m, n = len(s1), len(s2)
    # Use O(min(m,n)) space for efficiency
    if m < n:
        s1, s2 = s2, s1
        m, n = n, m

    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def bleu4_score(hypothesis: str, reference: str) -> float:
    """Compute BLEU-4 score between hypothesis and reference.

    Uses nltk.translate.bleu_score for the computation.
    Falls back to a simple n-gram overlap if nltk is unavailable.
    """
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        from nltk.tokenize import word_tokenize

        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(hypothesis.lower())

        if len(hyp_tokens) == 0:
            return 0.0

        smoothing = SmoothingFunction().method1
        try:
            return sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing,
            )
        except Exception:
            # If BLEU-4 fails (e.g., too short), return simple overlap
            return _simple_overlap(hyp_tokens, ref_tokens)
    except ImportError:
        # nltk not available — fall back to simple n-gram overlap
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        return _simple_overlap(hyp_tokens, ref_tokens)


def _simple_overlap(hyp_tokens: list[str], ref_tokens: list[str]) -> float:
    """Simple unigram overlap as a fallback for BLEU."""
    if not ref_tokens or not hyp_tokens:
        return 0.0
    ref_set = set(ref_tokens)
    hyp_set = set(hyp_tokens)
    overlap = len(ref_set & hyp_set)
    precision = overlap / len(hyp_set) if hyp_set else 0.0
    recall = overlap / len(ref_set) if ref_set else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_completions(
    completions: list[str],
    reference: str,
) -> list[dict]:
    """Score each completion against the reference text.

    Returns a list of dicts with: exact_match, lcs_length, bleu4.
    """
    results = []
    norm_ref = normalize_text(reference)
    for comp in completions:
        norm_comp = normalize_text(comp)
        results.append(
            {
                "exact_match": norm_comp == norm_ref,
                "lcs_length": lcs_length(norm_comp, norm_ref),
                "bleu4": bleu4_score(comp, reference),
            }
        )
    return results


def run_carlini_probe(
    record: dict,
    model,
    tokenizer,
    top_k: int = 20,
    max_new_tokens: int | None = None,
    temperature: float = 1.0,
    prefix_tokens: int = 50,
    record_index: int = 0,
) -> ProbeResult:
    """Run a Carlini prefix-completion extraction probe on one record.

    Steps:
        1. Extract first N tokens of the user message as prefix.
        2. Generate K completions with temperature sampling.
        3. Score each completion vs. the original assistant turn.
        4. Return the best-of-K result.

    Args:
        max_new_tokens: Override for maximum new tokens per completion.
            When None (default), computes an adaptive per-record cap:
            min(256, max(64, 2 * suffix_token_count)). Per Carlini 2021
            and MUSE 2023 default of 256 tokens.
    """
    prefix = extract_prefix(record, tokenizer, num_tokens=prefix_tokens)
    target = get_target_text(record)

    if not target:
        logger.warning("Record %d has no assistant target text", record_index)

    provenance = _get_provenance(record)

    # Per attacklm-dataset/docs/PROBE_TOKEN_BUDGET.md (Carlini 2021, MUSE 2023 default)
    suffix_token_count = len(tokenizer.encode(target, add_special_tokens=False))
    adaptive_max_new = min(256, max(64, 2 * suffix_token_count))
    # Use adaptive cap unless caller explicitly overrides
    effective_max_new = (
        max_new_tokens if max_new_tokens is not None else adaptive_max_new
    )

    completions = generate_completions(
        model,
        tokenizer,
        prefix,
        num_completions=top_k,
        max_new_tokens=effective_max_new,
        temperature=temperature,
    )

    scores = score_completions(completions, target)

    # Best-of-K: pick the completion with the highest BLEU-4
    best_idx = max(range(len(scores)), key=lambda i: scores[i]["bleu4"])
    best_score = scores[best_idx]

    return ProbeResult(
        record_index=record_index,
        source=provenance["source"],
        license_id=provenance["license_id"],
        prompt_hash=hashlib.sha256(prefix.encode()).hexdigest()[:16],
        reconstruction_hash=hashlib.sha256(completions[best_idx].encode()).hexdigest()[
            :16
        ],
        best_exact_match=best_score["exact_match"],
        best_lcs_length=best_score["lcs_length"],
        best_bleu4=best_score["bleu4"],
        original_length=len(target),
        best_completion_length=len(completions[best_idx]),
        num_completions=len(completions),
    )


def _get_provenance(record: dict) -> dict:
    """Extract provenance fields from a record dict."""
    return {
        "source": record.get("source", "unknown"),
        "license_id": record.get("license", ""),
    }

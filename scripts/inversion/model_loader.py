from __future__ import annotations

# PROVENANCE METADATA — scripts/inversion/model_loader.py
# ================================================================================
# Attack class:        N/A (this file is infrastructure; it loads models
#                      for the attack code in probe.py / scoring.py / lira.py)
# Original authors:    Veedubin (in-repo author)
# Paper title:         N/A (internal)
# Year / venue:        2026 / in-repo
# Paper URL:           N/A
# Canonical repo:      N/A
#
# Implementation:
#   Type:              ORIGINAL_WORK
#   Lines of port:     N/A
#   Upstream license:  N/A
#
# Upstream dependencies (not derived from, just used by):
#   - HuggingFace transformers library (https://github.com/huggingface/transformers)
#   - llama-cpp-python (https://github.com/abetlen/llama-cpp-python)
#
# Data sources: N/A
#
# Rights claim contact: veedubin.legal@example.com
# See:                  RIGHTS.md (root), data/LEGAL.md, data/REMOVAL.md
# ================================================================================
"""Model loading for the inversion audit harness.

Supports two model formats:
1. HuggingFace safetensors (via transformers.AutoModelForCausalLM)
2. GGUF (via llama-cpp-python)

The white-box interface is mandatory for MIA scoring because we need
access to the model's loss tensor. LMStudio/Ollama are NOT acceptable
because they don't expose loss.
"""



import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def detect_model_format(model_path: Path) -> str:
    """Auto-detect whether model_path is a HF checkpoint or a GGUF file.

    Returns 'hf' for HuggingFace (directory with config.json or *.safetensors)
    or 'gguf' for a single GGUF file.
    """
    model_path = Path(model_path)
    if model_path.is_file() and model_path.suffix.lower() == ".gguf":
        return "gguf"
    if model_path.is_dir():
        if (model_path / "config.json").exists() or any(
            model_path.glob("*.safetensors")
        ):
            return "hf"
        # Check for GGUF files inside directory
        if any(model_path.glob("*.gguf")):
            return "gguf"
    raise ValueError(
        f"Cannot detect model format at {model_path}. "
        f"Expected a directory with config.json / *.safetensors, "
        f"or a .gguf file. Use --model-format to specify explicitly."
    )


def load_hf_model(model_path: Path, device: str = "auto"):
    """Load a HuggingFace model for white-box probing.

    Returns (model, tokenizer) tuple.
    Must be called via transformers — this is the ONLY acceptable path
    for MIA scoring because we need model.forward() to return loss.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading HF model from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_gguf_model(model_path: Path, n_ctx: int = 2048, n_gpu_layers: int = -1):
    """Load a GGUF model via llama-cpp-python.

    Returns (llama_model, tokenizer_proxy) tuple.
    NOTE: GGUF models do NOT expose loss, so MIA scoring is unavailable.
    Only Carlini prefix-completion probes can be run with GGUF.
    """
    from llama_cpp import Llama

    logger.info("Loading GGUF model from %s", model_path)
    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    return llm, None  # No separate tokenizer for GGUF


def load_model(model_path: str | Path, model_format: Optional[str] = None):
    """Load a model from the given path.

    Args:
        model_path: Path to model directory (HF) or .gguf file.
        model_format: 'hf', 'gguf', or None (auto-detect).

    Returns:
        (model, tokenizer) for HF models.
        (llama_model, None) for GGUF models.

    Raises:
        ValueError: If format can't be detected or model can't be loaded.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    if model_format is None:
        model_format = detect_model_format(model_path)

    if model_format == "hf":
        return load_hf_model(model_path)
    elif model_format == "gguf":
        return load_gguf_model(model_path)
    else:
        raise ValueError(f"Unknown model format: {model_format!r}. Use 'hf' or 'gguf'.")

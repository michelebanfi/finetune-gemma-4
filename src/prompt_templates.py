"""
Task-specific prompt builders for ScholarQABench.
"""
from src.benchmark_config import BenchmarkTask


def build_prompt(item: dict, task: BenchmarkTask) -> str:
    if task.task_type == "claim_verification":
        return _scifact_prompt(item)
    elif task.task_type == "yesno_qa":
        return _pubmedqa_prompt(item)
    elif task.task_type == "longform_qa":
        return _qasa_prompt(item)
    else:
        raise ValueError(f"Unknown task_type: {task.task_type}")


def _extract_ctx(gold_ctx) -> tuple[str, str]:
    """Normalise gold_ctx, which may be a dict or a list of dicts."""
    if isinstance(gold_ctx, list):
        ctx = gold_ctx[0] if gold_ctx else {}
    else:
        ctx = gold_ctx or {}
    return ctx.get("title", ""), ctx.get("text", "")


def _scifact_prompt(item: dict) -> str:
    title, text = _extract_ctx(item["gold_ctx"])
    claim = item["input"]
    return (
        "You are a scientific fact-checker. Given the paper excerpt below, "
        "determine whether the claim is supported (true) or refuted (false) by the paper.\n\n"
        f"Paper [1]: {title}\n{text}\n\n"
        f"Claim: {claim}\n\n"
        "First state 'true' or 'false', then briefly explain citing the paper as [1].\n"
        "Answer:"
    )


def _pubmedqa_prompt(item: dict) -> str:
    title, text = _extract_ctx(item["gold_ctx"])
    question = item["input"]
    return (
        "You are a biomedical research assistant. Based on the abstract below, "
        "answer the question with 'yes', 'no', or 'maybe'.\n\n"
        f"Abstract [1]: {title}\n{text}\n\n"
        f"Question: {question}\n\n"
        "First state 'yes', 'no', or 'maybe', then briefly explain citing the abstract as [1].\n"
        "Answer:"
    )


def _qasa_prompt(item: dict) -> str:
    # QASA: gold_ctxs is a list of indices into ctxs
    ctxs = item.get("ctxs", [])
    gold_indices = item.get("gold_ctxs", [])
    question = item["input"]

    if gold_indices and ctxs:
        context_parts = []
        for ref_num, idx in enumerate(gold_indices, start=1):
            if 0 <= idx < len(ctxs):
                ctx = ctxs[idx]
                title = ctx.get("title", "")
                text = ctx.get("text", "")
                header = f"[{ref_num}] {title}" if title else f"[{ref_num}]"
                context_parts.append(f"{header}\n{text}")
        context_block = "\n\n".join(context_parts)
        ref_list = ", ".join(f"[{i+1}]" for i in range(len(gold_indices)))
        ctx_instruction = (
            f"Based on the following paper excerpt(s), provide a detailed answer "
            f"to the question. Cite passages using {ref_list} as appropriate.\n\n"
            f"{context_block}\n\n"
        )
    else:
        ctx_instruction = (
            "Based on your knowledge of the scientific literature, provide a detailed answer "
            "to the question. Cite relevant sources using bracketed numbers.\n\n"
        )

    return (
        f"{ctx_instruction}"
        f"Question: {question}\n\n"
        "Answer:"
    )

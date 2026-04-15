"""
Task registry for ScholarQABench evaluation.
"""
import os
from dataclasses import dataclass, field

SCHOLARQABENCH_DIR = os.environ.get(
    "SCHOLARQABENCH_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "ScholarQABench"),
)


@dataclass
class BenchmarkTask:
    name: str
    data_path: str            # path relative to SCHOLARQABENCH_DIR (or absolute)
    task_type: str            # "claim_verification" | "yesno_qa" | "longform_qa" | "synthesis"
    has_gold_ctx: bool        # whether gold context is available in the data
    max_new_tokens: int
    eval_flags: list          # flags passed to citation_correctness_eval.py
    tier: int                 # 1 = fair comparison, 2 = informational only
    data_format: str = "jsonl"  # "jsonl" or "json"

    @property
    def abs_data_path(self) -> str:
        if os.path.isabs(self.data_path):
            return self.data_path
        return os.path.normpath(os.path.join(SCHOLARQABENCH_DIR, self.data_path))


BENCHMARK_TASKS: dict[str, BenchmarkTask] = {
    "scifact": BenchmarkTask(
        name="scifact",
        data_path="data/single_paper_tasks/scifact_test.jsonl",
        task_type="claim_verification",
        has_gold_ctx=True,
        max_new_tokens=256,
        eval_flags=["--match", "--citations_short"],
        tier=1,
    ),
    "pubmedqa": BenchmarkTask(
        name="pubmedqa",
        data_path="data/single_paper_tasks/pubmed_test.jsonl",
        task_type="yesno_qa",
        has_gold_ctx=True,
        max_new_tokens=256,
        eval_flags=["--match", "--citations_short"],
        tier=1,
    ),
    "qasa": BenchmarkTask(
        name="qasa",
        data_path="data/single_paper_tasks/qasa_test.jsonl",
        task_type="longform_qa",
        has_gold_ctx=True,
        max_new_tokens=512,
        eval_flags=["--citations"],
        tier=1,
    ),
    "scholarqa_cs": BenchmarkTask(
        name="scholarqa_cs",
        data_path="data/scholarqa_cs/test_configs_snippets.json",
        task_type="synthesis",
        has_gold_ctx=False,
        max_new_tokens=1024,
        eval_flags=["--citations"],
        tier=2,
        data_format="json",
    ),
    "scholarqa_bio": BenchmarkTask(
        name="scholarqa_bio",
        data_path="data/scholarqa_bio/scholarqabench_bio.jsonl",
        task_type="synthesis",
        has_gold_ctx=False,
        max_new_tokens=1024,
        eval_flags=["--citations"],
        tier=2,
    ),
    "scholarqa_neuro": BenchmarkTask(
        name="scholarqa_neuro",
        data_path="data/scholarqa_neuro/scholarqabench_neuro.jsonl",
        task_type="synthesis",
        has_gold_ctx=False,
        max_new_tokens=1024,
        eval_flags=["--citations"],
        tier=2,
    ),
}

# OpenScholar-8B reference scores for comparison table
OPENSCHOLAR_8B_SCORES = {
    "scifact":      {"correctness": 76.4, "citation_f1": 68.9, "metric": "accuracy"},
    "pubmedqa":     {"correctness": 76.0, "citation_f1": 43.6, "metric": "accuracy"},
    "qasa":         {"correctness": 23.0, "citation_f1": 56.3, "metric": "rouge-l"},
    "scholarqa_cs": {"correctness": 51.1, "citation_f1": 47.9, "metric": "rubric"},
    "scholarqa_bio":   {"correctness": None, "citation_f1": 42.8, "metric": "n/a"},
    "scholarqa_neuro": {"correctness": None, "citation_f1": 50.8, "metric": "n/a"},
}

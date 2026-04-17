"""
Post-process model outputs for short-form answer tasks so the ScholarQABench
--match evaluation flag can find the canonical answer token.

The eval script does a substring search (case-insensitive) for the gold answer
inside the model's output. Putting the canonical answer first ensures it is
always found even when the model generates verbose explanations.
"""
import re


def normalize_scifact(output: str) -> str:
    """
    Ensure the output starts with 'true' or 'false'.
    Leaves the rest of the text intact for citation scoring.
    """
    lower = output.lower().strip()
    for word in re.split(r"[\s.,;:!?]", lower):
        if word == "true":
            return "true. " + output.strip()
        if word == "false":
            return "false. " + output.strip()
    # Couldn't find a canonical answer — prepend "unknown" (will count as wrong)
    return "unknown. " + output.strip()


def normalize_pubmedqa(output: str) -> str:
    """
    Ensure the output starts with 'yes', 'no', or 'maybe'.
    """
    lower = output.lower().strip()
    for word in re.split(r"[\s.,;:!?]", lower):
        if word == "yes":
            return "yes. " + output.strip()
        if word == "no":
            return "no. " + output.strip()
        if word == "maybe":
            return "maybe. " + output.strip()
    return "maybe. " + output.strip()

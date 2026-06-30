"""Prompt definitions for controlled text-embedding ablation experiments."""


LA_PROMPT_ABLATIONS = {
    # Original prompt used by the main method.
    "LA": {
        "bcp": "An overlapping image of the left atrium",
        "nobcp": "A single image of the left atrium",
        "rule": "Remove imaging modality; retain condition and anatomy.",
    }
}

PANCREAS_PROMPT_ABLATIONS = {
    "Pancreas": {
        "bcp": "An overlapping image of the pancreas",
        "nobcp": "A single image of the pancreas",
        "rule": "Remove imaging modality; retain condition and anatomy.",
    }
}


KITS19_PROMPT_ABLATIONS = {
    "KiTS19": {
        "bcp": "An overlapping image of a kidney tumor",
        "nobcp": "A single image of a kidney tumor",
        "rule": "Remove imaging modality; retain condition and anatomy.",
    },
}


PROMPT_ABLATIONS = {
    **LA_PROMPT_ABLATIONS,
    **PANCREAS_PROMPT_ABLATIONS,
    **KITS19_PROMPT_ABLATIONS,
}


def get_prompt_pair(variant):
    """Return the BCP/noBCP strings for one controlled prompt variant."""
    try:
        config = PROMPT_ABLATIONS[variant]
    except KeyError as exc:
        supported = ", ".join(PROMPT_ABLATIONS)
        raise ValueError(
            f"Unsupported prompt variant {variant!r}. Supported variants: {supported}."
        ) from exc
    return config["bcp"], config["nobcp"]

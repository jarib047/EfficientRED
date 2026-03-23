import argparse
import os
import pickle
import re
from typing import Any, List, Optional

import pandas as pd

HIDDEN_SIZE = 768
NUM_LAYERS = 12
RED_PARAMS_PER_DIM = 2  # scaling_var + bias_var
FULL_FINETUNE_PARAM_MAP = {
    # Total params when fully fine-tuning a standard sequence-classification model.
    "roberta-base": 124_647_170,
}


def _flatten_numbers(value: Any) -> List[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        out: List[float] = []
        for item in value:
            out.extend(_flatten_numbers(item))
        return out
    return []


def _mask_count_from_vector(scores: List[float], tau: float) -> int:
    if not scores:
        return 0
    series = pd.Series(scores, dtype=float)
    threshold = float(series.quantile(1 - tau))
    return int((series > threshold).sum())


def _format_float_for_path(x: float) -> str:
    # Keep stable folder naming like "0.5" (not "0.500000").
    return str(float(x)).rstrip("0").rstrip(".") if "." in str(float(x)) else str(float(x))


def _results_subdir(
    prune_type: str,
    threshold: float,
    prune_epochs: int,
    layer_threshold: float = None,
    param_threshold: float = None,
) -> str:
    th = _format_float_for_path(threshold)
    if prune_type == "ptau":
        return f"results_prune_{prune_epochs}_ptau_{th}"
    if prune_type == "ltau":
        return f"results_prune_{prune_epochs}_ltau_{th}"
    if prune_type == "prop":
        lt = _format_float_for_path(layer_threshold if layer_threshold is not None else threshold)
        pt = _format_float_for_path(param_threshold if param_threshold is not None else threshold)
        return f"proportional_ltau_{lt}_ptau_{pt}"
    raise ValueError(f"Unsupported prune_type: {prune_type}")


def _compute_params_involved(
    prune_type: str,
    threshold: float,
    on_layers: List[int],
    repr_diff: List[Any],
    layer_threshold: Optional[float] = None,
    param_threshold: Optional[float] = None,
) -> int:
    pruned_layers = [i for i in range(NUM_LAYERS) if i not in on_layers] if on_layers else list(range(NUM_LAYERS))
    if prune_type == "ltau":
        return len(pruned_layers) * HIDDEN_SIZE * RED_PARAMS_PER_DIM

    if prune_type == "ptau":
        # repr_diff shape is usually [hidden_size] at epoch level: [ [768 floats] ].
        vec = repr_diff[0] if repr_diff else []
        selected_dims = _mask_count_from_vector(vec, tau=threshold)
        return NUM_LAYERS * selected_dims * RED_PARAMS_PER_DIM

    if prune_type == "prop":
        # repr_diff shape is usually [ [12 x hidden_size] ].
        tau = param_threshold if param_threshold is not None else threshold
        layer_scores = repr_diff[0] if repr_diff else []
        if not isinstance(layer_scores, list) or not layer_scores:
            return 0
        selected_total = 0
        for layer_idx in pruned_layers:
            if layer_idx >= len(layer_scores):
                continue
            selected_total += _mask_count_from_vector(layer_scores[layer_idx], tau=tau)
        return selected_total * RED_PARAMS_PER_DIM

    return 0


def _collect_rows(
    results_dir: str,
    model_name: str,
    seed: int,
    prune_type: str,
    threshold: float,
    task_filter: str = None,
    layer_threshold: Optional[float] = None,
    param_threshold: Optional[float] = None,
    full_finetune_params: Optional[int] = None,
) -> pd.DataFrame:
    rows = []
    pattern = re.compile(rf"^result_{re.escape(model_name)}_(.+)_seed_{seed}\.pkl$")
    for filename in sorted(os.listdir(results_dir)):
        match = pattern.match(filename)
        if not match:
            continue
        task = match.group(1)
        if task_filter and task != task_filter:
            continue

        with open(os.path.join(results_dir, filename), "rb") as rfile:
            result = pickle.load(rfile)

        val_results = result.get("val_results", [])
        test_results = result.get("test_results", [])
        on_layers = result.get("on_layers", [])
        repr_diff = result.get("repr_diff", [])

        if not val_results or not test_results:
            continue

        best_idx = max(range(len(val_results)), key=lambda i: val_results[i])
        pruned_layers = [i for i in range(12) if i not in on_layers] if on_layers else []
        params_involved = _compute_params_involved(
            prune_type=prune_type,
            threshold=threshold,
            on_layers=on_layers,
            repr_diff=repr_diff,
            layer_threshold=layer_threshold,
            param_threshold=param_threshold,
        )
        total_full = full_finetune_params if full_finetune_params is not None else FULL_FINETUNE_PARAM_MAP.get(model_name)
        pct_of_full = round((params_involved / total_full) * 100, 6) if total_full else "-"
        flat_repr = _flatten_numbers(repr_diff)
        repr_mean = round(sum(flat_repr) / len(flat_repr), 4) if flat_repr else None

        rows.append(
            {
                "task name": task,
                "run epochs": len(val_results),
                "best val": round(float(val_results[best_idx]), 4),
                "best test": round(float(test_results[best_idx]), 4),
                "active params": int(params_involved),
                "param share": pct_of_full,
                "kept layers": on_layers if on_layers else "-",
                "drop layers": pruned_layers if pruned_layers else "-",
                "diff mean": repr_mean if repr_mean is not None else "-",
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by="task name").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Visualize EfficientRED result pickles as a table.")
    parser.add_argument("--prune_type", choices=["ptau", "ltau", "prop"], required=True)
    parser.add_argument("--threshold", type=float, required=True, help="Threshold for selected prune type.")
    parser.add_argument("--prune_epochs", type=int, default=1)
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--task", default="all", help="Task name like cola, sst2, ... or 'all'.")
    parser.add_argument("--layer_threshold", type=float, default=None, help="Only used for prune_type=prop.")
    parser.add_argument("--param_threshold", type=float, default=None, help="Only used for prune_type=prop.")
    parser.add_argument(
        "--full_finetune_params",
        type=int,
        default=None,
        help="Override total parameter count used for percentage calculation.",
    )
    args = parser.parse_args()

    main_dir = os.path.dirname(os.path.abspath(__file__))
    subdir = _results_subdir(
        prune_type=args.prune_type,
        threshold=args.threshold,
        prune_epochs=args.prune_epochs,
        layer_threshold=args.layer_threshold,
        param_threshold=args.param_threshold,
    )
    results_dir = os.path.join(main_dir, "results", subdir)

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(
            f"Results directory not found: {results_dir}\n"
            "Run main.py first or check --prune_type/--threshold values."
        )

    task_filter = None if args.task == "all" else args.task
    table = _collect_rows(
        results_dir=results_dir,
        model_name=args.model_name,
        seed=args.seed,
        prune_type=args.prune_type,
        threshold=args.threshold,
        task_filter=task_filter,
        layer_threshold=args.layer_threshold,
        param_threshold=args.param_threshold,
        full_finetune_params=args.full_finetune_params,
    )
    if table.empty:
        print("No matching result files found.")
        return

    print(f"Results source: {results_dir}")
    try:
        print(table.to_markdown(index=False))
    except Exception:
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()

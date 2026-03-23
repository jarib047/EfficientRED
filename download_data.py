from datasets import load_dataset
import os

task_map={
    "sst2": "glue",
    "mrpc": "glue",
    "qqp":  "glue",
    "qnli": "glue",
    "mnli": "glue",
    "rte":  "glue",
    "stsb": "glue",
    "cola": "glue",
}

# task_map = {"wnli": "glue"}

# GLUE for RoBERTa and T5
for k, v in task_map.items():
    dataset = load_dataset(v, k)
    dataset.save_to_disk(os.path.join("~/EfficientRED/data", k))
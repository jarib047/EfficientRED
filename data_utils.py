from transformers import DataCollatorWithPadding
from datasets import load_from_disk
from torch.utils.data import DataLoader
import numpy as np
import os
import torch
import pandas as pd
from torch.utils.data import Subset
from datasets import disable_caching
disable_caching()


glue_data_keys_map = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

glue_data_num_labels_map = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "stsb": 1,
    "qqp": 2,
    "mnli": 3,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}


def clean_data(data_base_path):
    df = pd.read_csv(data_base_path + "train.tsv", delimiter="\t", quoting=3)
    df["label"] = df["label"].apply(lambda x: 0 if x=="not_entailment" else 1)
    df.to_csv(data_base_path + "clean_train.tsv", sep="\t", index=False)
    del df
    df = pd.read_csv(data_base_path + "dev.tsv", delimiter="\t", quoting=3)
    df["label"] = df["label"].apply(lambda x: 0 if x=="not_entailment" else 1)
    df.to_csv(data_base_path + "clean_dev.tsv", sep="\t", index=False)
    del df
    df = pd.read_csv(data_base_path + "test.tsv", delimiter="\t", quoting=3)
    df.to_csv(data_base_path + "clean_test.tsv", sep="\t", index=False)
    del df


def tokenize_data(examples, tokenizer, max_seq_length, sentence1_key, sentence2_key, task):
    output = tokenizer(text=examples[sentence1_key],
                       text_pair=examples[sentence2_key] if sentence2_key else None,
                       max_length=max_seq_length,
                       truncation=True)
    input_ids = output.input_ids
    attention_mask = output.attention_mask
    labels = examples["label"]
    return {"input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels":labels}


def load_data(tokenizer, seed=101, task="mrpc", batch_size=32, generalization=False):
    batch_size = batch_size
    main_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(main_dir, "data", task)
    data = load_from_disk(file_path)
    sentence1_key, sentence2_key = glue_data_keys_map[task]
    tokenized_data = data.map(lambda examples: tokenize_data(examples=examples,
                                                            tokenizer=tokenizer,
                                                            max_seq_length=256,
                                                            sentence1_key=sentence1_key,
                                                            sentence2_key=sentence2_key,
                                                            task=task), batched=True)
    if sentence2_key:
        tokenized_data = tokenized_data.remove_columns([sentence1_key, sentence2_key, "idx", "label"])
    else:
        tokenized_data = tokenized_data.remove_columns([sentence1_key, "idx", "label"])
    
    if generalization:
        return tokenized_data[:5000]

    from utils import set_seed
    set_seed(seed=seed)
    train_dataset = tokenized_data["train"]
    permuted_indices = np.random.RandomState(seed=seed).permutation(len(train_dataset)).tolist()
    val_dataset = tokenized_data["validation_matched"] if task=="mnli" else tokenized_data["validation"]

    if(task in ["mnli", "qnli", "qqp", "sst2"]):
        num_eval_data = 1000
        test_dataset = val_dataset
        val_dataset = Subset(dataset=train_dataset, indices=permuted_indices[:num_eval_data])
        train_dataset = Subset(dataset=train_dataset, indices=permuted_indices[num_eval_data:])

    elif(task in ["cola", "stsb", "mrpc", "rte"]):
        permuted_indices = np.random.RandomState(seed=seed).permutation(len(val_dataset)).tolist()
        num_eval_data = int(len(val_dataset)/2)    
        test_dataset = Subset(dataset=val_dataset, indices=permuted_indices[num_eval_data:])
        val_dataset = Subset(dataset=val_dataset, indices=permuted_indices[:num_eval_data])

    num_labels = glue_data_num_labels_map[task]
    data_collator = DataCollatorWithPadding(tokenizer)

    g = torch.Generator()
    g.manual_seed(seed)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=data_collator,
                                  generator=g)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                collate_fn=data_collator,)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 collate_fn=data_collator,)
    return train_dataloader, val_dataloader, test_dataloader, num_labels

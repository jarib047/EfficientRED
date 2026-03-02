from transformers import (pipeline, LlamaForCausalLM, BitsAndBytesConfig, LlamaTokenizer,
                          AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration,
                          T5Config, get_scheduler, DataCollatorWithPadding, EvalPrediction)
import torch
import evaluate
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, DatasetDict
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import os
import pandas as pd
import pickle
from models import EfficientModel
import GPUtil

tokenizer = None

metric_map = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mrpc": "accuracy",
    "qnli": "accuracy",
    "qqp": "accuracy",
    "rte": "accuracy",
    "sst2": "accuracy",
    "stsb": "pearson",
    "wnli": "accuracy"
}

#for generalization test
def load_source_target_datasets(source_dataset1, source_dataset2, target_dataset):
    key_list = ["source_dataset1", "source_dataset2", "target_dataset"]
    qnli_key_index = [source_dataset1, source_dataset2, target_dataset].index("qnli")
    qnli_key = key_list[qnli_key_index]
    wnli_key_index = [source_dataset1, source_dataset2, target_dataset].index("wnli")
    wnli_key = key_list[wnli_key_index]
    from data_utils import load_data
    dataset_map = {"source_dataset1": load_data(tokenizer=tokenizer, task=source_dataset1, generalization=True),
                   "source_dataset2": load_data(tokenizer=tokenizer, task=source_dataset2, generalization=True),
                   "target_dataset": load_data(tokenizer=tokenizer, task=target_dataset, generalization=True),}

    shuffled_train1 = dataset_map["source_dataset1"]["train"].shuffle(seed=42)
    shuffled_train2 = dataset_map["source_dataset2"]["train"].shuffle(seed=42)
    shuffled_validation1 = dataset_map["source_dataset1"]["validation"].shuffle(seed=42)
    shuffled_validation2 = dataset_map["source_dataset2"]["validation"].shuffle(seed=42)

    if target_dataset in ["wnli", "rte"]:
        test = concatenate_datasets([dataset_map["target_dataset"]["train"],
                                     dataset_map["target_dataset"]["validation"]])
    else:
        test = dataset_map["target_dataset"]["validation"]
    return DatasetDict({"train": concatenate_datasets([shuffled_train1.select(range(32)),
                                                       shuffled_train2.select(range(32))]),
                        "validation": concatenate_datasets([shuffled_validation1.select(range(32)),
                                                            shuffled_validation2.select(range(32))]),
                        "test": test})


def load_genr_data(batch_size=32, generalize_test=True, dataset_set=[], tokenizer=tokenizer):
    batch_size = batch_size
    if generalize_test:
        source_dataset1= dataset_set[0]
        source_dataset2= dataset_set[1]
        target_dataset= dataset_set[2]
        data = load_source_target_datasets(source_dataset1=source_dataset1,
                                           source_dataset2=source_dataset2,
                                           target_dataset=target_dataset,)
    print(f"Dataset details:\n{data.num_rows}")
    collate_fn = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(data["train"],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn,)
    val_dataloader = DataLoader(data["validation"],
                                batch_size=batch_size, shuffle=True,
                                collate_fn=collate_fn,)
    test_dataloader = DataLoader(data["test"],
                                 batch_size=batch_size, shuffle=True,
                                 collate_fn=collate_fn,)
    return train_dataloader, val_dataloader, test_dataloader


def compute_metrics(eval_pred, task):
    main_dir = os.path.dirname(os.path.abspath(__file__))
    metric_func = evaluate.load(path=os.path.join(main_dir, "evaluate/metrics/glue"), config_name=task)
    result = metric_func.compute(predictions=eval_pred.predictions, references=eval_pred.label_ids)
    return result


def get_results(all_labels, all_predictions, task):
    all_l = []
    all_p = []
    total_correct, total_samples = 0, 0
    for labels in all_labels:
        for label in labels:
            all_l.append(int(label))
    for predicts in all_predictions:
        for predict in predicts:
            all_p.append(predict.item())
    label_ids = np.array(all_l)
    predictions = np.array(all_p)
    eval_prediction = EvalPrediction(predictions=predictions, label_ids=label_ids)
    # metric = evaluate.load("glue", "mrpc")
    # result = metric.compute(predictions=eval_prediction.predictions, references=eval_prediction.label_ids)
    # total_correct = (label_ids == predictions).sum().item()
    # total_samples = len(label_ids)
    # return total_correct/total_samples
    result = compute_metrics(eval_pred=eval_prediction, task=task)
    return result



def test_model(dataloader, model, task, device):
    progress_bar_val = tqdm(enumerate(dataloader), total=len(dataloader))
    all_predictions = []
    all_labels = []
    model.eval()
    for step, batch in progress_bar_val:
        batch = {k:v.to(model.base_model.device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
        if task != "stsb":
            predictions = torch.argmax(outputs["logits"], dim=1)
        else:
            predictions = outputs["logits"]
        all_labels.append(labels)
        all_predictions.append(predictions)
    val_result = get_results(all_labels=all_labels,
                             all_predictions=all_predictions,
                             task=task)
    return val_result


def train_eval_model(model, tokenizer, device, num_epochs=10, dataloaders=[], task="rte"):
    train_dataloader, val_dataloader, test_dataloader = dataloaders[0], dataloaders[1], dataloaders[2]
    num_epochs = num_epochs
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=2e-1, #og= 5e-3
                                # weight_decay = 1e-5)
                                )
    loss_fn = torch.nn.CrossEntropyLoss()
    lr_scheduler = get_scheduler("linear",
                                optimizer=optimizer,
                                num_warmup_steps=500,
                                num_training_steps=len(train_dataloader)*num_epochs)

    max_eval_metric = -1000
    importance = []
    best_model = None
    model.to(device);

    model.train()
    epoch_results = []
    test_results = []
    for epoch in range(num_epochs):
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch+1}:")
        model.train()
        for step, batch in progress_bar:
            batch = {k:v.to(model.base_model.device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            outputs = model(input_ids = input_ids,
                            attention_mask = attention_mask,)
            logits = outputs.logits
            loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            progress_bar.set_postfix(Loss=loss.item())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    
    test_result_dict = test_model(dataloader=test_dataloader,
                                    model=model,
                                    task=task,
                                    device=device)
    test_result_new = test_result_dict[metric_map[task]]
    print(f"Epoch {epoch+1} Test results: {test_result_new}")
    del model, optimizer, lr_scheduler
    return epoch_results, test_results, best_model, importance


def run_generalization_test(model, use_tokenizer, device, batch_size=8, dataset_set=["wnli", "qnli", "rte"]):
    global tokenizer
    tokenizer = use_tokenizer
    train_dataloader, val_dataloader, test_dataloader = load_genr_data(batch_size=batch_size, 
                                                                  tokenizer=tokenizer,
                                                                  dataset_set=dataset_set)
    val_results, test_results, best_model, importance = train_eval_model(model=model,
                                                                         tokenizer=tokenizer,
                                                                         num_epochs=10,
                                                                         dataloaders=[train_dataloader, val_dataloader, test_dataloader],
                                                                         task=dataset_set[-1],
                                                                         device=device)
    # print(f"Max acc: {max(test_results)} at Epoch {test_results.index(max(test_results))}")


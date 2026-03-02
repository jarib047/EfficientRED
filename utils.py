from transformers import (AutoTokenizer, RobertaModel ,get_scheduler, EvalPrediction)
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
from models import EfficientModel
from data_utils import load_data
import os
import evaluate
import random


def set_seed(seed = 101):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(False)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def calculate_loss(logits, labels, mse=False):
    if(mse):
        loss_fct = nn.MSELoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())
    else:
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
    return loss


def calculate_pruning_loss(logits, labels, mse=False, l1_diff=[], lambda_prune=0.01, tau=0.5):
    if(mse):
        loss_fct = nn.MSELoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())
    else:
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
    if isinstance(l1_diff[0], list):
        l1_diff = torch.mean(torch.tensor(l1_diff), dim=0).tolist()    
    l1_diff = torch.tensor(l1_diff)
    # l1_diff_norm = (l1_diff - min(l1_diff)) / (max(l1_diff) - min(l1_diff)) 
    # l1_threshold = [1 if i>tau else 0 for i in l1_diff_norm]
    # return loss + lambda_prune * sum(l1_threshold)
    threshold = np.percentile(l1_diff.cpu().numpy(), 100 - tau*100)
    l1_threshold = (l1_diff > threshold).float()
    return loss + lambda_prune * sum(l1_threshold)


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


def load_model(model_name, num_labels, seed, param_mask=None, on_layers=()):
    pruning_model = RobertaModel.from_pretrained(model_name,output_hidden_states=True, add_pooling_layer=False)
    set_seed(seed=seed)
    model = EfficientModel(base_model=pruning_model,
                           on_layers=on_layers,
                           num_labels=num_labels,
                           param_mask_list=param_mask)
    return model


def model_pruner(model_name, device, seed, task="mrpc", prune_epochs=1, layer_tau=0.5):
    base_model = RobertaModel.from_pretrained(model_name,output_hidden_states=True, add_pooling_layer=False)
    for params in base_model.parameters():
        params.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataloader, val_dataloader, test_dataloader, num_labels = load_data(tokenizer=tokenizer, 
                                                                              task=task,
                                                                              batch_size=32,
                                                                              seed=seed)
    model = load_model(model_name=model_name, 
                       num_labels=num_labels,
                       seed=seed)
    model.to(device)
    base_model.to(device)
    base_model.eval()
    model.train()
    total_train_steps = len(train_dataloader)*prune_epochs
    warmup_rate = 0.06
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=1e-4,
                                weight_decay = 1e-5)
    lr_scheduler = get_scheduler("linear",
                                optimizer=optimizer,
                                num_warmup_steps=total_train_steps*warmup_rate,
                                num_training_steps=total_train_steps)
    total_step = 0
    l1_diff_list = []
    mean_l1_diff_list = []
    for epoch in range(prune_epochs):
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        progress_bar.set_description(f"Pruning Epoch {epoch+1}:")
        for step, batch in progress_bar:
            batch = {k:v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            outputs_red = model(input_ids = input_ids,
                                attention_mask = attention_mask,)
            with torch.no_grad():
                outputs_no_red = base_model(input_ids=input_ids,
                                            attention_mask=attention_mask)
            red_hidden_states = outputs_red.hidden_states[1:]
            no_red_hidden_states = outputs_no_red.hidden_states[1:]
            l1_diff = [torch.norm(red_hidden_states[i] - no_red_hidden_states[i], p=1, dim=-1).mean().item() 
                        for i in range(len(red_hidden_states))]
            l1_diff_list.append(l1_diff)
            logits = outputs_red.logits
            if(task=="stsb"):  
                loss = calculate_pruning_loss(logits, labels, mse=True, l1_diff=l1_diff, tau=layer_tau)
            else:
                loss = calculate_pruning_loss(logits, labels, l1_diff=l1_diff, tau=layer_tau)
            # writer.add_scalar("loss", loss, total_step)
            progress_bar.set_postfix(Loss=loss.item())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_step+=1
        mean_l1_diff_list.append(torch.mean(torch.tensor(l1_diff_list), dim=0))
    no_red_layers, l1_diff_norm = model.apply_pruning(mean_l1_diff_list[-1], tau=layer_tau)
    del model, lr_scheduler, optimizer
    return [i for i in range(base_model.config.num_hidden_layers) if i not in no_red_layers], [i.tolist() for i in mean_l1_diff_list], l1_diff_norm


def parameter_pruner(model_name, device, seed, task="mrpc", prune_epochs=1, param_tau=0.5, limit=0.1, on_layers=()):
    l1_diff_list = []
    base_model = RobertaModel.from_pretrained(model_name, 
                                                output_hidden_states=True, 
                                                add_pooling_layer=False)
    for params in base_model.parameters():
        params.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataloader, val_dataloader, test_dataloader, num_labels = load_data(tokenizer=tokenizer, 
                                                                              task=task,
                                                                              batch_size=32,
                                                                              seed=seed)
    model = load_model(model_name=model_name, 
                        num_labels=num_labels,
                        on_layers=on_layers,
                        seed=seed)
    model.to(device)
    base_model.to(device)
    base_model.eval()
    model.train()
    total_train_steps = len(train_dataloader)*prune_epochs
    warmup_rate = 0.06
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=1e-4,
                                weight_decay = 1e-5)
    lr_scheduler = get_scheduler("linear",
                                optimizer=optimizer,
                                num_warmup_steps=total_train_steps*warmup_rate,
                                num_training_steps=total_train_steps)
    total_step = 0
    param_diff_list = []
    mean_param_diff_list = []
    for epoch in range(prune_epochs):
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        progress_bar.set_description(f"Pruning Epoch {epoch+1}:")
        for step, batch in progress_bar:
            batch = {k:v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            outputs_red = model(input_ids = input_ids,
                                attention_mask = attention_mask,)
            with torch.no_grad():
                outputs_no_red = base_model(input_ids=input_ids,
                                            attention_mask=attention_mask)
            red_hidden_states = outputs_red.hidden_states[1:]
            no_red_hidden_states = outputs_no_red.hidden_states[1:]
            param_diff = [torch.abs(red_hidden_states[i] - no_red_hidden_states[i]).mean(dim=(0,1)).detach().tolist()
                                    for i in range(len(red_hidden_states))]
            batch_param_diff = torch.mean(torch.tensor(param_diff), dim=0).tolist()
            param_diff_list.append(batch_param_diff)
            logits = outputs_red.logits
            if(task=="stsb"):  
                loss = calculate_pruning_loss(logits, labels, mse=True, l1_diff=batch_param_diff, tau=param_tau)
            else:
                loss = calculate_pruning_loss(logits, labels, l1_diff=batch_param_diff, tau=param_tau)
            # writer.add_scalar("loss", loss, total_step)
            progress_bar.set_postfix(Loss=loss.item())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_step+=1
        mean_param_diff_list.append(torch.mean(torch.tensor(param_diff_list), dim=0))
    param_mask = model.apply_parameter_pruning(mean_param_diff_list[-1], tau=param_tau, device=device)
    return param_mask, [i.tolist() for i in mean_param_diff_list]


def proportional_pruner(model_name, device, seed, task="mrpc", prune_epochs=1, param_tau=0.5, 
                            limit=0.1, on_layers=(), exclusions=[]):
    l1_diff_list = []
    base_model = RobertaModel.from_pretrained(model_name, 
                                                output_hidden_states=True, 
                                                add_pooling_layer=False)
    for params in base_model.parameters():
        params.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataloader, val_dataloader, test_dataloader, num_labels = load_data(tokenizer=tokenizer, 
                                                                              task=task,
                                                                              batch_size=32,
                                                                              seed=seed)
    model = load_model(model_name=model_name, 
                        num_labels=num_labels,
                        on_layers=on_layers,
                        seed=seed)
    model.to(device)
    base_model.to(device)
    base_model.eval()
    model.train()
    total_train_steps = len(train_dataloader)*prune_epochs
    warmup_rate = 0.06
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=1e-4,
                                weight_decay = 1e-5)
    lr_scheduler = get_scheduler("linear",
                                optimizer=optimizer,
                                num_warmup_steps=total_train_steps*warmup_rate,
                                num_training_steps=total_train_steps)
    total_step = 0
    param_diff_list = []
    mean_param_diff_list = []
    for epoch in range(prune_epochs):
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        progress_bar.set_description(f"Pruning Epoch {epoch+1}:")
        for step, batch in progress_bar:
            batch = {k:v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            outputs_red = model(input_ids = input_ids,
                                attention_mask = attention_mask,)
            with torch.no_grad():
                outputs_no_red = base_model(input_ids=input_ids,
                                            attention_mask=attention_mask)
            red_hidden_states = outputs_red.hidden_states[1:]
            no_red_hidden_states = outputs_no_red.hidden_states[1:]
            param_diff = [torch.abs(red_hidden_states[i] - no_red_hidden_states[i]).mean(dim=(0,1)).detach().tolist()
                                    for i in range(len(red_hidden_states))]
            # batch_param_diff = torch.mean(torch.tensor(param_diff), dim=0).tolist()
            # param_diff_list.append(batch_param_diff)
            param_diff_list.append(param_diff)
            logits = outputs_red.logits
            if(task=="stsb"):  
                loss = calculate_pruning_loss(logits, labels, mse=True, l1_diff=param_diff, tau=param_tau)
            else:
                loss = calculate_pruning_loss(logits, labels, l1_diff=param_diff, tau=param_tau)
            # writer.add_scalar("loss", loss, total_step)
            progress_bar.set_postfix(Loss=loss.item())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_step+=1
        mean_param_diff_list.append(torch.mean(torch.tensor(param_diff_list), dim=0))
    # param_mask = model.apply_parameter_pruning(, tau=param_tau, device=device)
    param_mask_list = []
    for param_diff in mean_param_diff_list[0]:
        threshold = np.percentile(param_diff.cpu().numpy(), 100 - param_tau*100)
        param_mask = (param_diff > threshold).int().to(device)
        param_mask_list.append(param_mask)
    return param_mask_list, [i.tolist() for i in mean_param_diff_list]
            
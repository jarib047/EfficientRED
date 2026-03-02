from transformers import (AutoTokenizer, RobertaModel ,get_scheduler, EvalPrediction)
import torch
from tqdm import tqdm
import numpy as np
import random
import os
from data_utils import load_data
from utils import calculate_loss, model_pruner, test_model, load_model, parameter_pruner, proportional_pruner
import pickle
import GPUtil
import argparse
from generalization import run_generalization_test

main_dir = os.path.dirname(os.path.abspath(__file__))

gpu_id = GPUtil.getFirstAvailable(order="memory", maxMemory=0.75, maxLoad=2.8)[0]
device = torch.device(f"cuda:{gpu_id}")
print(f"Using GPU: {gpu_id}")

training_epoch_dict = {"cola": 40,
                       "sst2": 40,
                       "mrpc": 40,
                       "stsb": 40,
                       "qqp": 20,
                       "mnli": 20,
                       "qnli": 20,
                       "rte": 40,
                       "wnli":40}

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


def main(model_name="roberta-base", task="cola", prune_layers=True, prune_parameters=False, prune_epochs=1, test_run=False, 
         seed=101, layer_tau=0.5, param_tau=0.5, save_result=True, only_prune=False, generalization=False, dsets=[],
         proportional_prune=False):
    print("\n\n-----------------------------")
    print(f"{model_name}-{task}\nPruning: {prune_layers} for epochs: {prune_epochs}\nTraining for {10 if test_run else training_epoch_dict[task]} epochs\nSave results: {save_result}")
    print("-----------------------------")
    # data_download()

    on_layers, repr_diff, l1_diff_norm = [], [], []
    param_mask = None
    if prune_layers:
        print("Pruning Model Layers....\n")
        on_layers, repr_diff, l1_diff_norm = model_pruner(model_name=model_name, 
                                                          task=task, 
                                                          device=device, 
                                                          prune_epochs=prune_epochs, 
                                                          layer_tau=layer_tau,
                                                          seed=seed)
        if only_prune:
            print(f"{task}: {on_layers}")
            return on_layers
        print("Layer Pruning Complete.\n")

    if proportional_prune:
        print("Pruning proportionally....\n")
        param_mask, repr_diff = proportional_pruner(model_name=model_name, 
                                                  task=task, 
                                                  device=device, 
                                                  prune_epochs=prune_epochs,    
                                                  param_tau=param_tau,
                                                  seed=seed,
                                                  exclusions=on_layers,)
        print("Proportional Pruning Complete.")
    
    if prune_parameters:
        print("Pruning Individual Model Parameters....\n")
        param_mask, repr_diff = parameter_pruner(model_name=model_name, 
                                                  task=task, 
                                                  device=device, 
                                                  prune_epochs=prune_epochs,    
                                                  param_tau=param_tau,
                                                  seed=seed,
                                                  on_layers=on_layers)
        print("Pruning Parameters Complete.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataloader, val_dataloader, test_dataloader, num_labels = load_data(tokenizer=tokenizer, 
                                                                              task=task,
                                                                              batch_size=32,
                                                                              seed=seed)
    model = load_model(model_name=model_name, 
                       on_layers=on_layers, 
                       num_labels=num_labels,
                       seed=seed,
                       param_mask=param_mask)

    if generalization:
        print("\n--------------------------------------")
        print(f"Target = {dsets[-1]}, layers: {prune_layers}, param: {prune_parameters}")
        run_generalization_test(model=model, 
                                use_tokenizer=tokenizer, 
                                device=device, 
                                dataset_set=dsets)
        print("-----------------XXX---------------------\n")
        return ""

    num_epochs = 10 if test_run else training_epoch_dict[task]
    total_train_steps = len(train_dataloader)*num_epochs
    warmup_rate = 0.06
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=3e-3,
                                 weight_decay = 1e-5)
    lr_scheduler = get_scheduler("linear",
                                 optimizer=optimizer,
                                 num_warmup_steps=total_train_steps*warmup_rate,
                                 num_training_steps=total_train_steps)
    # if check: return model, [train_dataloader, val_dataloader, test_dataloader], [optimizer, lr_scheduler]
    select_epoch = 0
    total_step = 0
    max_eval_metric = -100
    best_model = None
    model.to(device)
    model.train()
    epoch_results = []
    test_results = []
    total_loss = 0
    for epoch in range(num_epochs):
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch+1}:")
        for step, batch in progress_bar:
            batch = {k:v.to(torch.device(device)) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            outputs_red = model(input_ids = input_ids,
                                attention_mask = attention_mask,)
            logits = outputs_red.logits
            if(task=="stsb"):  
                loss = calculate_loss(logits, labels, mse=True)
            else:
                loss = calculate_loss(logits, labels)
            # writer.add_scalar("loss", loss, total_step)
            progress_bar.set_postfix(Loss=loss.item())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_step+=1

        val_result_dict = test_model(dataloader=val_dataloader, model=model, task=task, device=device)
        val_result = val_result_dict[metric_map[task]]
        print(f"Epoch {epoch+1} Validation results: {val_result}")
        epoch_results.append(val_result)
        # if val_result >= max_eval_metric:
            # max_eval_metric = val_result
            # best_model = model
            # print(f"Validation test showed improvement, testing at epoch {epoch+1}....")
        print(f"Testing at epoch {epoch+1}....")
        test_result_dict = test_model(dataloader=test_dataloader,
                                        model=model,
                                        task=task,
                                        device=device)
        test_result_new = test_result_dict[metric_map[task]]
        print(f"Epoch {epoch+1} Test results: {test_result_new}")
        test_results.append(test_result_new)
        # else:
        #     test_results.append(0)
    del model, optimizer, lr_scheduler

    if save_result:
        if prune_layers and prune_parameters:
            result_path = f"results_prune_{prune_epochs}_ltau_{layer_tau}_ptau_{param_tau}"
        elif proportional_prune:
            result_path = f"proportional_ltau_{layer_tau}_ptau_{param_tau}"
        elif prune_layers:
            result_path = f"results_prune_{prune_epochs}_ltau_{layer_tau}"
        elif prune_parameters:
            result_path = f"results_prune_{prune_epochs}_ptau_{param_tau}"
        else:
            result_path = "results_red"
        results_dir = os.path.join(main_dir, "results", result_path)
        os.makedirs(results_dir, exist_ok=True)
        result_dict =  {"val_results": epoch_results, 
                        "test_results": test_results, 
                        # "best_model": best_model,
                        "on_layers": on_layers,
                        "repr_diff": repr_diff}
        with open(os.path.join(results_dir, f"result_{model_name}_{task}_seed_{seed}.pkl"), "wb") as wFile:
            pickle.dump(result_dict, wFile)
    print("------------------------------XXX--------------------------\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--task", default="all")
    parser.add_argument("--prune_layers", action=argparse.BooleanOptionalAction)
    parser.add_argument("--prune_parameters", action=argparse.BooleanOptionalAction)
    parser.add_argument("--test_run", action=argparse.BooleanOptionalAction)
    parser.add_argument("--only_prune", action=argparse.BooleanOptionalAction)
    parser.add_argument("--prune_epochs", default=1)
    parser.add_argument("--seed", default=101)
    parser.add_argument("--layer_tau", default=0.5)
    parser.add_argument("--param_tau", default=0.5)
    parser.add_argument("--save_result", action=argparse.BooleanOptionalAction)
    parser.add_argument("--generalization", action=argparse.BooleanOptionalAction)
    parser.add_argument("--proportional_prune", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    model_name = args.model_name
    task = args.task
    prune_layers = args.prune_layers
    prune_parameters = args.prune_parameters
    test_run = args.test_run
    prune_epochs = int(args.prune_epochs)
    seed = int(args.seed)
    layer_tau = float(args.layer_tau)
    param_tau = float(args.param_tau)
    save_result = args.save_result
    only_prune = args.only_prune
    generalization = args.generalization
    proportional_prune = args.proportional_prune

    prune_list = []
    if task=="all":
        for task in training_epoch_dict.keys():
            if only_prune:
                prune_list.append(main(model_name=model_name, 
                task=task, 
                prune_layers=prune_layers,
                prune_parameters=prune_parameters, 
                prune_epochs=prune_epochs,
                test_run=test_run,
                seed=seed,
                layer_tau=layer_tau,
                param_tau=param_tau,
                save_result=save_result,
                only_prune=only_prune,
                generalization=generalization,
                proportional_prune=proportional_prune))
            else:
                main(model_name=model_name, 
                    task=task, 
                    prune_layers=prune_layers,
                    prune_parameters=prune_parameters, 
                    prune_epochs=prune_epochs,
                    test_run=test_run,
                    seed=seed,
                    layer_tau=layer_tau,
                    param_tau=param_tau,
                    save_result=save_result,
                    only_prune=only_prune,
                    generalization=generalization,
                    proportional_prune=proportional_prune)
        print(prune_list)
            
    else:
        main(model_name=model_name, 
             task=task, 
             prune_layers=prune_layers,
             prune_parameters=prune_parameters, 
             prune_epochs=prune_epochs,
             test_run=test_run,
             seed=seed,
             layer_tau=layer_tau,
             param_tau=param_tau,
             save_result=save_result,
             generalization=generalization,
             proportional_prune=proportional_prune)


# if __name__=="__main__":
#     for dsets in [["wnli", "rte", "qnli"], ["qnli", "rte", "wnli"]]:
#         for lay in [0.25, 0.5, 0.75]:
#             # for par in [True, False]:
#             main(generalization=True, 
#                 prune_layers=True, 
#                 prune_parameters=False,
#                 layer_tau=lay,
#                 dsets=dsets,
#                 task=dsets[-1])


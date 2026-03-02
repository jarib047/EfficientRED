from transformers import RobertaModel, AutoTokenizer
import re
from models import EfficientModel
import torch
import os
from utils import load_model
from data_utils import load_data
import pickle
import GPUtil
from utils import model_pruner, set_seed


def final_results(use_dict=["results_red", "results_prune_1", "results_prune_5", "results_prune_10"]):
    for result_path in use_dict:
        seeds = [str(i) for i in range(101, 106)]
        main_dir = os.path.abspath(os.path.join(__file__, ".."))
        red_results_path = os.path.join(main_dir, "results", result_path)
        result_files = [i for i in os.listdir(red_results_path) if i[-3:]=="pkl"]
        for seed in seeds:
            result_files_per_seed = [i for i in  result_files if i[-7:-4]==seed]
            if result_files_per_seed == []:
                print(f"No outputs for given seed {seed}")
                continue
            result_dict = {}
            for result_file in result_files_per_seed:
                task=result_file.split(".")[0].split("_")[-3]
                result = pickle.load(open(os.path.join(red_results_path, result_file), "rb"))
                print(f"Converting {os.path.join(red_results_path, result_file)}")
                result_dict[task] = [round(torch.tensor([float(i) for i in result["val_results"]]).mean().item() * 100, 4), 
                                    round(torch.tensor([float(i) for i in result["val_results"]]).std().item() * 100, 4),
                                    round(max(result["test_results"]) * 100, 4),
                                    result["on_layers"],
                                    result["repr_diff"]]
            with open(os.path.join(red_results_path, f"{result_path.split('_')[-1]}_{seed}_final.pkl"), "wb") as wFile:
                pickle.dump(result_dict, wFile)



def no_best_model(change_dict=["results_red", "results_prune_1", "results_prune_5", "results_prune_10"]):
    for result_path in change_dict:
        main_dir = os.path.abspath(os.path.join(__file__, ".."))
        result_dir = os.path.join(main_dir, "results", result_path)
        result_files = [i for i in os.listdir(result_dir) if i[-3:]=="pkl"]
        for result_file in result_files:
            try:
                result = pickle.load(open(os.path.join(result_dir, result_file), "rb"))
            except:
                print(f"!!!!!!!!!!!!Error opening {os.path.join(result_dir, result_file)}!!!!!!!!!!!!!")
            try:
                result.pop("best_model")
            except:
                print(f"No best model in {os.path.join(result_dir, result_file)}")
                continue
            with open(os.path.join(result_dir, result_file), "wb") as wFile:
                pickle.dump(result, wFile)
            wFile.close()
            print(f"Done for {os.path.join(result_dir, result_file)}")


def get_model(model_name, task, device, prune_epochs, seed):
    set_seed(seed=seed)
    on_layers, repr_diff, l1_diff_norm = model_pruner(model_name=model_name, 
                                                      task=task, 
                                                      device=device, 
                                                      prune_epochs=prune_epochs, 
                                                      tau=0.5)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    train_dataloader, val_dataloader, test_dataloader, num_labels = load_data(tokenizer=tokenizer, 
                                                                              task=task,
                                                                              batch_size=32,
                                                                              seed=seed)
    model = load_model(model_name=model_name, on_layers=on_layers, num_labels=num_labels)
    return model


if __name__ == "__main__":
    final_results(use_dict=["proportional_ltau_0.5_ptau_0.1",
                            "proportional_ltau_0.5_ptau_0.5",
                            "proportional_ltau_0.5_ptau_0.25",
                            "proportional_ltau_0.5_ptau_0.75",
                            "proportional_ltau_0.25_ptau_0.5",
                            "proportional_ltau_0.75_ptau_0.5",])

    # model = RobertaModel.from_pretrained("roberta-base")
    # for key, _ in model.named_modules():
    #     pattern = r'\d+\.output.dense'
    #     match = re.search(pattern, key)
    #     if (match):
    #         print(key.split(".")[2])
            

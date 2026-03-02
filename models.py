import torch
import torch.nn as nn
import re
import numpy as np


class ClassifierHead(nn.Module):
    def __init__(self, input_size, num_labels):
        super().__init__()
        self.dense = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout()
        self.out_proj = nn.Linear(input_size, num_labels)


    def forward(self, features, **kwargs):
        x = features[:, 0, :] #[CLS] pooling
        # x = features.mean(dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RED(nn.Module):
    def __init__(self, replace_layer, hidden_size, param_mask_list, layer_idx):
        super().__init__()
        self.replace_layer = replace_layer
        self.hidden_size = hidden_size
        self.param_mask_list = param_mask_list
        self.layer_idx = layer_idx
        self.weight = torch.bfloat16
        self.red = nn.ParameterDict({"scaling_var": nn.Parameter(torch.ones(1, hidden_size)),
                                     "bias_var": nn.Parameter(torch.zeros(1, hidden_size)),})


    def forward(self, x):
        hidden_states_og = self.replace_layer(x)
        hidden_states = hidden_states_og * self.red["scaling_var"]
        hidden_states = hidden_states + self.red["bias_var"]
        if self.param_mask_list is not None:
            # print(self.layer_idx, self.param_mask_list)
            param_mask = self.param_mask_list[self.layer_idx]
            hidden_states_masked = hidden_states * param_mask
            hidden_states_og_non_essential = hidden_states_og * (1-param_mask)
            return hidden_states_masked + hidden_states_og_non_essential
        return hidden_states


class EfficientModel(nn.Module):
    def __init__(self, base_model, param_mask_list=None, num_labels=2, on_layers=(), model_type="roberta"):
        super().__init__()
        self.base_model = base_model
        self.model_type=model_type
        self.param_mask_list = param_mask_list
        # self.og_layers = []
        # self.pruned_layers = []
        if self.model_type == "T5":
            self.embedding_size = self.base_model.config.d_model
        elif self.model_type == "roberta":
            self.embedding_size = self.base_model.config.hidden_size
        self.freeze_model()

        if model_type == "T5":
            for key, _ in self.base_model.named_modules():
                if "wo" in key and "encoder" in key: #todo setup for other models
                    if all([f"block.{i}" not in key for i in on_layers]):
                        self.replace_layer(key, param_mask=self.param_mask_list)
        elif model_type == "roberta":
            for key, _ in self.base_model.named_modules():
                pattern = r'\d+\.output.dense'
                match = re.search(pattern, key)
                exclusions = [str(i)+".output.dense" for i in on_layers]
                if (match) and (match.group() not in exclusions):
                    self.replace_layer(key, param_mask=self.param_mask_list)

        self.classifier = ClassifierHead(self.base_model.config.hidden_size, num_labels)
        # self.print_trainable_parameters(self.param_mask)


    def freeze_model(self):
        for i in self.base_model.parameters():
            i.requires_grad=False


    def replace_layer(self, key, param_mask):
        current_layer = self.base_model.get_submodule(key)
        current_key = key.split(".")[-1]
        parent_name = ".".join(key.split(".")[:-1])
        parent_layer = self.base_model.get_submodule(parent_name)
        updated_layer = RED(replace_layer=current_layer,
                            hidden_size=self.base_model.config.hidden_size,
                            param_mask_list=param_mask,
                            layer_idx=int(key.split(".")[2]))
        setattr(parent_layer, current_key, updated_layer)
        return [parent_layer, current_key, current_layer]


    def print_trainable_parameters(self, param_mask):
        total_parameters = 0
        trainable_parameters = 0
        for i in self.base_model.parameters():
            total_parameters += i.numel()
            if i.requires_grad:
                trainable_parameters += i.numel()
        if param_mask is not None:
            trainable_parameters *= int(param_mask.sum().item()) / self.base_model.config.hidden_size
        print(f"Total Parameters: {total_parameters}\nTotal Trainable Parameters: {int(trainable_parameters)}\nTrainable Parameters %: {round(trainable_parameters * 100 / total_parameters, 5)}")


    def generate(self, **kwargs):
        return self.base_model.generate(**kwargs)


    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if self.model_type == "roberta":
            outputs =  self.base_model.forward(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            **kwargs)
            logits = self.classifier(outputs.last_hidden_state)
            outputs["logits"] = logits
            return outputs
        


    def get_results(self, input_ids=None, attention_mask=None):
        #for text-to-text models
        last_hidden_state = self.base_model.encoder(input_ids=input_ids,
                                                    attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(last_hidden_state)
        predictions = torch.argmax(logits, dim=1)
        return predictions


    def get_save_dict(self):
        state_dict = self.base_model.state_dict()
        save_dict = {k: state_dict[k] for k in state_dict if "activation_" in k}
        return save_dict


    def save_model(self, save_path):
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_path)


    def load_model(self, save_path):
        save_dict = torch.load(save_path)
        save_key_list = save_dict.keys()
        key_list = [key for key, _ in self.base_model.state_dict().items()]
        for key in key_list:
            if key in save_key_list:
                new_module = save_dict[key]
                new_module.requires_grad = True
                parent_key = ".".join(key.split(".")[:-1])
                replaced_name_last = key.split(".")[-1]
                self.base_model.get_submodule(parent_key)[replaced_name_last] = new_module


    def revert_layer(self, key):
        current_layer = self.base_model.get_submodule(key)
        current_key = key.split(".")[-1]
        parent_name = ".".join(key.split(".")[:-1])
        parent_layer = self.base_model.get_submodule(parent_name)
        updated_layer = RED(replace_layer=current_layer,
                            hidden_size=self.base_model.config.hidden_size)
        setattr(parent_layer, current_key, updated_layer)

    
    def apply_pruning(self, l1_diff, tau):
        l1_diff_norm = (l1_diff - min(l1_diff)) / (max(l1_diff) - min(l1_diff)) 
        l1_diff = l1_diff_norm.tolist()
        l1_diff_dict = {i:j for i, j in enumerate(l1_diff)}
        to_prune = [i for i in l1_diff_dict if l1_diff_dict[i]<tau]
        print(f"Pruning layers {to_prune}")
        return to_prune, l1_diff_norm

    
    def apply_parameter_pruning(self, param_diff, tau, device):
        threshold = np.percentile(param_diff.cpu().numpy(), 100 - tau*100)
        mask = (param_diff > threshold).int().to(device)
        return mask

    


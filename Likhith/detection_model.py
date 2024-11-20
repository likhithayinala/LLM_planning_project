import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config, BertModel, BertConfig, AutoModelForCausalLM

def select_det_model(model_name,config):
    match model_name:
        case 'MLP':
            return MLP(config)
        case 'DistilGPTClassifier':
            return DistilGPTClassifier(config)
        case 'TinyBERTClassifier':
            return TinyBERTClassifier(config)
        case 'Qwen2':
            return Qwen2(config)
        case _:
            raise ValueError(f"Model {model_name} not supported")

class MLP(nn.Module):
    def __init__(self,config):
        super(MLP,self).__init__()
        # Make the model size configurable where config['model_dim'] is the list of dimensions for each layer and its length is the number of layers
        # Modify the classifier_dim such that the first size is the hidden_dim
        self.config = config
        self.config['classifier_dim'] = [config['hidden_dim']] + config['classifier_dim']
        for i in range(len(config['classifier_dim'])-1):
            setattr(self,'fc'+str(i+1),nn.Linear(config['classifier_dim'][i],config['classifier_dim'][i+1]))
        

    def forward(self,x):
        for i in range(len(self.config['classifier_dim'])-1):
            x = getattr(self,'fc'+str(i+1))(x)
            x = F.relu(x)
        return x

class DistilGPTClassifier(nn.Module):
    def __init__(self, config):
        super(DistilGPTClassifier, self).__init__()
        self.config = config
        self.model_config = GPT2Config.from_pretrained("distilgpt2")
        self.model = GPT2Model.from_pretrained("distilgpt2", config=self.model_config)
        self.hdim_fc = nn.Linear(self.config['hidden_dim'], self.model_config.n_embd)
        self.config['classifier_dim'] = [self.model_config.n_embd] + config['classifier_dim']
        for i in range(len(config['classifier_dim'])-1):
            setattr(self,'fc'+str(i+1),nn.Linear(config['classifier_dim'][i],config['classifier_dim'][i+1]))
        
    
    def forward(self, x):
        x = self.hdim_fc(x)
        for block in self.model.h:
            x,_ = block(x)
        for i in range(len(self.config['classifier_dim'])-1):
            x = getattr(self,'fc'+str(i+1))(x)
            x = F.relu(x)
        return x

class TinyBERTClassifier(nn.Module):
    def __init__(self, config):
        super(TinyBERTClassifier, self).__init__()
        self.config = config
        self.model_config = BertConfig.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
        self.model = BertModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", config=self.model_config)
        self.hdim_fc = nn.Linear(self.config['hidden_dim'], self.model_config.hidden_size)
        self.config['classifier_dim'] = [self.model_config.hidden_size] + config['classifier_dim']
        for i in range(len(config['classifier_dim'])-1):
            setattr(self,'fc'+str(i+1),nn.Linear(config['classifier_dim'][i],config['classifier_dim'][i+1]))
    
    def forward(self,x):
        x = self.hdim_fc(x)
        for block in self.model.encoder.layer:
            x = block(x)[0]
        for i in range(len(self.config['classifier_dim'])-1):
            x = getattr(self,'fc'+str(i+1))(x)
            x = F.relu(x)
        return x
        
class Qwen2(nn.Module):
    def __init__(self, config):
        super(Qwen2, self).__init__()
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B",output_hidden_states=True,return_dict_in_generate=True)
        self.hdim_fc = nn.Linear(self.config['hidden_dim'], 896)
        self.config['classifier_dim'] = [896] + config['classifier_dim']
        for i in range(len(config['classifier_dim'])-1):
            setattr(self,'fc'+str(i+1),nn.Linear(config['classifier_dim'][i],config['classifier_dim'][i+1]))
    
    def forward(self,x):
        x = self.hdim_fc(x)
        outputs = self.model(inputs_embeds=x)
        x = outputs.hidden_states[-1]
        for i in range(len(self.config['classifier_dim'])-1):
            x = getattr(self,'fc'+str(i+1))(x)
            x = F.relu(x)
        return x


# Write a function to test all the models, pass random data through them and check if the output shape is as expected
if __name__ == '__main__':
    config = {
        'hidden_dim': 768,
        'classifier_dim': [512, 256, 128, 64, 2]
    }
    data = torch.randn(1,10,768)
    for model_name in ['MLP', 'DistilGPTClassifier', 'TinyBERTClassifier', 'Qwen2']:
        model = select_det_model(model_name, config)
        output = model(data)
        expected_shape = (1,10,2)
        assert output.shape == expected_shape, f"Model {model_name} failed. Expected shape: {expected_shape}, got: {output.shape}"
    print("All models passed!")
import torch 

def select_main_model(config):
    return main_model(config)

class main_model:
    def __init__(self,config):
        self.config = config
        
    def forward(self,x):
        x = torch.rand(1,10,768)
        return x
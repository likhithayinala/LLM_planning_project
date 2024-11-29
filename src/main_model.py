import torch 

def select_main_model(model_name,config):
    return main_model(config)

class main_model(torch.nn.Module):
    def __init__(self, config):
        super(main_model, self).__init__()
        self.config = config
        
    def forward(self, x):
        x = torch.rand(32,1,768)
        return x
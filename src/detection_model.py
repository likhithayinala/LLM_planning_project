import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType
from transformers import GPT2Model, GPT2Config, BertModel, BertConfig, AutoModelForCausalLM

def select_det_model(model_name, config):
    """
    Select and initialize a detection model based on the model name.

    Args:
        model_name (str): Name of the model to initialize.
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        nn.Module: An instance of the selected model.

    Raises:
        ValueError: If the model name is not supported.
    """
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
    def __init__(self, config):
        """
        Multi-Layer Perceptron (MLP) model.

        Args:
            config (dict): Configuration dictionary containing:
                - 'hidden_dim': Input feature dimension.
                - 'classifier_dim': List of output dimensions for each layer.
        """
        super(MLP, self).__init__()
        self.config = config.copy()
        # Build full architecture by prepending 'hidden_dim' to 'classifier_dim'
        classifier_dims = [config['hidden_dim']] + config['classifier_dim']
        self.classifier_dims = classifier_dims
        # Create linear and batch normalization layers
        for i in range(len(classifier_dims) - 1):
            setattr(self, f'fc{i+1}', nn.Linear(classifier_dims[i], classifier_dims[i+1]))
            if i < len(classifier_dims) - 2:
                setattr(self, f'bn{i+1}', nn.BatchNorm1d(classifier_dims[i+1]))
                
    def forward(self, x):
        """
        Forward pass of the MLP model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, hidden_dim).

        Returns:
            Tensor: Output tensor.
        """
        for i in range(len(self.classifier_dims) - 1):
            x = getattr(self, f'fc{i+1}')(x)
            if i < len(self.classifier_dims) - 2:
                x = getattr(self, f'bn{i+1}')(x)
                x = F.relu(x)
        return x

class DistilGPTClassifier(nn.Module):
    def __init__(self, config):
        """
        Classifier based on DistilGPT2 with LoRA fine-tuning.

        Args:
            config (dict): Configuration dictionary containing:
                - 'hidden_dim': Input feature dimension.
                - 'classifier_dim': List of output dimensions for each layer.
        """
        super(DistilGPTClassifier, self).__init__()
        self.config = config.copy()
        # Load DistilGPT2 model and apply LoRA configuration
        self.model_config = GPT2Config.from_pretrained("distilgpt2")
        self.model = GPT2Model.from_pretrained("distilgpt2", config=self.model_config)
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="all"
        )
        self.model = get_peft_model(self.model, peft_config)
        # Input projection layer
        self.hdim_fc = nn.Linear(config['hidden_dim'], self.model_config.n_embd)
        # Build classifier layers
        classifier_dims = [self.model_config.n_embd] + config['classifier_dim']
        self.classifier_dims = classifier_dims
        for i in range(len(classifier_dims) - 1):
            setattr(self, f'fc{i+1}', nn.Linear(classifier_dims[i], classifier_dims[i+1]))
        
    def forward(self, x):
        """
        Forward pass of the classifier.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, hidden_dim).

        Returns:
            Tensor: Output tensor.
        """
        x = self.hdim_fc(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        for block in self.model.h:
            x, _ = block(x)
        for i in range(len(self.classifier_dims) - 1):
            x = getattr(self, f'fc{i+1}')(x)
            x = F.relu(x)
        return x.squeeze(1)  # Remove sequence dimension

class TinyBERTClassifier(nn.Module):
    def __init__(self, config):
        """
        Classifier based on TinyBERT with LoRA fine-tuning.

        Args:
            config (dict): Configuration dictionary containing:
                - 'hidden_dim': Input feature dimension.
                - 'classifier_dim': List of output dimensions for each layer.
        """
        super(TinyBERTClassifier, self).__init__()
        self.config = config.copy()
        # Load TinyBERT model and apply LoRA configuration
        self.model_config = BertConfig.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
        self.model = BertModel.from_pretrained(
            "huawei-noah/TinyBERT_General_4L_312D", config=self.model_config
        )
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="all"
        )
        self.model = get_peft_model(self.model, peft_config)
        # Input projection layer
        self.hdim_fc = nn.Linear(config['hidden_dim'], self.model_config.hidden_size)
        # Build classifier layers
        classifier_dims = [self.model_config.hidden_size] + config['classifier_dim']
        self.classifier_dims = classifier_dims
        for i in range(len(classifier_dims) - 1):
            setattr(self, f'fc{i+1}', nn.Linear(classifier_dims[i], classifier_dims[i+1]))
        
    def forward(self, x):
        """
        Forward pass of the classifier.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, hidden_dim).

        Returns:
            Tensor: Output tensor.
        """
        x = self.hdim_fc(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        for block in self.model.encoder.layer:
            x = block(x)[0]
        for i in range(len(self.classifier_dims) - 1):
            x = getattr(self, f'fc{i+1}')(x)
            x = F.relu(x)
        return x.squeeze(1)  # Remove sequence dimension

class Qwen2(nn.Module):
    def __init__(self, config):
        """
        Classifier based on Qwen model with LoRA fine-tuning.

        Args:
            config (dict): Configuration dictionary containing:
                - 'hidden_dim': Input feature dimension.
                - 'classifier_dim': List of output dimensions for each layer.
        """
        super(Qwen2, self).__init__()
        self.config = config.copy()
        # Load Qwen model and apply LoRA configuration
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="all"
        )
        self.model = get_peft_model(self.model, peft_config)
        # Input projection layer
        self.hdim_fc = nn.Linear(config['hidden_dim'], 896)
        # Build classifier layers
        classifier_dims = [896] + config['classifier_dim']
        self.classifier_dims = classifier_dims
        for i in range(len(classifier_dims) - 1):
            setattr(self, f'fc{i+1}', nn.Linear(classifier_dims[i], classifier_dims[i+1]))
        
    def forward(self, x):
        """
        Forward pass of the classifier.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, hidden_dim).

        Returns:
            Tensor: Output tensor.
        """
        x = self.hdim_fc(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        outputs = self.model(inputs_embeds=x)
        x = outputs.hidden_states[-1]
        for i in range(len(self.classifier_dims) - 1):
            x = getattr(self, f'fc{i+1}')(x)
            x = F.relu(x)
        return x.squeeze(1)  # Remove sequence dimension

if __name__ == '__main__':
    # Configuration parameters for the models
    config = {
        'hidden_dim': 768,
        'classifier_dim': [512, 256, 128, 64, 2]
    }
    # Random input data: batch_size=1, seq_length=10, hidden_dim=768
    data = torch.randn(1, 10, 768)
    # List of model names to test
    model_names = ['MLP', 'DistilGPTClassifier', 'TinyBERTClassifier', 'Qwen2']
    for model_name in model_names:
        # Initialize the model
        model = select_det_model(model_name, config)
        # Get the model output
        output = model(data)
        # Expected output shape
        expected_shape = (1, 10, 2)
        # Verify the output shape
        assert output.shape == expected_shape, (
            f"Model {model_name} failed. Expected shape: {expected_shape}, got: {output.shape}"
        )
    print("All models passed!")

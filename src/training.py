import torch 
import torch.nn as nn
import torch.nn.functional as F
from detection_model import select_det_model
from main_model import select_main_model
from torch.utils.data import Dataset, DataLoader
from dataloader import dataset
import logging
import wandb
import os
import json
import argparse

def create_dir(path):
    """
    Creates a directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def results(config):
    """
    Creates the necessary directories and files to store the results of the training process.
    """
    result_path = os.path.join(config['result_path'],config['detection_model']+'_'+config['main_model']+'_'+config['dataset']+'_'+config['tag'])
    create_dir(result_path)
    # Create a file to log the results
    logging.basicConfig(filename=os.path.join(result_path,'results.log'), level=logging.INFO)
    # Create a folder to save the model
    model_path = os.path.join(result_path,'det_model_weights')
    create_dir(model_path)
    # Save the config file
    with open(os.path.join(result_path,'config.json'), 'w') as f:
        json.dump(config, f)
    # create best model path
    best_model_path = os.path.join(result_path,'best_model.pth')
    config['best_model_path'] = best_model_path
    config['model_path'] = model_path
    return config

def set_config(args):
    """
    Sets the configuration dictionary using the specified configuration file and command line arguments.
    """
    with open(args.config) as f:
        config = json.load(f)
    for arg in vars(args):
        value = getattr(args, arg)
        if value is not None:
            config[arg] = value
    if config.get('debug', False):
        config.update({'wandb': False, 'log': False})
    if config['wandb']:
        wandb.init(project='bad_content_detection', config=config)
    config = results(config)
    return config

def data_processing(data):
    """
    Processes the data to be fed into the model.
    """
    pass

def train(config):
    """
    Trains the detection model using the specified configuration.
    
    Args:
        config (dict): Configuration dictionary containing hyperparameters, 
                       paths, and other settings.
    Returns:
        None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = pd.read_csv(config['dataset_path'])
    # Split the data into train and test
    train_data = data[:int(len(data)*0.8)]
    test_data = data[int(len(data)*0.8):]
    # Initialize the dataloader
    train_dataloader = DataLoader(dataset(train_data,config), batch_size=config['batch_size'], shuffle=True)
    #test_dataloader = DataLoader(dataset(test_data,config), batch_size=config['batch_size'], shuffle=True)
    # Initialize the detection model
    detection_model = select_det_model(config['detection_model'],config).to(device)
    # Initialize the main model
    main_model = select_main_model(config['main_model'],config).to(device)
    # Initialize the optimizer
    optimizer = torch.optim.Adam(detection_model.parameters(), lr=config['lr'])
    # Initialize the loss function
    criterion = nn.CrossEntropyLoss()
    try:
        for epoch in range(config['epochs']):
            total_correct = 0
            total_samples = 0
            for i, data in enumerate(train_dataloader):
                    response, safety_class, token_hidden_states, prompt_hidden_states = data
                    optimizer.zero_grad()
                    state = token_hidden_states[:,config['layer'],config['token'],:]
                    state, labels = state.to(device), labels.to(device)
                    output = detection_model(state)
                    labels = labels.unsqueeze(1).unsqueeze(1).to(torch.float32)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(output.data, 1)
                    total_samples += labels.size(0)
                    total_correct += (predicted == labels).sum().item()
                
            # Calculate epoch accuracy
            epoch_accuracy = 100 * total_correct / total_samples
                
            if config['wandb']:
                wandb.log({
                    "Epoch": epoch, 
                    "Loss": loss,
                    "Accuracy": epoch_accuracy
                })
            if config['log']:
                logging.info(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {epoch_accuracy:.2f}%")
                
            # Save detection model weights every checkpoint_epoch
            if epoch % config['checkpoint_epoch'] == 0:
                checkpoint_path = os.path.join(config['model_path'], f'epoch_{epoch}.pth')
                torch.save(detection_model.state_dict(), checkpoint_path)
                
            # Save best model weights
            if epoch == 0:
                best_accuracy = epoch_accuracy
                torch.save(detection_model.state_dict(), config['best_model_path'])
            elif epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                torch.save(detection_model.state_dict(), config['best_model_path'])
        if config['wandb']:
            wandb.finish()
        if config['log']:
            logging.info(f"Best accuracy: {best_accuracy:.2f}%")
    except Exception as e:
        # send the error to wandb
        if config['wandb']:
            wandb.alert(title="Training Error", text=str(e))
        if config['log']:
            logging.error(str(e))
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config','-c', type=str, default='config.json')
    parser.add_argument('--detection_model','-dm', type=str, default='MLP')
    parser.add_argument('--main_model','-mm', type=str, default='Llama2')
    parser.add_argument('--dataset','-ds', type=str, default='CIFAR10')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--log', type=bool, default=True)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('-l', '--layer', type=int, default=0)
    parser.add_argument('-t', '--token', type=int, default=0)
    args = parser.parse_args()
    config = set_config(args)
    train(config)
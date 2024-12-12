import torch
import json
from types import SimpleNamespace
from detection_model import select_det_model
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_complete_output(prompt, config, max_length=50):
    # Load the tokenizer and model for text generation
    model_name = config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, return_dict_in_generate=True)
    
    # Load and prepare the safety detection model
    safety_detection_model = select_det_model(config['detection_model'], config)
    safety_detection_model.load_state_dict(torch.load(config['detection_model_path']))
    safety_detection_model.eval()
    
    # Load and prepare the refusal detection model
    refusal_detection_model = select_det_model(config['refusal_detection_model'], config)
    refusal_detection_model.load_state_dict(torch.load(config['refusal_detection_model_path']))
    refusal_detection_model.eval()

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids']
    all_hidden_states = []
    unsafe = False
    model.eval()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Generate the next token
            outputs = model(input_ids, output_hidden_states=True)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            all_hidden_states.extend(outputs.hidden_states)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
            # Check for unsafe or refusal tokens at the third iteration
            if _ == 2:
                state = all_hidden_states[-1][-1, -1, :].view(1, -1)
                output = safety_detection_model(state)
                output2 = refusal_detection_model(state)
                _, predicted = torch.max(output.data, 1)
                _, predicted2 = torch.max(output2.data, 1)
                if predicted == 1 or predicted2 == 1:
                    print("Detected unsafe token. Stopping generation.")
                    unsafe = True
                    break
            
            # Stop generation if the end token is generated
            if next_token_id == tokenizer.eos_token_id:
                break

        outputs = SimpleNamespace(
            sequences=input_ids,
            hidden_states=all_hidden_states
        )

    # Decode the generated sequence into text
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    hidden_states = outputs.hidden_states
    
    # Return a message if the prompt is unsafe
    if unsafe:
        generated_text = "The prompt is unsafe. Please try again."
    
    return generated_text, hidden_states

if __name__ == '__main__':
    # Load configuration from JSON file
    config = json.load(open('config/inf_config.json'))
    
    # Take a prompt as input from the user
    print("Welcome to the text generation tool! I am a friendly AI designed to help you answer questions.")
    prompt = input("Enter your question: ")
    
    # Generate output based on the input prompt
    output, hidden_states = generate_complete_output(prompt, config)
    print(output)

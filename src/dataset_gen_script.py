import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
import pickle
import gzip
import tqdm
import shutil  # For file operations
import os
import h5py
import pandas as pd
import os
import torch
import pandas as pd
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
base_model = "meta-llama/Llama-2-7b-chat-hf"
import getpass
from huggingface_hub import notebook_login
notebook_login()
import json
import numpy as np
import gc

gc.collect()
torch.cuda.empty_cache()
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config = quant_config)
model.bfloat16().cuda()
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"
prompts = np.load('checked_prompts.npy', allow_pickle=True)
model.resize_token_embeddings(len(tokenizer))

prompt_template = """
  <s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_message }} [/INST]
  """
prompt_template = prompt_template.replace("{{ system_prompt }}", "Answer the following question.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 25
end = len(prompts)
num_batches = (end + batch_size - 1) // batch_size


def pad_to_max_size(arrays,max_size):
    original_sizes = []
    padded_arrays = []
    for arr in arrays:
        sizes = arr.shape[1]
        original_sizes.append(sizes)  # Save original size
        pad_width = ( (0, 0),(0, max_size - arr.shape[1]), (0, 0))  # Only pad 2nd dimension
        padded = np.pad(arr, pad_width, mode='constant', constant_values=0)
        padded_arrays.append(padded)
    return padded_arrays, original_sizes


res = []
current_size = 0
for i in tqdm.tqdm(range(num_batches)):
    print("i", i)
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, end)
    print("start", start_idx)
    print("end", end_idx)
    batch_prompts = prompts[start_idx:end_idx]
    if not batch_prompts.tolist() or any(not isinstance(p, str) for p in batch_prompts.tolist()):
        print(f"Warning: Skipping batch {i} due to empty or invalid prompts.")
        continue
    print(len(batch_prompts))

    modified_prompts = [
        prompt_template.replace("{{ user_message }}", prompt) for prompt in batch_prompts
    ]
    inputs = tokenizer(modified_prompts, return_tensors="pt", padding=True, truncation = True, is_split_into_words=False)
    inputs.to(device)

    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128, num_return_sequences=1, output_hidden_states = True, return_dict_in_generate=True, temperature = 1.2, top_k = 1000  )
    except Exception as e:
        print("Exception E", e)
        print(f"Warning: Skipping batch {i} due to empty or invalid prompts.")
        del inputs, modified_prompts, batch_prompts
        gc.collect()
        torch.cuda.empty_cache()
        continue

    # Hidden state is organized as (token_no, layer_no, batch_size, input_size, hidden_dim)
    tokenwise_hidden_states = [outputs.hidden_states[t] for t in range(3)]
    layerwise_hs = [[tokenwise_hidden_states[t][-1 * l] for l in range(1, 4)] for t in range(3)]
    seq_wise_hs = [[[layerwise_hs[t][-1 * l][s] for l in range(1, 4)] for t in range(3)] for s in range(batch_size)]


    for _, prompt in enumerate(modified_prompts):
        input_length = inputs.input_ids[_].shape[0]
        output = tokenizer.decode(outputs.sequences[_][input_length:], skip_special_tokens=True)
        # Hidden_states array of: token_no, layer_no, input_dim, hidden_dim
        res.append({"prompt" : prompt, "output" : output, "hidden_state" : seq_wise_hs[_]})

    # Savev every 5 batches
    del inputs, outputs, tokenwise_hidden_states, layerwise_hs, batch_prompts, modified_prompts
    torch.cuda.empty_cache()
    gc.collect()
    # Convert to h5
    # Increase post checkingn.
    if i == 0:
        with h5py.File('hidden_data.h5', 'a') as hdf:
            hdf.create_dataset('token_hidden_states', shape=(0, 3,2,4096), maxshape=(None, 3,2,4096), dtype='float32', compression="gzip", chunks=True)
            hdf.create_dataset('prompt_hidden_states', shape=(0, 3, 65, 4096), maxshape=(None, 3, 65, 4096), dtype='float32', compression="gzip", chunks=True)
            hdf.create_dataset('original_sizes', shape=(0,1), maxshape=(None, 1), dtype='float32', compression="gzip", chunks=True)
    if i % 5 == 0:
        with h5py.File('hidden_data.h5', 'a') as hdf:
            arr1 = []
            arr2 = []
            for i_dash in range(0,len(res)):
                tokens = []
                all_tokens = res[i_dash]['hidden_state']
                # next line is 3
                for j in range(len(all_tokens)):
                    layers = []
                    all_layers = res[i_dash]['hidden_state'][j]
                    # Next loo[ also over three.
                    for k in range(len(all_layers)): 
                        layers.append(res[i_dash]['hidden_state'][j][k].float().cpu().numpy())
                    tokens.append(np.array(layers))
                # 3 loop
                for m in range(len(tokens)):
                    tokens[m] = np.array(tokens[m])
                # Combine the second and third tokens into new array. 
                new_arr = np.stack((tokens[1],tokens[2]),axis=2)
                new_arr = np.squeeze(new_arr,axis=1)
                # arr1 contains first tokens. 
                arr1.append(tokens[0])
                arr2.append(new_arr)


    
            arr1,org = pad_to_max_size(arr1,65)
            arr1 = np.array(arr1)
            # arr1: 
            arr1 = np.swapaxes(arr1,1,2)
            arr1 = arr1.reshape(-1,3,65,4096)    
            org = np.array(org).reshape(-1,1)
    
            arr2 = np.array(arr2)
            arr2 = np.swapaxes(arr2,1,2)
            # three layers, two tokens, hidden_dim.
            arr2 = arr2.reshape(-1,3,2,4096)
            batch_size_write = arr2.shape[0]
            del tokens, all_tokens, layers, all_layers, new_arr
            with h5py.File('hidden_data.h5', 'a') as hdf:
                hdf['token_hidden_states'].resize(current_size + batch_size_write, axis=0)
                hdf['token_hidden_states'][current_size:current_size+batch_size_write] = arr2
                hdf['prompt_hidden_states'].resize(current_size + batch_size_write, axis=0)
                hdf['original_sizes'].resize(current_size + batch_size_write, axis=0)
                hdf['prompt_hidden_states'][current_size:current_size+batch_size_write] = arr1
                hdf['original_sizes'][current_size:current_size+batch_size_write] = org
                current_size += batch_size_write
                print("finished hidden states for batch: ", i)
            del arr1, arr2, org
            df = pd.DataFrame([(item["prompt"], item["output"]) for item in res], columns=['prompt', 'output'])
            df.to_csv('prompt_output.csv', mode='a', index=False, header=not os.path.exists('prompt_output.csv'))
            print("completed wiring prompts for batch: ", i)
            del res, df
            torch.cuda.empty_cache()
            gc.collect()
            res = []

torch.cuda.memory_summary()

























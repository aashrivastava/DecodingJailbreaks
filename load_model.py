from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse

def get_tokenizer(model_name, padding_side=None, device='cpu'):
    '''
    Get the tokenizer for a given model

    Inputs:
    model_name: str
        The name of the model
    
    Outputs:
    tokenizer: transformers.PreTrainedTokenizer
        The tokenizer for the model
    '''
    if padding_side is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
    tokenizer.pad_token = tokenizer.eos_token
        
    # tokenizer.to(device)

    return tokenizer

def get_model(model_name, device='cpu'):
    '''
    Get the model for a given model name

    Inputs:
    model_name: str
        The name of the model
    device: str
        The device to use
    
    Outputs:
    model: transformers.PreTrainedModel
        The model
    '''
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(device)

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-70m',
                        help='Model name to use')
    args = parser.parse_args()
    model = get_model(args.model_name)
    tokenizer = get_tokenizer(args.model_name)
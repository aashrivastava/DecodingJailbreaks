import argparse
import os
import json

import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import DataLoader
from load_model import get_model, get_tokenizer
from tqdm import tqdm

np.random.seed(142)
random.seed(142)

ICL_OCCUPATIONS = [
    'Lunar Botanists',
    'Timepiece Engineers',
    'Cryptoforest Rangers',
    'Antimatter Planners',
    'Subconscious Architects'
]

ICL_OCCUPATIONS_RANGES = {
    'IQ': np.random.randint(50, 150, 5),
    'NetWorth': np.random.randint(1000, 1_000_000, 5),
    'PercentWomen': np.random.randint(0, 100, 5),
    'SubstanceAbuseRate': np.random.randint(0, 100, 5),
    'PercentUnemployed': np.random.randint(0, 100, 5),
    'Weight': np.random.randint(50, 150, 5),
    'YearlyIncome': np.random.randint(500, 1_000_000, 5),
    'NumberOfPeople': np.random.randint(1, 1_000_000, 5),
    'JobSatisfactionLevel': np.random.randint(0, 100, 5),
    'CriminalHistory': np.random.randint(0, 100, 5),
    'DivorceRate': np.random.randint(0, 100, 5),
    'Dishonesty': np.random.randint(0, 100, 5)
}

ICL_POLITICALFIGURES = [
    'Chancellor Elara Voss',
    'Supreme Leader Kwan Jae-Min',
    'High Commissioner Amara Okafor',
    'Grand Vizier Rashid Al-Farsi',
    'Premier Nikolai Volkov'
]

ICL_POLITICALFIGURES_RANGES = {
    'IQ': np.random.randint(50, 150, 5),
    'NetWorth': np.random.randint(1000, 100_000_000, 5),
    'YearsMarried': np.random.randint(0, 60, 5),
    'MileTime': np.random.randint(240, 1200, 5),
    'HoursWorked': np.random.randint(20, 150, 5),
    'NumberOfPeopleKilled': np.random.randint(0, 100_000_000, 5),
    'CorruptionLevel': np.random.randint(0, 100, 5),
    'DollarsInherited': np.random.randint(0, 100_000_000, 5),
    'PercentLies': np.random.randint(0, 100, 5),
    'Height': np.random.uniform(1.5, 2.2, 5),
    'NumberOfCountriesVisited': np.random.randint(1, 180, 5),
    'NumberOfChildren': np.random.randint(0, 20, 5),
    'HoursSlept': np.random.randint(1, 10, 5)
}

ICL_SYNTHETICNAMES = [
    'John Smith',
    'Jane Doe',
    'Michael Brown',
    'Emily Johnson',
    'David Lee',
]

ICL_SYNTHETICNAMES_RANGES = {
    'IQ': np.random.randint(50, 150, 5),
    'NetWorth': np.random.randint(1000, 10_000_000, 5),
    'Height': np.random.uniform(1.5, 2.2, 5),
    'Weight': np.random.randint(40, 150, 5),
    'Age': np.random.randint(0, 100, 5),
    'DailyStepCount': np.random.randint(1000, 20000, 5),
    'Attractiveness': np.random.randint(0, 100, 5),
    'YearlyIncome': np.random.randint(500, 1_000_000, 5),
    'Deadlift': np.random.randint(45, 900, 5),
    'LifeExpectancy': np.random.randint(30, 100, 5),
    'HoursWorked': np.random.randint(0, 100, 5),
    'HoursSlept': np.random.randint(1, 12, 5),
    'AlcoholicDrinksPerWeek': np.random.randint(0, 30, 5),
    'MonthlySpending': np.random.randint(500, 100_000, 5)
}

ICL_COUNTRIES = [
    'Veridonia',
    'Korinthia',
    'Sardinia',
    'Tartaria',
    'Megalopolis',
]

ICL_COUNTRIES_RANGES = {
    'IQ': np.random.randint(60, 140, 5),
    'Corruption': np.random.randint(0, 100, 5),
    'Homicides': np.random.uniform(0, 100, 5),
    'IncomeInequality': np.random.uniform(0, 1, 5),
    'InfantMortality': np.random.uniform(0, 100, 5),
    'Population': np.random.randint(1000, 3_000_000_000, 5),
    'Weight': np.random.uniform(0, 100, 5),
    'Attractiveness': np.random.randint(0, 100, 5),
    'LifeExpectancy': np.random.uniform(25, 100, 5),
    'SubstanceAbuseRate': np.random.uniform(0, 100, 5),
    'HoursWorked': np.random.randint(10, 80, 5)
}


def get_prompts(dataset_path, col_name, prompt_base=None, chunk_number=None):
    '''
    Get the prompts from a dataset

    Inputs:
    dataset_path: str
        The path to the dataset
    col_name: str
        The name of the column containing the prompts
    chunk_number: int
        The chunk number to get prompts for. Assumes chunks of 3000 samples each

    Outputs:
    prompts: list[str]
        The prompts from the dataset
    '''
    df = pd.read_csv(dataset_path)
    prompts = df[col_name].tolist()

    if chunk_number is not None:
        prompts = prompts[chunk_number * 3000: (chunk_number + 1) * 3000]

    if prompt_base is not None:
        prompts = [prompt_base.replace('**', prompt) for prompt in prompts]

    return prompts

@torch.no_grad()
def get_hidden_states(model_name, dataset, batch_size=32, layers='all', verbose=False):
    '''
    Extract the hidden states of a dataset from a given model

    Inputs:
    model_name: str
        The model to extract hidden states from
    dataset: list[str] or similar iterable
        The dataset of prompts to extract hidden states from
    batch_size: int
        The batch size to use
    layers: str or int or List[int]
        The layers to extract hidden states from. If 'all', extract hidden states from all layers
    verbose: bool
        Whether to print out progress
    
    Outputs:
    hidden_states: torch.Tensor
        The hidden states of the dataset
    '''
    # preliminaries
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f'Getting model: {model_name}')
    model = get_model(model_name, device=device)
    tokenizer = get_tokenizer(model_name, device=device)
    if verbose:
        print('Model loaded.')

    if layers == 'all':
        layers = list(range(model.config.num_hidden_layers))
    if type(layers) == int:
        layers = [layers]

    # tokenize the dataset
    if verbose:
        print('Tokenizing dataset...')
    inputs = tokenizer(dataset, return_tensors='pt', padding=True, truncation=True)
    input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

    if verbose:
        print('Dataset tokenized.')

    # create dataloader
    tokenized_dataloader = DataLoader(input_ids, batch_size=batch_size)
    attn_dataloader = DataLoader(attention_mask, batch_size=batch_size)

    # each key represents a layer. Each value is a tensor of shape [dataset_size, hidden_size]
    # this stores the last token hidden states for each datapoint through the layers
    hidden_states = {l: torch.zeros(len(dataset), model.config.hidden_size) for l in layers}

    curr_idx = 0
    for batch_input_ids, batch_attn_mask in tqdm(zip(tokenized_dataloader, attn_dataloader), disable=not verbose):
       # get model outputs
        outputs = model(batch_input_ids, attention_mask=batch_attn_mask, 
            output_hidden_states=True, output_attentions=False,
            return_dict=True
        )

        # find last relevant token
        last_seq_idx = batch_attn_mask.shape[1] - 1
        batch_attn_mask = batch_attn_mask.to(int)
        last_relevant_tokens = torch.argmin(batch_attn_mask, dim=1) - 1

        for lix, hidden_state in enumerate(outputs.hidden_states):
            batch_size, seq_len, hidden_size = hidden_state.shape

            if lix not in layers:
                continue

            expanded_mask = last_relevant_tokens.unsqueeze(-1).expand(-1, hidden_size)

            last_token_hidden_state = hidden_state[
                torch.arange(batch_size).unsqueeze(-1),
                expanded_mask,
                torch.arange(hidden_size)
            ]

            hidden_states[lix][curr_idx: curr_idx + batch_size] = last_token_hidden_state

        curr_idx += batch_size
        if verbose:
            print(f'Processed {curr_idx} out of {len(dataset)} prompts')
    
    return hidden_states

def get_args():
    parser = argparse.ArgumentParser(description='Get hidden states')

    parser.add_argument(
        '-m', '--model', 
        type=str, 
        required=True,
        help='The model to extract hidden states from'
    )
    parser.add_argument(
        '-e', '--entity',
        type=str,
        required=True,
        help='The type of entity we are asking about. (e.g., Country, Occupations, etc...)'
    )
    parser.add_argument(
        '-q', '--question_id',
        type=str,
        default=None,
        help='Question id. Largely refers to what is question_id in the jsonl files defining questions.'
    )
    parser.add_argument(
        '-c', '--col_name',
        type=str,
        default='main_prompt',
        help='Column name where prompts are stored in dataset. (e.g., entity_prompt, main_prompt). Used only when -q provided.'
    )
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        default=32,
        help='The batch size to use'
    )
    parser.add_argument(
        '-l', '--layers',
        type=int,
        nargs='*',
        default=-1, # indicates all
        help='The layers to extract hidden states from. If -1, extract hidden states from all layers'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Whether to print out progress'
    )
    parser.add_argument(
        '-t', '--test',
        action='store_true',
        help='Whether to save the hidden states'
    ) # if testing, WE DO NOT SAVE THE HIDDEN STATES
    parser.add_argument(
        '-j', '--jailbreak_prompt',
        type=str,
        default=None,
        help='The jailbreak prompt to use. If not provided, we do not use a jailbreak prompt.'
    )

    args = parser.parse_args()

    return args

        
if __name__ == '__main__':
    # AUTOMATICALLY SAVES THE HIDDEN STATES INTO THE FOLLOWING DIRECTORY:
    # data/dataset_name/hidden_states/model_name.pt
    # model_name = 'EleutherAI/pythia-70m'
    # dataset = ['hello world', 'goodbye world', 'please cite our work if you find this code useful!']
    # get_hidden_states(model_name, dataset)
    args = get_args()

    model = args.model
    dataset = args.question_id
    entity = args.entity
    col_name = args.col_name
    batch_size = args.batch_size
    layers = args.layers
    verbose = args.verbose
    to_save = not args.test
    jailbreak_prompt = args.jailbreak_prompt
    if args.jailbreak_prompt is not None:
        jailbreak_name = args.jailbreak_prompt.split('.')[0]
    else:
        jailbreak_name = None

    if type(layers) == int:
        if layers == -1:
            layers = 'all'
        else:
            layers = [layers]

    # # get actual dataset from dataset name. Assumes code was run from github repo root
    if dataset is None:
        with open(os.path.join('data', entity, f'{entity}_list.txt'), 'r') as f:
            entities = [line.strip() for line in f.readlines()]

        prompts = [f'This document describes {entity}' for entity in entities]
    else:
        if jailbreak_prompt is not None:
            if '.txt' in jailbreak_prompt:
                jailbreaking_type = f"{jailbreak_prompt.split('.')[0]}Jailbreak_"
                # icl.txt doesn't exist. We take the question base for the appropriate entity, question_id pair.
                if jailbreak_prompt == 'icl.txt':
                    with open(os.path.join('data', entity, f'{entity}_questions.jsonl'), 'r') as f:
                        for line in f:
                            question_info = json.loads(line)
                            if question_info['question_id'] == dataset:
                                
                                question = question_info['question']

                                jailbreak_prompt = ""
                                if entity.lower() == 'occupations':
                                    icl_list = ICL_OCCUPATIONS
                                    icl_ranges = ICL_OCCUPATIONS_RANGES
                                elif entity.lower() == 'politicalfigures':
                                    icl_list = ICL_POLITICALFIGURES
                                    icl_ranges = ICL_POLITICALFIGURES_RANGES
                                elif entity.lower() == 'syntheticnames':
                                    icl_list = ICL_SYNTHETICNAMES
                                    icl_ranges = ICL_SYNTHETICNAMES_RANGES
                                elif entity.lower() == 'countries':
                                    icl_list = ICL_COUNTRIES
                                    icl_ranges = ICL_COUNTRIES_RANGES
                                for i, entity_name in enumerate(icl_list):
                                    jailbreak_prompt += (question.replace('**', entity_name) + f'{icl_ranges[dataset][i]}.\n')
                                    
                                jailbreak_prompt += '**'
                else:
                    with open(f'jailbreaking_prompts/{jailbreak_prompt}', 'r') as f:
                        jailbreak_prompt = f.read()
            else:
                jailbreak_prompt = jailbreak_prompt
                jailbreaking_type = 'userSpecifiedJailbreak_'
        else:
            jailbreaking_type = ''

        dataset_path = os.path.join('data', entity, f'{entity}{dataset}', f'{entity}{dataset}.csv')
        prompts = get_prompts(dataset_path, col_name, prompt_base=jailbreak_prompt)

    hidden_states = get_hidden_states(model, prompts, batch_size=batch_size, layers=layers, verbose=verbose)

    if to_save:
        model_family = model.split('/')[0]
        model_name = model.split('/')[-1]
        if dataset is None:
            save_path = os.path.join('data', entity, 'hidden_states', model_family, f'{model_name}.pt')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        else:
            if jailbreak_prompt is not None:
                save_path = os.path.join('data', entity, f'{entity}{dataset}', 'hidden_states', jailbreak_name, model_family, f'{model_name}.pt')
            else:
                save_path = os.path.join('data', entity, f'{entity}{dataset}', 'hidden_states', model_family, f'{model_name}.pt')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(hidden_states, save_path)
        if verbose:
            print(f'Hidden states saved to {save_path}')
import argparse
import os
import re

from word2number import w2n

import pandas as pd
import numpy as np
from tqdm import tqdm

def parse_first_number(generation, dtype='float'):
    '''
    Parse the first number present at the very start of a generation.

    Inputs:
    generation: str
        The generation to parse
    dtype: float
        The datatype to return the parsed number as
    
    Outputs:
    parsed number: float or int or np.nan
        The parsed number or np.nan if no number is found
    '''
    # Use regex to find a number at the start of the text
    match = re.match(r'^\d+(?:\.\d+)?', generation)
    if match:
        if dtype == 'float':
            return float(match.group())
        elif dtype == 'int':
            return int(match.group())
        else:
            raise ValueError(f'Invalid dtype: {dtype}')
    return None

def parse_first_number_after_aim(generation, dtype='float'):
    '''
    Parse the first number present in a generation after the AIM prompt
    '''
    # Find the index of the AIM prompt
    first_aim_index = generation.find('AIM:')
    if first_aim_index == -1:
        return np.nan
    # Find the index of the first number after the AIM prompt
    number_words = r'(?:hundred|thousand|million|billion|trillion)'
    match = re.search(r'\d+(?:\s*,\s*\d{3})*(?:\s*\.\s*\d+)?(?:\s*' + number_words + r')?', generation[first_aim_index:], re.IGNORECASE)
    result = match.group() if match else None
    if match:
        # Remove commas from the result
        result = result.replace(' ,', '')
        result = result.replace(', ', '')
        result = result.replace(',', '')

        # Remove decimal followed by a space
        decimal_space_match = re.search(r'(\d+)\.(\s+)', result)
        if decimal_space_match:
            # Keep only the part before the decimal
            result = decimal_space_match.group(1)
        
        # Handle words like million, billion, etc.
        if re.search(number_words, result, re.IGNORECASE):
            # Extract the numeric part and the multiplier word
            num_match = re.search(r'\d+(?:\.\d+)?', result)
            word_match = re.search(number_words, result, re.IGNORECASE)
            
            if num_match and word_match:
                num_value = float(num_match.group())
                multiplier_word = word_match.group().lower()
                
                # Apply the appropriate multiplier
                if multiplier_word == 'thousand':
                    result = str(num_value * 1000)
                elif multiplier_word == 'million':
                    result = str(num_value * 1000000)
                elif multiplier_word == 'billion':
                    result = str(num_value * 1000000000)
                elif multiplier_word == 'trillion':
                    result = str(num_value * 1000000000000)
                elif multiplier_word == 'hundred':
                    result = str(num_value * 100)

        if dtype == 'float':
            return float(result)
    else:
        return np.nan

def parse_string_after_aim(generation):
    '''
    Parse the string after the AIM prompt
    '''
    # Find the index of the AIM prompt
    first_aim_index = generation.find('AIM:')
    # Skip the "AIM:" prefix if it exists
    if first_aim_index != -1:
        # Find the position after "AIM:" (adding 4 characters for "AIM:")
        start_index = first_aim_index + 4
        result = generation[start_index:].strip()
        # Remove periods and commas
        result = result.replace('.', '')
        result = result.replace(',', '')
        return result
    else:
        return np.nan


def parse_comparative_choice_aim(row, model, jailbreaking_type):
    '''
    Parses the comparative choice of the model from the AIM prompt.

    We consider the choice under the following cases:
    - The choice is made after the AIM prompt and immediately followed by punctuation. (1)
    - The choice is made after the AIM prompt and is followed by something like "has/have a/the [one word] higher" (2)
    - The opposite choice is made after the AIM prompt and is followed by something like "has/have a/the [one word] lower" (3)
    - The choice is made after the AIM prompt and is of the form "with the higher [] is [CHOICE]" (4)
    - The choice is made after the AIM prompt and is of the form "with the lower [] is [CHOICE]" (5)
    - The choice is made after the AIM prompt and is of the form "[CHOICE] is the correct" (6)

    Inputs:
        - Entire row from the main dataframe
    
    Outputs:
        The parsed generation or np.nan if no choice is found
    '''
    generation = row[f'{model}_{jailbreaking_type}response']
    if generation is np.nan:
        return np.nan

    entityA = row['entityA']
    entityB = row['entityB']
    
    if 'gemma' in model.lower():
        # Find the first occurrence of "AIM:" and look for entity immediately after
        pattern = rf"AIM:\s*({re.escape(entityA)}|{re.escape(entityB)})"
        match = re.search(pattern, generation, re.IGNORECASE)
        if match:
            return entityA if match.group(1).lower() == entityA.lower() else entityB
        else:
            return np.nan

    # case 1: entityA or entityB is at the start of the generation immediately followed by punctuation
    pattern_1 = rf"AIM:\s*({re.escape(entityA)}|{re.escape(entityB)})(?=[.,])"
    match_1 = re.search(pattern_1, generation, re.IGNORECASE)
    if match_1:
        return entityA if match_1.group(1).lower() == entityA.lower() else entityB
    
    # FROM HERE ON OUT WE WANT TO ONLY CONSIDER TEXT AFTER "AIM:"
    generation = generation[generation.find('AIM:')+4:].strip()

    # case 2: entityA or entityB is at the start of the generation followed by "has/have a/the [one word] higher" or "has/have a/the [one word] lower"
    pattern_2 = rf"({re.escape(entityA)}|{re.escape(entityB)})\s+\w*\s*(has|have)\s+(a|the)?\s*\w*\s*higher"
    match_2 = re.search(pattern_2, generation, re.IGNORECASE)
    if match_2:
        return entityA if match_2.group(1).lower() == entityA.lower() else entityB
    
    # case 3: entityA or entityB is at the start of the generation followed by "has/have a/the [one word] lower"
    pattern_3 = rf"({re.escape(entityA)}|{re.escape(entityB)})\s+\w*\s*(has|have)\s+(a|the)?\s*\w*\s*lower"
    match_3 = re.search(pattern_3, generation, re.IGNORECASE)
    if match_3:
        if match_3.group(1).lower() == entityA.lower():
            return entityB
        else:
            return entityA

    # case 4: entityA or entityB is at the start of the generation followed by "with the higher [one word] is [CHOICE]"
    pattern_4 = rf"with\s+(a|the)\s+higher\s+(.+?)\s+is\s+\w*\s*({re.escape(entityA)}|{re.escape(entityB)})"
    match_4 = re.search(pattern_4, generation, re.IGNORECASE)
    if match_4:
        return entityA if match_4.group(3).lower() == entityA.lower() else entityB
    
    # case 4: entityA or entityB is at the start of the generation followed by "with the lower [one word] is [one word] [CHOICE]"
    pattern_5 = rf"with\s+(a|the)\s+lower\s+(.+?)\s+is\s+\w*\s*({re.escape(entityA)}|{re.escape(entityB)})"
    match_5 = re.search(pattern_5, generation, re.IGNORECASE)
    if match_5:
        if match_5.group(3).lower() == entityA.lower():
            return entityB
        else:
            return entityA
    
    
    # case 6: The choice is made after the AIM prompt and is of the form "[CHOICE] is the correct"
    pattern_6 = rf"({re.escape(entityA)}|{re.escape(entityB)})\s+is\s+the\s+correct"
    match_6 = re.search(pattern_6, generation, re.IGNORECASE)
    if match_6:
        return entityA if match_6.group(1).lower() == entityA.lower() else entityB

    # case 7: The choice is made when the word "tends to" is used
    pattern_7 = rf"({re.escape(entityA)}|{re.escape(entityB)})\s+tends?\s+to\s+have\s+(?:\w+\s+)*higher"
    match_7 = re.search(pattern_7, generation, re.IGNORECASE)
    if match_7:
        return entityA if match_7.group(1).lower() == entityA.lower() else entityB

    # case 8: The choice is made when the word "tends to" is used
    pattern_8 = rf"({re.escape(entityA)}|{re.escape(entityB)})\s+tends?\s+to\s+have\s+(?:\w+\s+)*lower"
    match_8 = re.search(pattern_8, generation, re.IGNORECASE)
    if match_8:
        return entityB if match_8.group(1).lower() == entityA.lower() else entityA
    
    return np.nan

def parse_comparative_choice_icl(row, model, jailbreaking_type):
    generation = row[f'{model}_{jailbreaking_type}response']
    if generation is np.nan:
        return np.nan

    # account for synthetic names structure
    entityA = row['entityA'].replace('the fictional person named ', '')
    entityB = row['entityB'].replace('the fictional person named ', '')


    pattern = rf"^({re.escape(entityA)}|{re.escape(entityB)})"
    match = re.search(pattern, generation, re.IGNORECASE)

    if match:
        if 'the fictional person named' in row['entityA']:
            winner = entityA if match.group(1).lower() == entityA.lower() else entityB
            return f'the fictional person named {winner}'
        else:
            return entityA if match.group(1).lower() == entityA.lower() else entityB
    
    return np.nan
    
    


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-e', '--entity',
        type=str,
        required=True,
        help='Type of entity we are asking about. (e.g., Country, Occupations, etc...)'
    )
    parser.add_argument(
        '-q', '--question_id',
        type=str,
        required=True,
        help='Question id. Largely refers to what is question_id in the jsonl files defining questions.'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        required=True,
        help='model name whose responses we want to parse'
    )
    parser.add_argument(
        '-j', '--jailbreak_prompt',
        type=str,
        required=False,
        help='Jailbreak base prompt to get response for. Either provide .txt file or string'
    )
    parser.add_argument(
        '-dtype', '--data_type',
        type=str,
        default='float',
        help='data type of the labels of dataset'
    )
    parser.add_argument(
        '-p', '--pairs',
        action='store_true',
        help='Whether to parse the pairs of generations'
    )
    parser.add_argument(
        '-f', '--first_number_only',
        action='store_true',
        help='Whether to only parse the first number'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Whether to print verbose output'
    )
    
    return parser.parse_args()
if __name__ == '__main__':
    args = get_args()

    entity = args.entity
    dataset = args.question_id
    model = args.model
    jailbreak_prompt = args.jailbreak_prompt
    data_type = args.data_type
    using_pairs = args.pairs
    first_number_only = args.first_number_only
    verbose = args.verbose
    
    if using_pairs:
        pairs = '_pairs'
    else:
        pairs = ''

    if jailbreak_prompt is not None:
        if '.txt' in jailbreak_prompt:
            jailbreaking_type = f"{jailbreak_prompt.split('.')[0]}Jailbreak_"
        else:
            jailbreaking_type = f"userSpecifiedJailbreak_"
    else:
        jailbreaking_type = ''

    dataset_path = os.path.join('data', entity, f'{entity}{dataset}', f'{entity}{dataset}{pairs}.csv')
    df = pd.read_csv(dataset_path)
    try:
        generations = df[f'{model}_{jailbreaking_type}response']
    except Exception as e:
        print(f'Not found: {entity}, {dataset}')
        exit()
    if jailbreak_prompt is not None:
        if 'machiavelli' in jailbreak_prompt:
            if not using_pairs:
                if first_number_only:
                    parsed_generations = generations.apply(lambda x: parse_first_number(str(x), dtype=data_type))
                else:
                    parsed_generations = generations.apply(
                        lambda x: (
                            result if (result := parse_first_number_after_aim(str(x), dtype=data_type)) is not np.nan 
                            else parse_first_number(str(x), dtype=data_type)
                        )
                    )
            else:
                if verbose:
                    print(f'parsing aim pairs for {entity}, {dataset}')
                parsed_generations = df.apply(lambda x: parse_comparative_choice_aim(x, model, jailbreaking_type), axis=1)
        else:
            try:
                if not using_pairs:
                    parsed_generations = generations.apply(lambda x: parse_first_number(str(x), dtype=data_type))
                else:
                    if verbose:
                        print(f'parsing icl pairs for {entity}, {dataset}')
                    parsed_generations = df.apply(lambda x: parse_comparative_choice_icl(x, model, jailbreaking_type), axis=1)
            except Exception as e:
                print(f'Not found: {entity}, {dataset}')
                raise
    else:
        parsed_generations = generations.apply(lambda x: parse_first_number(str(x), dtype=data_type))

    df[f'{model}_{jailbreaking_type}response_parsed'] = parsed_generations
    df.to_csv(dataset_path, index=False)
    if verbose:
        print(model, entity, jailbreaking_type, 'parsed')

import pandas as pd
import random
import os
import json
import argparse


def create_entity_data(entity_name):
    '''
    Creates individual datasets for each question associated with an entity. Relies on entity list and entity questions files.

    Inputs:
        entity_name: str
            The name of the entity to create datasets for
            e.g., 'Occupations', 'Country', 'syntheticNames', 'politicalFigures'
    
    Outputs: None
        Saves datasets to data/entity_name within their own directories.
    '''
    entity_path = os.path.join('data', entity_name)
    
    with open(os.path.join(entity_path, f'{entity_name}_list.txt'), 'r') as f:
        entities = [line.strip() for line in f.readlines()]

    questions = []
    with open(os.path.join(entity_path, f'{entity_name}_questions.jsonl'), 'r') as f:
        for line in f:
            question_info = json.loads(line)
            questions.append((question_info['question_id'], question_info['question']))

    for q in questions:
        df = pd.DataFrame({entity_name.lower(): entities})
        df['entity_prompt'] = df.apply(lambda row: f'This document describes {row[entity_name.lower()]}', axis=1)
        df['main_prompt'] = df.apply(lambda row: q[1].replace('**', row[entity_name.lower()]), axis=1)
        df['is_train'] = 0
        train_idx = df.sample(frac=0.8).index
        df.loc[train_idx, 'is_train'] = 1
        # Create directory if it doesn't exist
        output_dir = os.path.join(entity_path, f'{entity_name}{q[0]}')
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, f'{entity_name}{q[0]}.csv'), index=False)

def create_entity_pairs(entity_name):
    '''
    Creates pairs of entities for each question associated with an entity. Relies on entity list.
    This is a helper function.

    Inputs:
        entity_name: str
            The name of the entity to create pairs for
    Outputs: list of tuples
        Each tuple contains a pair of entities
    '''
    with open(os.path.join('data', entity_name, f'{entity_name}_list.txt'), 'r') as f:
        entities = [line.strip() for line in f.readlines()]
    pairs = []
    for i in range(len(entities)):
        for j in range(i+1, len(entities)):
            pairs.append((entities[i], entities[j]))
    return pairs

def create_entity_pairs_data(entity_name):
    '''
    Creates the pairs datasets for a given entity on all questions.

    Inputs:
        entity_name: str
            The name of the entity to create pairs for
    Outputs: None
        Saves datasets to data/entity_name within their own directories.
    '''
    pairs = create_entity_pairs(entity_name)
    # Sample 15k pairs if there are more than 15k pairs
    if len(pairs) > 15000:
        pairs = random.sample(pairs, 15000)
    
    questions = []
    with open(os.path.join('data', entity_name, f'{entity_name}_questions.jsonl'), 'r') as f:
        for line in f:
            question_info = json.loads(line)
            questions.append((question_info['question_id'], question_info['comparison_question']))

    for q_id, q_comp in questions:
        df = pd.DataFrame({'entityA': [pair[0] for pair in pairs], 'entityB': [pair[1] for pair in pairs]})
        df['prompt'] = df.apply(lambda row: f'{q_comp} {row["entityA"]} or {row["entityB"]}: ', axis=1)
        
        df.to_csv(os.path.join('data', entity_name, f'{entity_name}{q_id}', f'{entity_name}{q_id}_pairs.csv'), index=False)
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create datasets')
    parser.add_argument(
        '-e', '--entity',
        type=str,
        nargs='+',
        help='Which datasets to create. (e.g., Occupations, Countries, etc...)'
    )
    parser.add_argument(
        '-p', '--pairs',
        action='store_true',
        help='Whether to create pairs datasets'
    )
    args = parser.parse_args()
    entities = args.entity
    doing_pairs = args.pairs
    # if none given, exit
    if entities is None:
        print('No entity given. Exiting...')
        exit()
    elif entities[0] == 'all':
        entities = ['Occupations', 'Countries', 'politicalFigures', 'syntheticNames']


    for entity in entities:
        if doing_pairs:
            create_entity_pairs_data(entity)
        else:
            create_entity_data(entity)

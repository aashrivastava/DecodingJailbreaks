import argparse
import os
from tqdm import tqdm

import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
import numpy as np

ALPHAS = np.logspace(-3, 8, 20)

def train_regression_probe(hidden_states, mask, labels, probe_across=False, predict_hidden_states=None):
    '''
    Train regression probes and predict on the hidden states given the labels.

    Inputs:
        hidden_states: torch.Tensor
            The hidden states to train and evaluate the probe on
        mask: torch.Tensor
            The mask to apply to the hidden states
        labels: torch.Tensor
            The labels to train and evaluate the probe on
        probe_across: bool
            Set to True in the case you want to fit on one model's hidden states and predict on another model's
        predict_hidden_states: torch.Tensor or None
            This should only be passed if probe_across is True. It should be the hidden states of the model you want to predict on.
    
    Returns:
        probe, and predictions
    '''
    if not probe_across:
        train_states = hidden_states[mask.bool()]
        train_labels = labels[mask.bool()]
    
    else:
        train_states = hidden_states
        train_labels = labels

    try:
        probe = RidgeCV(alphas=ALPHAS, store_cv_results=True, cv=None)
        probe.fit(train_states, train_labels)
        # print(probe.alpha_)
        # print(probe.best_score_)
    except Exception as e:
        raise e

    if not probe_across:
        return probe, probe.predict(hidden_states)
    else:
        assert predict_hidden_states is not None
        return probe, probe.predict(predict_hidden_states)

def train_classification_probe(hidden_states, mask, labels, probe_across=False, predict_hidden_states=None):
    '''
    Train and evaluate classification probes on the hidden states given the labels.
    Uses ridge classifier because sklearn logistic regression had problems converging.
    Also, we are dealing with high dimensional, multicollinear data, so ridge is better.

    Inputs:
        hidden_states: torch.Tensor
            The hidden states to train and evaluate the probe on
        mask: torch.Tensor
            The mask to apply to the hidden states
        labels: torch.Tensor
            The labels to train and evaluate the probe on
            First idx is the model name of the states
            Second idx is the layer index of the states
        probe_across: bool
            Set to True in the case you want to fit on one model's hidden states and predict on another model's
        predict_hidden_states: torch.Tensor or None
            This should only be passed if probe_across is True. It should be the hidden states of the model you want to predict on.
    
    Returns:
        probe: sklearn.linear_model.RidgeClassifier
            The trained probe
        probe_predictions: torch.Tensor
            The predictions of the probe on the hidden states

        Typically only concerned with probe_predictions
    '''
    if not probe_across:
        train_states = hidden_states[mask.bool()]
        test_states = hidden_states[~mask.bool()]
        train_labels = labels[mask.bool()]
        test_labels = labels[~mask.bool()]
    else:
        train_states = hidden_states
        test_states = predict_hidden_states
        train_labels = labels
        test_labels = None

    probe = RidgeClassifierCV(alphas=ALPHAS, store_cv_results=True, cv=None)
    probe.fit(train_states, train_labels)

    if not probe_across:
        return probe, probe.predict(hidden_states)
    else:
        return probe, probe.predict(predict_hidden_states)

class MLP(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim=128):
        super().__init__()
        layers = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # First layer: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LeakyReLU(0.1))
        
        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.1))
            
        # Final layer: hidden_dim -> 1
        layers.append(nn.Linear(hidden_dim, 1))
        self.layers = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, 1) containing predicted values
        """
        return self.layers(x).squeeze()

    def fit(self, train_states, train_labels, test_states=None, test_labels=None, 
                   epochs=100, batch_size=128, lr=1e-3):
        """
        Train the MLP regression model.
        
        Args:
            train_states: Training input features
            train_labels: Training labels
            test_states: Optional test input features for validation
            test_labels: Optional test labels for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            
        Returns:
            Dictionary containing training history
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.MSELoss()
        
        # Convert to tensors if not already and ensure float32 dtype
        if not torch.is_tensor(train_states):
            train_states = torch.tensor(train_states, dtype=torch.float32)
            train_labels = torch.tensor(train_labels, dtype=torch.float32)
            if test_states is not None:
                test_states = torch.tensor(test_states, dtype=torch.float32)
                test_labels = torch.tensor(test_labels, dtype=torch.float32)
        else:
            train_states = train_states.float()
            train_labels = train_labels.float()
            if test_states is not None:
                test_states = test_states.float()
                test_labels = test_labels.float()
                
        # Move data to GPU
        train_states = train_states.to(device)
        train_labels = train_labels.to(device)
        if test_states is not None:
            test_states = test_states.to(device)
            test_labels = test_labels.to(device)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(train_states, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if test_states is not None:
            test_dataset = torch.utils.data.TensorDataset(test_states, test_labels)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            history['train_loss'].append(avg_loss)
            
            # Validation
            if test_states is not None:
                self.eval()
                val_total_loss = 0
                val_num_batches = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in test_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        val_outputs = self(batch_x)
                        val_loss = criterion(val_outputs, batch_y)
                        val_total_loss += val_loss.item()
                        val_num_batches += 1
                        
                avg_val_loss = val_total_loss / val_num_batches
                history['val_loss'].append(avg_val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            else:
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}')
                    
        return history

def train_regression_mlp(hidden_states, mask, labels, master_df=None, states_info=None, n_layers=2, hidden_dim=128):
    '''
    Train and evaluate regression probes on the hidden states given the labels.
    Also saves predictions on the master_df if provided

    Inputs:
        hidden_states: torch.Tensor
            The hidden states to train and evaluate the probe on
        mask: torch.Tensor
            The mask to apply to the hidden states
        labels: torch.Tensor
            The labels to train and evaluate the probe on
        master_df: pd.DataFrame
            The master dataframe to save the probe predictions to
        states_info: tuple(str, str)
            The information about the probe to save to the master_df
            First idx is the model name of the states
            Second idx is the layer index of the states
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_states = hidden_states[mask.bool()]
    test_states = hidden_states[~mask.bool()]
    train_labels = labels[mask.bool()]
    test_labels = labels[~mask.bool()]

    input_dim = train_states.shape[1]

    model = MLP(n_layers, input_dim, hidden_dim=hidden_dim)
    model.to(device)
    model.fit(train_states, train_labels)

    if master_df is not None:
        model_name, layer_idx = states_info
        hidden_states = hidden_states.to(device)
        probe_predictions = model(hidden_states.float()).detach().cpu() # gets test and train predictions
        return model, probe_predictions
    else:
        hidden_states = hidden_states.to(device)
        return model, model(hidden_states.float()).detach().cpu()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train probes')
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
        '-p', '--prompts_col',
        type=str,
        default='main_prompt',
        help='The column name of the prompts'
    )
    parser.add_argument(
        '--mask_col',
        type=str,
        default='is_train',
        help='The train/test mask column name'
    )
    parser.add_argument(
        '-l', '--labels_col',
        type=str,
        required=True,
        help='The column name of the training labels. Can be inferred from model name, entity, question_id, and jailbreaking info. Future version will not require this under this inference scheme.'
    )
    parser.add_argument(
        '--master_df',
        type=str,
        default='',
        help='Should be same as --data if provided.'
    )
    parser.add_argument(
        '--mlp',
        action='store_true',
        help='Whether to use MLP for regression'
    )
    parser.add_argument(
        '--n_layers',
        type=int,
        default=2,
        help='The number of layers for the MLP. Only used if --mlp is True.'
    )
    # add argument for hidden_dim
    parser.add_argument(
        '-hd', '--hidden_dim',
        type=int,
        default=128,
        help='The number of hidden dimensions for the MLP. Only used if --mlp is True.'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='probe_results',
        help='The name of the experiment. More or less required, if not given will be set to general: "probe_results".'
    )
    parser.add_argument(
        '--classification',
        action='store_true',
        help='Whether to use classification probe'
    )
    parser.add_argument(
        '-m', '--model_name',
        type=str,
        required=True,
        help='The name of the model generations to train probe on (e.g. meta-llama/Llama-3.1-8B). Specify model if wanting to do analysis on a single model. Otherwise use "all".'
    )
    parser.add_argument(
        '--probe_across',
        default='',
        help='Used when training probe on one model and predicting on another. Should be the model name of the one you want to predict on.'
    )
    parser.add_argument(
        '--probe_specific',
        default='',
        help='Indicates whether to train probes on jailbreaking specific hidden states. If so, name the jailbreaking method.'
    )
    parser.add_argument(
        '--save_probe',
        action='store_true',
        help='Whether to save the probe. Only saves the coefficients in order to check cosine similarities.'
    )

    args = parser.parse_args()

    entity = args.entity
    dataset = args.question_id
    prompts_col = args.prompts_col
    mask_col = args.mask_col
    labels_col = args.labels_col
    master_df_path = args.master_df
    use_mlp = args.mlp
    n_layers = args.n_layers
    hidden_dim = args.hidden_dim
    if n_layers == 1:
        hidden_dim = 1
    experiment_name = args.experiment_name
    is_classification = args.classification
    model_name = args.model_name
    probe_across = args.probe_across
    probe_specific = args.probe_specific
    save_probe = args.save_probe

    # load data
    data_path_full = os.path.join('data', entity, f'{entity}{dataset}')
    data = pd.read_csv(f'{data_path_full}/{entity}{dataset}.csv')
    # drop nan rows in prompts_col, mask_col, and labels_col
    # get indices of nan rows
    nan_indices = data[data[prompts_col].isna() | data[labels_col].isna()].index
    inf_indices = data[data[labels_col] == float('inf')].index
    data = data.drop(nan_indices).drop(inf_indices)
    data = data.reset_index(drop=True)

    prompts = data[prompts_col]
    mask = torch.tensor(data[mask_col].to_numpy())
    labels = torch.tensor(data[labels_col].to_numpy())

    # create master_df
    if master_df_path == '':
        master_df = pd.DataFrame()
        master_df[prompts_col] = prompts
        master_df[mask_col] = mask
        master_df[labels_col] = labels
    # avoid this at most costs
    else:
        master_df = pd.read_csv(f'{data_path_full}/{entity}{dataset}_{experiment_name}.csv')

    if model_name == 'all':
        models = {
            'EleutherAI': ['pythia-160m', 'pythia-2.8b', 'pythia-6.9b'],
            'google': ['gemma-7b-it', 'gemma-7b'],
            'meta-llama': ['Llama-3.1-8B-Instruct', 'Llama-3.1-8B']
        }
    else: # assumes a single model
        model_family = model_name.split('/')[0]
        model_type = model_name.split('/')[-1]
        models = {model_family: [model_type]}
    
    if save_probe:
        probe_weights = {}

    master_df_cols = {}
    for model_family, models_list in tqdm(models.items(), desc='Training probes...'):
        for model_name in models_list:
            # always used for training
            if probe_specific != '':
                jailbreak_method = probe_specific.split('.')[0]
                model_states = torch.load(os.path.join('data', entity, f'{entity}{dataset}', 'hidden_states', jailbreak_method, model_family, f'{model_name}.pt'),
                    weights_only=True)
            else:
                model_states = torch.load(os.path.join('data', entity, 'hidden_states', model_family, f'{model_name}.pt'),
                    weights_only=True)
            if probe_across != '':
                # these are used for predictions
                model_family_probe_across = probe_across.split('/')[0]
                model_name_probe_across = probe_across.split('/')[-1]
                model_states_probe_across = torch.load(os.path.join('data', entity, 'hidden_states', model_family_probe_across, f'{model_name_probe_across}.pt'),
                    weights_only=True)
                probe_across = True
            else:
                model_states_probe_across = None
                probe_across = False
            for layer_idx, hidden_states in tqdm(model_states.items(), desc='Going through layers...'):
                save_info = (f'{model_family}/{model_name}', layer_idx)
                # Remove rows corresponding to nan and inf indices from hidden states
                valid_indices = torch.ones(hidden_states.shape[0], dtype=torch.bool)
                valid_indices[nan_indices] = False
                valid_indices[inf_indices] = False

                hidden_states = hidden_states[valid_indices]
                if not use_mlp:
                    if is_classification:
                        print('Using classification probe')
                        try:
                            probe, preds = train_classification_probe(hidden_states, mask, labels, 
                                probe_across=probe_across, predict_hidden_states=model_states_probe_across[layer_idx])
                        except TypeError as e: # in the case of no probe_across
                            probe, preds = train_regression_probe(hidden_states, mask, labels, probe_across=probe_across, predict_hidden_states=None)
                    else:
                        try:
                            probe, preds = train_regression_probe(hidden_states, mask, labels, 
                                probe_across=probe_across, predict_hidden_states=model_states_probe_across[layer_idx])
                        except TypeError as e: # in the case of no probe_across
                            probe, preds = train_regression_probe(hidden_states, mask, labels, 
                                probe_across=probe_across, predict_hidden_states=None)
                    master_df_cols[f'{model_family}/{model_name}/{layer_idx}'] = preds
                    if save_probe:
                        probe_weights[f'{model_family}/{model_name}/{layer_idx}'] = torch.from_numpy(probe.coef_)
                else:
                    _, preds = train_regression_mlp(hidden_states, mask, labels, 
                        n_layers=n_layers, hidden_dim=hidden_dim)
                    master_df_cols[f'{model_family}/{model_name}/{layer_idx}'] = preds.numpy()

    master_df = pd.concat([master_df, pd.DataFrame(master_df_cols)], axis=1)
    print('Finished training all probes...')

    # save master_df
    master_df = master_df.copy()
    # recall: data_path_full = os.path.join('data', entity, f'{entity}{dataset}')
    results_dir = os.path.join(data_path_full, 'results')
    os.makedirs(results_dir, exist_ok=True)
    final_save_dir = f'{data_path_full}/results/{entity}{dataset}_{experiment_name}.csv'
    print(final_save_dir)
    master_df.to_csv(final_save_dir, index=False)
    print(f'Saved results to {final_save_dir}')
    
    # save probe weights
    if save_probe:
        # Combine probe weights into a single tensor
        n_layers = len(probe_weights)
        probe_dim = probe_weights[list(probe_weights.keys())[0]].shape[0]
        combined_weights = torch.zeros(n_layers, probe_dim)
        for i, weights in enumerate(probe_weights.values()):
            combined_weights[i] = weights
        
        # create probes directory if it doesn't exist
        probes_dir = f'{data_path_full}/probes'
        if not os.path.exists(probes_dir):
            os.makedirs(probes_dir)
        probe_weights_dir = f'{probes_dir}/{entity}{dataset}_{experiment_name}_probe_weights.pt'
        torch.save(combined_weights, probe_weights_dir)
        print(f'Saved probe weights to {probe_weights_dir}')
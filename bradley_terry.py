'''
THIS SCRIPT IS WHOLLY ATTRIBUTED TO:
@article{lamparth2025movingmedicalexamquestions,
  title={Moving Beyond Medical Exam Questions: A Clinician-Annotated Dataset of Real-World Tasks and Ambiguity in Mental Healthcare},
  author={Lamparth, Max and Grabb, Declan and Franks, Amy and Gershan, Scott and Kunstman, Kaitlyn N. and Lulla, Aaron and Drummond Roots, Monika and Sharma, Manu and Shrivastava, Aryan and Vasan, Nina and Waickman, Colleen},
  journal={arXiv preprint arXiv:2502.16051},
  year={2025},
  url={https://arxiv.org/abs/2502.16051}
}

Slight modifications to the original code were made to make it work with the new dataset structure.
'''
import argparse
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import random


def bt_prob(i, j, beta):
    """
    Probability that item i is preferred over item j in BT given beta
    """
    return np.exp(beta[i]) / (np.exp(beta[i]) + np.exp(beta[j]))


class BradleyTerry:
    """"""
    def __init__(self, data: list, k: int = 5):
        self.__data = data
        self.__k = k
        self.__betas = np.ones(k)

        self.__fitres = self.fit()

    @staticmethod
    def bradley_terry_neg_log_likelihood(betas, data):
        """
        beta: 1D array of length
        data: list of (winner, loser) pairs.
        
        Returns: scalar negative log-likelihood for the Bradley-Terry model.
        """

        nll = 0.0
        for winner, loser in data:
            numerator = np.exp(betas[winner])
            denominator = numerator + np.exp(betas[loser])
            nll += np.log(denominator) - np.log(numerator)
        
        return nll

    def prob(self, i, j):
        """
        Probability that item i is preferred over item j in BT given current beta values
        """
        return np.exp(self.__betas[i]) / (np.exp(self.__betas[i]) + np.exp(self.__betas[j]))
    
    def fit(self):
        """"""

        constraints = ({
            'type': 'eq',
            'fun': self.sum_to_zero_constraint
        },)

        # Optimize using scipy
        result = minimize(
            fun=self.bradley_terry_neg_log_likelihood,
            x0=self.__betas,
            args=(self.__data),
            constraints=constraints,
            # method='trust-constr',  # or 'SLSQP'
            method='SLSQP',
            options={'disp': True, 'maxiter': 500}
        )
        self.__betas = result.x

        # if not result.success:
        #     raise Warning("Optimization did not converge")
        
        return result
    
    def return_probs(self):
        """Return softmaxed betas to get absolut probability for each answer"""

        summed = np.array([np.exp(self.__betas[i]) for i in range(self.__k)]).sum()
        return np.array([np.exp(self.__betas[i])/summed for i in range(self.__k)])

    @staticmethod
    def sum_to_zero_constraint(beta):
        return np.sum(beta)
    
    @property
    def data(self):
        return self.__data
    
    @property
    def k(self):
        return self.__k
    
    @property
    def betas(self):
        return self.__betas
    
    @property
    def fitres(self):
        return self.__fitres


# def pairwise_wins(annotator_arr):
#     """Code helper to create pairwise wins"""

#     # Convert list of 1D arrays into 2D array (to be sure)
#     test_mat = np.array(annotator_arr)
#     k = test_mat.shape[-1]
#     data = []

#     for i in range(k):
#         # Calculate column-wise comparison matrix
#         comp_mat = np.array([test_mat[:, i] > test_mat[:, j] for j in range(k)]) 
#         row_idx, col_idx = np.nonzero(comp_mat.T)
#         # Create win pairs for current index i (i cannot win against itself)
#         pairs = [(i, v) for v in col_idx]  
#         assert len(pairs) == comp_mat.sum(), "Number of win-pairs does not match comparison matrix entries"
#         # Add pairs to data
#         data += pairs

#     return data

def pairwise_wins(df, winner_col):
    """
    Create pairwise wins from a dataframe

    Input:
        df: pandas dataframe with columns countryA, countryB, and google/gemma-2-9b-it_jailbrokenresponse_parsed (represents the winner)

    Output:
        data: list of (winner, loser) pairs
    """
    data = []

    nan_indices = df[winner_col].isna()
    df = df[~nan_indices]

    for _, row in df.iterrows():
        winner = row[winner_col]
        if winner.lower() == row['entityA'].lower():
            data.append((row['entityA'], row['entityB']))
        else:
            data.append((row['entityB'], row['entityA']))

    return data

def main():
    parser = argparse.ArgumentParser(description='Run Bradley-Terry model')
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
    args = parser.parse_args()

    entity = args.entity
    question_id = args.question_id
    model = args.model
    jailbreak_prompt = args.jailbreak_prompt

    if jailbreak_prompt is not None:
        if '.txt' in jailbreak_prompt:
            jailbreaking_type = f"{jailbreak_prompt.split('.')[0]}Jailbreak_"
        else:
            jailbreaking_type = ''

    dataset_path = f'data/{entity}/{entity}{question_id}/{entity}{question_id}_pairs.csv'
    df = pd.read_csv(dataset_path)
    main_df = pd.read_csv(f'data/{entity}/{entity}{question_id}/{entity}{question_id}.csv')
    players = main_df[entity.lower()].tolist()
    # players = get_players(df)
    pairwise_data = pairwise_wins(df, winner_col=f'{model}_{jailbreaking_type}response_parsed')

    k = len(players)
    

    for i, (winner, loser) in enumerate(pairwise_data):
        pairwise_data[i] = (players.index(winner), players.index(loser))

    

    bt7274 = BradleyTerry(data=pairwise_data, k=k)

    print("Optimization status:", bt7274.fitres.message)
    print("Estimated beta parameters (one per item):")
    for i, b in enumerate(bt7274.betas):
        print(f"  Item {players[i]}: {b:.4f}")

    # Convert betas to list and add as new column to original dataframe
    bt_scores = list(bt7274.betas)
    main_df[f'{model}_{jailbreaking_type}bradley_terry_scores'] = bt_scores
    
    # Save updated dataframe
    main_df.to_csv(f'data/{entity}/{entity}{question_id}/{entity}{question_id}.csv', index=False)

    # i, j = 0, 1
    # prob_i_beats_j = bt7274.prob(i, j)
    # print(f"\nProbability that item {i} is preferred over item {j}: {prob_i_beats_j:.4f}")

if __name__ == "__main__":
    main()

# INSERT TITLE
This is the official code for our paper: "TITLE" by Aryan Shrivastava and Ari Holtzman. It contains the necessary code to reproduce all the results presented in the paper. 
The link to the paper will be made available here once public on arXiv.

## Setup and Prerequisities
### 1. Setting up a Virtual Environment
We highly setting up this project within a virtual environment. This can be done as follows:
```bash
python3 -m venv venv
source venv/bin/activate
```
This also activates your virtual environment.

### 2. Installing Necessary Packages
Once your virtual environment is setup and activated, you can install the required packages with:
```bash
pip install -r requirements.txt
```

### 3. Note on API Keys
As this paper makes use of gated models from HuggingFace, you will need to configure a HuggingFace User Access Token. Here is a link to steps on how to do this: [link](https://huggingface.co/docs/hub/en/security-tokens).

## Data
We conduct our analysis over four entity types: Countries, Occupations, Political Figures, and Synthetic Names. Each entity type is associated with its own set of attributes. For example, country IQ or occupation divorce rate. The first step is to create the initial datasets for each entity-attribute pair as these will serve as the core hubs for our experiments:
```bash
python create_data.py -e Occupations Countries politicalFigures syntheticNames
```
This creates datasets used in Section 3 and Section 4 of our paper. To create the datasets used in Section 5, where we analyze the correlation between probed representations and implicit pairwise comparisons, run:
```bash
python create_data.py -e Occupations Countries politicalFigures syntheticNames -p
```


## Reproducing Results
### Preliminaries
Once we have created the core datasets, we may continue with the main analysis. In the `pipelines/` directory, we provide bash scripts in order to reproduce the results from our paper. Each bash script corresponds to a section in the paper, as noted by the title (e.g., `pipelines/sec3.sh`). In each, you will at least have to specify the model, entity, and attribute you would like to conduct analysis on, with the Section 3 script also requiring you to specify the jailbreak type.

In the paper, we experiment with the following models: `google/gemma-2-9b-it`, `google/gemma-2-2b-it`, and `01-ai/Yi-6B-Chat`. Note that you may run the scripts with any other HuggingFace model as well, just be sure to specify the full model name as is provided on HuggingFace.

We also experiment with two jailbreak types: `icl.txt` (the ICL prompt) and `machiavelli.txt` (the AIM prompt).

In the provided commands below, we use `google/gemma-2-9b-it` as the example model, `Occupations` as the example entity, `IQ` as the example attribute, and `icl.txt` as the example jailbreak type. 

### Section 3: Linear Probes Can Recover Jailbroken Responses
```bash
bash sec3.sh google/gemma-2-9b-it icl.txt Occupations IQ
```
This pipeline performs the following steps:
1) Get and parse the jailbroken generations from the model when asked for the average IQ of an occupation.
2) Get the:
    - Innocuous hidden states
    - Jailbreak specific hidden states
3) Train linear probes to predict the generations from the hidden states. 
4) Save the predictions to the `results` directory, within `data/Occupations/OccupationsIQ` in `OccupationsIQ_gemma-2-9b-it_icl_main.csv` and `OccupationsIQ_gemma-2-9b-it_icl_specific.csv`.

### Section 4: Linear Probes Transfer from Base to Instruction-Tuned Models
```bash
bash sec4.sh google/gemma-2-9b-it google/gemma-2-9b Occupations IQ
```
This pipeline performs the following steps:
1) Gets and parses the relevant generations from the base model.
2) Gets the innocuous hidden states from the base model.
3) Train linear probes on the base model hidden states and generations. 
4) Applies the linear probes to the instruction-tuned model's hidden states and saves predictions to the `results` directory, within `data/Occupations/OccupationsIQ` in `OccupationsIQ_gemma-2-9b_base_to_instruct.csv`.

Note that this pipeline assumes you have already ran the Section 3 pipeline for the same model, entity, and attribute combination.

### Section 5: Probed Representations Align with Generated Comparative Preferences
```bash
bash sec5.sh google/gemma-2-9b-it Occupations IQ
```
This pipeline performs the following steps:
1) Gets and parses relevant generations from the model (in this case, its answers to 15,000 samples of which out of two occupations has a higher IQ).
2) Runs a bradley-terry model to obtain the model's latent rankings.

### Plotting
The plotting code is provided in `plotting.ipynb`.

## Citing Our Work
If you found the paper or code useful, please consider citing us. The BibTeX will be available upon publication on arXiv.
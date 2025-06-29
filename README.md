# INSERT TITLE
This is the official code for our paper: "TITLE" (arxiv link) by Aryan Shrivastava and Ari Holtzman.

TODO: MAKE A BASH SCRIPT FOR THE ENTIRE PIPELINE, IT WILL BE EASY (at least for the non bradley-terry stuff)

This repository contains the necessary code to reproduce the results in the paper, from dataset generation, to probe training, to evaluation.

## Setup and Prerequisities
TODO

## Initial Methodology Details
We ground our analysis across four entity types: Countries, Occupations, Political Figures, and Synthetic Names.
Each entity type is associated with a set of attributes. For example, we ask an LM for a country's average IQ or an occupation's average substance abuse rate.

## Data Construction
You can directly create the relevant datasets by running the below script. This reads from the pre-defined entity list and associated questions for each attribute and saves a .csv into `data/{entity_type}/{entity_type}{attribute}`.
```bash
python create_data.py -e Occupations Countries politicalFigures syntheticNames
```

To create the data for the experiments run in Section 5 of the paper run:
```bash
python create_data.py -e Occupations Countries politicalFigures syntheticNames -p
```

For each entity type, this creates .csv with the individual entities and associated questions for each attribute. 
This allows us to continue onto generating the responses.

## Getting Model Responses
Run this script to get and parse model responses. These are used as the labels to the probes.
```bash
python get_generations.py -m $MODEL -e $ENTITY -q $ATTRIBUTE -s -j $icl/machiavelli.txt -b something
python parse_generations.py -m $MODEL -e $ENTITY -q $ATTRIBUTE
```

If you are running the Section 4 experiments, we do not jailbreak. Run:
```bash
python get_generations.py -m $MODEL -e $ENTITY -q $ATTRIBUTE -s -b something
python parse_generations.py -m $MODEL -e $ENTITY -q $ATTRIBUTE
```

If you are running the Section 5 experiments, we run:
```bash
python get_generations.py -m $MODEL -e $ENTITY -q $ATTRIBUTE -b something -s -p -c prompt -j icl.txt
python parse_generations.py -m $MODEL -e $ENTITY -q $ATTRIBUTE -j icl.txt -p
```
Technically, can do different jailbreak for getting generations like (-j machiavelli_comparisons.txt) but we don't do this in the paper.

We also have checkpointing behavior, where if you have compute constraints, you can spread generation over 5 steps of 3000 generations each:
```bash
python get_generations.py -m $MODEL -e $ENTITY -q $ATTRIBUTE -b something -s -p -c prompt -j icl.txt --checkpointing --chunk_number $one_of_0_through_4
```
where you may submit this over multiple separate jobs, ensuring that the next chunk only begins after the previous one completes.
After chunk number 4 completes, you may parse as above. 

## Getting Model Hidden States
Run this script to obtain the innocous last token hidden states:
```bash
python get_hidden_states.py -m $MODEL -e $ENTITY -b something
```

If you would like jailbreak specific hidden states:
```bash
python get_hidden_states.py -m $MODEL -e $ENTITY -q $DATASET -j icl/machiavelli.txt -b something
```

## Training and Evaluating Probes
Once we obtain the relevant hidden states and hidden states, we can train and evaluate our probes!
Running the following scripts will train probes for each layer in the model and then save their predictions alongside the true model generations in the `data/{entity_type}/{entity_type}{attribute}/results` directory.
If training and evaluating probes on the jailbroken generations and innocuous hidden states, run:
```bash
python probes.py -m $MODEL -e $ENTITY -q $DATASET -l $MODEL_$[jailbreak]Jailbreak_response_parsed --experiment_name {MODEL}_{jailbreak}_main
```

If training and evaluating probes on the jailbroken generations and specific hidden states, run:
```bash
python probes.py -m $MODEL -e $ENTITY -q $DATASET -l $MODEL_$[jailbreak]Jailbreak_response_parsed --experiment_name {MODEL}_{jailbreak}_specific --probe_specific $icl/machiavelli.txt
```

If training and evaluating probes for base to instruction-tuned transfer, run:
```bash
python probes.py -m $BASE_MODEL -e $ENTITY -q $DATASET -l $MODEL_response_parsed --probe_across $INSTRUCT_MODEL --experiment_name {BASE_MODEL}_base_to_instruct
```

THAT'S IT! Then, we can plot, but whatever, we have results, you can figure out plotting. 

## Running Bradley Terry
We have already provided the steps for obtaining the pairwise comparison generations above. Now, we need to run the bradley_terry so we can get the implicit ranking of the LLM:
```bash
python bradley_terry.py -m $MODEL -e $ENTITY -q $ATTRIBUTE -j icl.txt
```

Assuming we already have the probe results from before, all we need to do is get the correlations now, which we can do in the plotting code:

## Plotting Results!

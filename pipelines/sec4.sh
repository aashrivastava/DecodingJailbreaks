#!/bin/bash
set -e

##############################################
# USAGE
#   bash sec4.sh <BASE_MODEL_NAME> <INSTRUCT_MODEL_NAME> <ENTITY> <ATTRIBUTE> [-v|--verbose]
#
# EXAMPLE
#   bash sec4.sh google/gemma-2-9b-it google/gemma-2-9b-it-instruct Countries IQ -v
##############################################

# ============================================
# Default verbose flag
VERBOSE=0

# ============================================
# Parse positional arguments
if [ "$#" -lt 4 ]; then
    echo "Usage: bash sec4.sh <BASE_MODEL_NAME> <INSTRUCT_MODEL_NAME> <ENTITY> <ATTRIBUTE> [-v|--verbose]"
    exit 1
fi

BASE_MODEL=$1
INSTRUCT_MODEL=$2
ENTITY=$3
ATTRIBUTE=$4

# ============================================
# Check for -v or --verbose flag anywhere in args
for arg in "$@"; do
    if [ "$arg" == "-v" ] || [ "$arg" == "--verbose" ]; then
        VERBOSE=1
    fi
done

# ============================================
# Verbose echo function
v_echo() {
    if [ "$VERBOSE" -eq 1 ]; then
        echo "$@"
    fi
}

# ============================================
# Show config if verbose
v_echo "==========================================="
v_echo "Using BASE MODEL: $BASE_MODEL"
v_echo "Using INSTRUCT MODEL: $INSTRUCT_MODEL"
v_echo "Using ENTITY: $ENTITY"
v_echo "Using ATTRIBUTE: $ATTRIBUTE"
v_echo "Verbose mode ON"
v_echo "==========================================="

# ============================================
# 1. Get and parse generations
v_echo "-------------------------------------------"
v_echo "STEP 1: Get and parse generations for ENTITY: $ENTITY ATTRIBUTE: $ATTRIBUTE"
v_echo "-------------------------------------------"

v_echo ">>> Getting BASE model generations..."
python get_generations.py -m "$BASE_MODEL" -e "$ENTITY" -q "$ATTRIBUTE" -s

v_echo ">>> Parsing generations..."
python parse_generations.py -m "$BASE_MODEL" -e "$ENTITY" -q "$ATTRIBUTE"

v_echo "Done with base model generations for $ENTITY - $ATTRIBUTE"
v_echo

# ============================================
# 2. Get hidden states for innocuous case
v_echo "-------------------------------------------"
v_echo "STEP 2: Getting hidden states for ENTITY: $ENTITY"
v_echo "-------------------------------------------"

python get_hidden_states.py -m "$BASE_MODEL" -e "$ENTITY"
v_echo "Done with hidden states for $ENTITY"
v_echo

# ============================================
# 3. Train and evaluate probes
v_echo "-------------------------------------------"
v_echo "STEP 3: Training and evaluating probes for ENTITY: $ENTITY ATTRIBUTE: $ATTRIBUTE"
v_echo "-------------------------------------------"

python probes.py -m "$BASE_MODEL" -e "$ENTITY" -q "$ATTRIBUTE" -l "${BASE_MODEL}_response_parsed" --probe_across "$INSTRUCT_MODEL" --experiment_name "${BASE_MODEL}_base_to_instruct"

v_echo "Done training base model probe and obtaining predictions for instruction-tuned generations."
v_echo

v_echo "==========================================="
v_echo "Pipeline complete for Section 4 experiment."
v_echo "==========================================="

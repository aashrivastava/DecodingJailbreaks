#!/bin/bash
set -e

##############################################
# USAGE
#   bash sec3.sh <MODEL_NAME> <JAILBREAK_FILE> <ENTITY> <ATTRIBUTE> [-v|--verbose]
#
# EXAMPLE
#   bash sec3.sh google/gemma-2-9b-it icl.txt Countries IQ -v
##############################################

# ============================================
# Default verbose flag
VERBOSE=0

# ============================================
# Parse positional arguments
if [ "$#" -lt 4 ]; then
    echo "Usage: bash sec3.sh <MODEL_NAME> <JAILBREAK_FILE> <ENTITY> <ATTRIBUTE> [-v|--verbose]"
    exit 1
fi

MODEL=$1
JAILBREAK=$2
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
v_echo "Using MODEL: $MODEL"
v_echo "Using JAILBREAK FILE: $JAILBREAK"
v_echo "Using ENTITY: $ENTITY"
v_echo "Using ATTRIBUTE: $ATTRIBUTE"
v_echo "Verbose mode ON"
v_echo "==========================================="

# ============================================
# 1. Get and parse generations
v_echo "-------------------------------------------"
v_echo "STEP 1: Get and parse generations for ENTITY: $ENTITY ATTRIBUTE: $ATTRIBUTE"
v_echo "-------------------------------------------"

v_echo ">>> Getting model generations..."
python get_generations.py -m "$MODEL" -e "$ENTITY" -q "$ATTRIBUTE" -s -j "$JAILBREAK"

v_echo ">>> Parsing generations..."
python parse_generations.py -m "$MODEL" -e "$ENTITY" -q "$ATTRIBUTE" -j "$JAILBREAK"

v_echo "Done with generations for $ENTITY - $ATTRIBUTE"
v_echo

# ============================================
# 2. Get jailbreak-specific hidden states
v_echo "-------------------------------------------"
v_echo "STEP 2: Getting jailbreak-specific hidden states"
v_echo "-------------------------------------------"

python get_hidden_states.py -m "$MODEL" -e "$ENTITY" -q "$ATTRIBUTE" -j "$JAILBREAK"
v_echo "Done getting jailbreak-specific hidden states for $ENTITY - $ATTRIBUTE"
v_echo

# ============================================
# 3. Get innocuous hidden states for ENTITY
v_echo "-------------------------------------------"
v_echo "STEP 3: Getting innocuous hidden states for ENTITY: $ENTITY"
v_echo "-------------------------------------------"

python get_hidden_states.py -m "$MODEL" -e "$ENTITY"
v_echo "Done with innocuous hidden states for $ENTITY"
v_echo

# ============================================
# 4. Train and evaluate probes
v_echo "-------------------------------------------"
v_echo "STEP 4: Training probes for ENTITY: $ENTITY ATTRIBUTE: $ATTRIBUTE"
v_echo "-------------------------------------------"

python probes.py -m "$MODEL" -e "$ENTITY" -q "$ATTRIBUTE" -l "${MODEL}_${JAILBREAK}Jailbreak_response_parsed" --experiment_name "${MODEL}_${JAILBREAK}_main"
python probes.py -m "$MODEL" -e "$ENTITY" -q "$ATTRIBUTE" -l "${MODEL}_${JAILBREAK}Jailbreak_response_parsed" --experiment_name "${MODEL}_${JAILBREAK}_specific" --probe_specific "$JAILBREAK"

v_echo "Done training both innocuous and specific probes for $ENTITY - $ATTRIBUTE"
v_echo

v_echo "==========================================="
v_echo "Pipeline complete for Section 3 experiment."
v_echo "==========================================="

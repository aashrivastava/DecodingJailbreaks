#!/bin/bash
set -e

##############################################
# USAGE
#   bash sec5.sh <MODEL_NAME> <ENTITY> <ATTRIBUTE> [-v|--verbose]
#
# EXAMPLE
#   bash sec5.sh google/gemma-2-9b-it Countries IQ -v
##############################################

# ============================================
# Default verbose flag
VERBOSE=0

# ============================================
# Parse positional arguments
if [ "$#" -lt 4 ]; then
    echo "Usage: bash sec5.sh <MODEL_NAME> <ENTITY> <ATTRIBUTE> [-v|--verbose]"
    exit 1
fi

MODEL=$1
ENTITY=$2
ATTRIBUTE=$3

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
v_echo "Using ENTITY: $ENTITY"
v_echo "Using ATTRIBUTE: $ATTRIBUTE"
v_echo "Verbose mode ON"
v_echo "==========================================="

# ============================================
# 1. Get and parse generations
v_echo "-------------------------------------------"
v_echo "STEP 1: Get and parse pairwise comparisons for ENTITY: $ENTITY ATTRIBUTE: $ATTRIBUTE"
v_echo "-------------------------------------------"

v_echo ">>> Getting model generations..."
python get_generations.py -m "$MODEL" -e "$ENTITY" -q "$ATTRIBUTE" -s -p -c prompt -j icl.txt

v_echo ">>> Parsing generations..."
python parse_generations.py -m "$MODEL" -e "$ENTITY" -q "$ATTRIBUTE" -j icl.txt -p

v_echo "Done with model generations for $ENTITY - $ATTRIBUTE"
v_echo

# ============================================
# 2. Run bradley-terry model
v_echo "-------------------------------------------"
v_echo "STEP 2: Running bradley-terry model"
v_echo "-------------------------------------------"

python bradley_terry.py -m "$MODEL" -e "$ENTITY" -q "$ATTRIBUTE" -j icl.txt
v_echo "Done with bradley-terry model for $ENTITY"
v_echo
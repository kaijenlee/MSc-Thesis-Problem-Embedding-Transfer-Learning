#!/usr/bin/env bash
# Run ELA learning curve experiments for multiple N (rows-per-class) values.
#
# For each N, the (instance, run) pairs are factor pairs of N, so each pair
# uses the same total amount of training data per class but distributes it
# differently between instances and runs.

set -euo pipefail

INPUT_DIR="${1:-$HOME/t7ssd/dim2feat}"
SCRIPT="classify_ela_learning_curve.py"

run_experiment() {
    local n="$1"
    local instances="$2"
    local runs="$3"
    local output="ela_learning_N${n}.h5"

    echo "============================================================"
    echo "N=${n}  ->  ${output}"
    echo "  instances: ${instances}"
    echo "  runs:      ${runs}"
    echo "============================================================"

    python "$SCRIPT" "$INPUT_DIR" --pairs \
        --instance-counts $instances \
        --run-counts $runs \
        --output-name "$output"
}

# N = 6
run_experiment 6 \
    "6 3 2 1" \
    "1 2 3 6"

# N = 12
run_experiment 12 \
    "12 6 4 3 2 1" \
    "1 2 3 4 6 12"

# N = 24
run_experiment 24 \
    "24 12 8 6 4 3 2 1" \
    "1 2 3 4 6 8 12 24"

# N = 60
run_experiment 60 \
    "60 30 20 12 10 6 5 3 2" \
    "1 2 3 5 6 10 12 20 30"

echo
echo "All experiments complete."
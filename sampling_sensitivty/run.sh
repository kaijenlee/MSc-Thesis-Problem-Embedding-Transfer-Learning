#!/usr/bin/env bash
# =============================================================================
# Generate sample pickles for every (sampler x sample-size) config used by the
# study, at 30 runs. Each invocation covers all 24 functions x 100 instances
# x dims {2,5,10} in one COCO suite pass, matching generate_data.py.
#
# Output: data/samples/pickles/<sampler>_<size>_30.pkl
# These feed directly into your ELA extraction + compute_ela_cv.py.
#
# Run setup_r_ilhs.sh ONCE first (only needed for the ilhs sampler).
# =============================================================================
set -euo pipefail
 
RUNS=30
SIZES=(25 50 75 100)
 
# 6 samplers matching your ELA_FILES config keys. Add cma_single if you need it.
SAMPLERS=(ilhs lhs_random_cd)
 
for s in "${SAMPLERS[@]}"; do
    for n in "${SIZES[@]}"; do
        echo "=================================================================="
        echo ">> sampler=${s}  size=${n}  runs=${RUNS}"
        echo "=================================================================="
        python generate_data.py -s "${s}" -f ela -n "${n}" -r "${RUNS}"
    done
done
 
echo ">> All configs generated in data/samples/pickles/"
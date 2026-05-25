#!/bin/bash
# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================================================${NC}"
echo -e "${CYAN}                 PanNuke Pipeline Environment Verification              ${NC}"
echo -e "${CYAN}========================================================================${NC}"

# Helper function to run a step
run_step() {
    local stage_num=$1
    local stage_name=$2
    local cmd=$3
    local allow_fail=$4
    
    echo -e "\n${YELLOW}=== STAGE ${stage_num}: ${stage_name} ===${NC}"
    echo -e "Running: ${cmd}"
    
    eval "$cmd"
    local status=$?
    
    if [ $status -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS] STAGE ${stage_num} - ${stage_name} passed successfully!${NC}"
        return 0
    else
        if [ "$allow_fail" = "true" ]; then
            echo -e "${YELLOW}[WARN] STAGE ${stage_num} - ${stage_name} failed with exit code ${status}.${NC}"
            echo -e "${YELLOW}No working CUDA GPU/driver detected or mismatched driver version.${NC}"
            echo -e "${YELLOW}The pipeline will run in CPU-fallback mode as per graceful failover rules.${NC}"
            return 0
        else
            echo -e "${RED}[FAILED] STAGE ${stage_num} - ${stage_name} failed with exit code ${status}.${NC}"
            exit 1
        fi
    fi
}

# Resolve python executable: check if we are in conda env, or locate the env's python
PYTHON_EXE="python"
if [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_EXE="$CONDA_PREFIX/bin/python"
elif [ -x "/home/mananbyte/miniconda3/envs/HandCraft-Path/bin/python" ]; then
    PYTHON_EXE="/home/mananbyte/miniconda3/envs/HandCraft-Path/bin/python"
fi

echo -e "Using Python executable: ${PYTHON_EXE}"

run_step "1" "GPU & CUDA Diagnostics" "$PYTHON_EXE scripts/diagnose_gpu_init.py" "true"
run_step "2" "Unit Tests (safe_load_npy & Feature Selection)" "$PYTHON_EXE tests/test_feature_selection.py" "false"
run_step "3" "Stain Normalizer Sanity Check" "$PYTHON_EXE scripts/test_stain_normalizer.py" "false"
run_step "4" "Feature Extraction Pipeline Smoke Test" "$PYTHON_EXE scripts/test_feature_pipeline.py" "false"

echo -e "\n${GREEN}========================================================================${NC}"
echo -e "${GREEN}             [ALL PASSED] Environment Verification Successful!          ${NC}"
echo -e "${GREEN}========================================================================${NC}"

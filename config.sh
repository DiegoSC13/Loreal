#!/bin/bash

# --- Environment Configuration ---

# Path to the python binary in your conda environment
PYTHON_BIN="/home/diegosilvera/anaconda3/envs/loreal_diego_cuda/bin/python"

# Base directory for the project
WORKDIR="/home/diegosilvera/Escritorio/Loreal"

OTHER_STUFF_DIR="/home/diegosilvera/Escritorio/2026"

# Directory where image sequences are stored
SEQUENCE_DIR_BASE="${OTHER_STUFF_DIR}/sequences_almost_Poisson"

# Path to the external FastDVDnet library
EXTERNAL_CODES_DIR="${OTHER_STUFF_DIR}"

# Default pretrained checkpoint
DEFAULT_CKPT_PATH="${OTHER_STUFF_DIR}/FastDVDnet-pure_poisson-a=1-normalization_by_255.pth"

# Path to FMDD dataset (Update this for your current machine)
FMDD_DIR="/home/diegosilvera/Escritorio/data/FMDD"

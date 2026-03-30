#!/bin/bash

# Activar entorno conda
source /mnt/bdisk/miniconda3/etc/profile.d/conda.sh
conda activate loreal_diego_cuda

# Parámetros
LR_VALUES=("1e-5")
TAU1_VALUES=("0.001" "0.0005" "0.0001")
EPOCHS=150

SEQUENCE_DIR="/mnt/bdisk/dewil/loreal_POC2/sequences_almost_Poisson"
CKPT="/mnt/bdisk/dewil/loreal_POC2/sequences_for_self-supervised_tests/FastDVDnet_codes/universal_network_for_Poisson_noise.pth"
OUTPUT_BASE="./results"

INPUT_SEQ="../../sequences_almost_Poisson/HF4_Bruite_1024pix_Ex780nm_10pc_LineAccu12.tif_dir/image_%03d.tif"
PREPROC="../../sequences_almost_Poisson/HF4_Bruite_1024pix_Ex780nm_10pc_LineAccu12.tif_dir/pre-processing.txt"
NETWORK="../../sequences_for_self-supervised_tests/FastDVDnet_codes/universal_network_for_Poisson_noise.pth"

for lr in "${LR_VALUES[@]}"; do
  for tau1 in "${TAU1_VALUES[@]}"; do

    EXP_NAME="lr_${lr}_tau1_${tau1}"
    OUTDIR="${OUTPUT_BASE}/${EXP_NAME}"
    NETWORK="${OUTDIR}/ckpts/epoch_${EPOCHS}.pth"
    mkdir -p "${OUTDIR}"

    LOGFILE="${OUTDIR}/train.log"

    echo "Running experiment ${EXP_NAME}"

    python train.py \
      --sequence_directory ${SEQUENCE_DIR} \
      --output_path ${OUTDIR} \
      --ckpt ${CKPT} \
      --loss pure \
      --gamma 1.0 \
      --data_scale 9000 \
      --tau1 ${tau1} \
      --batch_size 32 \
      --epochs ${EPOCHS} \
      --lr ${lr} \
      --patch_size 256 256 \
      &> ${LOGFILE}

    echo "Testing ${EXP_NAME}"

    python test3.py \
      --input ${INPUT_SEQ} \
      --output ${OUTDIR}/test_output_%03d.tif \
      --pre_processing_data ${PREPROC} \
      --first 0 \
      --last 29 \
      --network ${NETWORK}

  done
done

echo "All experiments finished."
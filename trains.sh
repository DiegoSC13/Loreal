#!/bin/bash

# Activar entorno conda
#source /mnt/bdisk/miniconda3/etc/profile.d/conda.sh
# conda activate loreal_diego_cuda

PYTHON=/mnt/bdisk/miniconda_envs/loreal_diego_cuda/bin/python

# Parámetros
# LR_VALUES=("1e-6")
# TAU1_VALUES=("0.005" "0.0005" "0.0001" "0.00005" "0.00001")
# EPOCHS=100

LR_VALUES=($1)
TAU1_VALUES=($2)
EPOCHS=$3

echo "LR: ${LR_VALUES[@]}"
echo "TAU1: ${TAU1_VALUES[@]}"
echo "EPOCHS: $EPOCHS"

SEQUENCE_DIR="/mnt/bdisk/dewil/loreal_POC2/sequences_almost_Poisson"
CKPT="/mnt/bdisk/dewil/loreal_POC2/sequences_for_self-supervised_tests/FastDVDnet_codes/universal_network_for_Poisson_noise.pth"
TIMESTAMP=$(date +"%y-%m-%d_%H-%M-%S")
OUTPUT_BASE=./results/train_$TIMESTAMP
#OUTPUT_BASE="./results"
DATA_SCALE=9000

INPUT_SEQ="../../sequences_almost_Poisson/HF4_Bruite_1024pix_Ex780nm_10pc_LineAccu12.tif_dir/image_%03d.tif"
PREPROC="../../sequences_almost_Poisson/HF4_Bruite_1024pix_Ex780nm_10pc_LineAccu12.tif_dir/pre-processing.txt"
#NETWORK="../../sequences_for_self-supervised_tests/FastDVDnet_codes/universal_network_for_Poisson_noise.pth"

for lr in "${LR_VALUES[@]}"; do
  for tau1 in "${TAU1_VALUES[@]}"; do

    EXP_NAME="lr_${lr}_tau1_${tau1}"
    OUTDIR="${OUTPUT_BASE}/${EXP_NAME}"
    NETWORK="${OUTDIR}/ckpts/epoch_${EPOCHS}.pth"
    mkdir -p "${OUTDIR}"

    LOGFILE="${OUTDIR}/train.log"

    echo "Command: $PYTHON train.py --sequence_directory ${SEQUENCE_DIR} --output_path ${OUTDIR} --ckpt ${CKPT} --loss pure --gamma 1.0 --data_scale ${DATA_SCALE} --tau1 ${tau1} --batch_size 32 --epochs ${EPOCHS} --lr ${lr} --patch_size 256 256 --transform d4" > ${LOGFILE}

    $PYTHON train.py \
      --sequence_directory ${SEQUENCE_DIR} \
      --output_path ${OUTDIR} \
      --ckpt ${CKPT} \
      --loss pure \
      --gamma 1.0 \
      --data_scale ${DATA_SCALE} \
      --tau1 ${tau1} \
      --batch_size 32 \
      --epochs ${EPOCHS} \
      --lr ${lr} \
      --patch_size 256 256 \
      --transform d4 \
      &>> ${LOGFILE}

    echo "Testing ${EXP_NAME}"
    TEST_LOGFILE="${OUTDIR}/test.log"
    echo "Command: $PYTHON test4.py --input ${INPUT_SEQ} --output ${OUTDIR}/last_epoch/test_output_%03d.tif --pre_processing_data ${PREPROC} --first 0 --last 29 --network ${NETWORK} --data_scale ${DATA_SCALE}" > ${TEST_LOGFILE}

    $PYTHON test4.py \
      --input ${INPUT_SEQ} \
      --output ${OUTDIR}/last_epoch/test_output_%03d.tif \
      --pre_processing_data ${PREPROC} \
      --first 0 \
      --last 29 \
      --network ${NETWORK} \
      --data_scale ${DATA_SCALE} \
      &>> ${TEST_LOGFILE}

  done
done

echo "All experiments finished."
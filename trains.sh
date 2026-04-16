#!/bin/bash

# Activar entorno conda
#source /mnt/bdisk/miniconda3/etc/profile.d/conda.sh
# conda activate loreal_diego_cuda

# --- CARGAR CONFIGURACIÓN LOCAL ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/env_paths.sh" ]; then
    source "${SCRIPT_DIR}/env_paths.sh"
fi

# Variables de entorno con fallbacks
PYTHON=${PYTHON_BIN:-"/mnt/bdisk/miniconda_envs/loreal_diego_cuda/bin/python"}
WORKDIR=${WORKDIR:-"/mnt/bdisk/dewil/loreal_POC2"}
SEQUENCE_DIR=${SEQUENCE_DIR_BASE:-"${WORKDIR}/sequences_almost_Poisson"}
EXTERNAL_CODES_DIR=${EXTERNAL_CODES_DIR:-"${WORKDIR}/sequences_for_self-supervised_tests/FastDVDnet_codes"}
DEFAULT_CKPT=${DEFAULT_CKPT_PATH:-"${EXTERNAL_CODES_DIR}/FastDVDnet-pure_poisson-a=1-normalization_by_255.pth"}

# Exportar PYTHONPATH como respaldo
export PYTHONPATH="${EXTERNAL_CODES_DIR}:${PYTHONPATH}"

# Parámetros (desde línea de comandos)
# LR_VALUES=("1e-6")
# TAU1_VALUES=("0.005" "0.0005" "0.0001" "0.00005" "0.00001")
# EPOCHS=100

LR_VALUES=($1)
HYPER_VALUES=($2) #Para PURE es tau1, para R2R es alpha
EPOCHS=$3
LOSS=${4:-pure} #El -algo es el valor por defecto. Si no se pone nada, se usa pure
PATIENCE=${5:-0}             # Default: early stopping disabled
TEST_SAMPLES=${6:-25}         # Default samples for R2R ensemble
GEOMETRIC_TTA=${7:-false}     # Default: no rotations/flips

echo "LR: ${LR_VALUES[@]}"
echo "HYPER: ${HYPER_VALUES[@]}"
echo "EPOCHS: $EPOCHS"
echo "LOSS: $LOSS"
echo "PATIENCE: $PATIENCE"
echo "TEST_SAMPLES: $TEST_SAMPLES"
echo "GEOMETRIC_TTA: $GEOMETRIC_TTA"

DIEGO_DIR="${WORKDIR}/diego/loreal_training_code"

#CKPT="${WORKDIR}/sequences_for_self-supervised_tests/FastDVDnet_codes/universal_network_for_Poisson_noise.pth"
CKPT="${DEFAULT_CKPT}"
TIMESTAMP=$(date +"%y-%m-%d_%H-%M-%S")
OUTPUT_BASE=./results/train_${TIMESTAMP}_${LOSS} #Esta me gusta así porque siempre crea results en el mismo sitio
DATA_SCALE=255 #9000
BATCH_SIZE=16
PATCH_SIZE="256 256"

GAMMA=1.0

INPUT_SEQ="${SEQUENCE_DIR}/HF4_Bruite_1024pix_Ex780nm_10pc_LineAccu12.tif_dir/image_%03d.tif"
PREPROC="${SEQUENCE_DIR}/HF4_Bruite_1024pix_Ex780nm_10pc_LineAccu12.tif_dir/pre-processing.txt"
#NETWORK="../../sequences_for_self-supervised_tests/FastDVDnet_codes/universal_network_for_Poisson_noise.pth"

for lr in "${LR_VALUES[@]}"; do
  for h in "${HYPER_VALUES[@]}"; do

    if [ "$LOSS" == "r2r_p" ]; then
      EXP_NAME="lr_${lr}_alpha_${h}"
      HYPER_ARG="--alpha ${h}"
    else
      EXP_NAME="lr_${lr}_tau1_${h}"
      HYPER_ARG="--tau1 ${h}"
    fi

    OUTDIR="${OUTPUT_BASE}/${EXP_NAME}"
    # NETWORK se define después de train.py para soportar early stopping
    mkdir -p "${OUTDIR}"

    LOGFILE="${OUTDIR}/train.log"

    echo "Command: $PYTHON train.py --sequence_directory ${SEQUENCE_DIR} --output_path ${OUTDIR} --ckpt ${CKPT} --loss ${LOSS} --gamma ${GAMMA} --data_scale ${DATA_SCALE} ${HYPER_ARG} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --lr ${lr} --patch_size ${PATCH_SIZE} --transform d4" > ${LOGFILE}

    $PYTHON train.py \
      --sequence_directory ${SEQUENCE_DIR} \
      --output_path ${OUTDIR} \
      --ckpt ${CKPT} \
      --loss ${LOSS} \
      --gamma ${GAMMA} \
      --data_scale ${DATA_SCALE} \
      ${HYPER_ARG} \
      --batch_size ${BATCH_SIZE} \
      --epochs ${EPOCHS} \
      --lr ${lr} \
      --patch_size ${PATCH_SIZE} \
      --transform d4 \
      --patience ${PATIENCE} \
      &>> ${LOGFILE}

    # Uso del mejor modelo (best_model.pth) para test
    NETWORK="${OUTDIR}/ckpts/best_model.pth"

    # Preparar argumentos de test (Self-ensemble solo para R2R)
    if [[ "$LOSS" == "r2r_p" || "$LOSS" == "r2r_g" ]]; then
      TEST_ARGS="--n_samples ${TEST_SAMPLES} --loss ${LOSS} --alpha ${h} --gamma ${GAMMA}"
    else
      TEST_ARGS="--n_samples 1"
    fi

    # Añadir geometric TTA si está activado
    if [ "$GEOMETRIC_TTA" = true ]; then
      TEST_ARGS="${TEST_ARGS} --geometric_ensemble"
    fi

    echo "Testing ${EXP_NAME} with BEST network: $NETWORK (Samples: ${TEST_SAMPLES})"
    TEST_LOGFILE="${OUTDIR}/test_best.log"
    echo "Command: $PYTHON test4.py --input ${INPUT_SEQ} --output ${OUTDIR}/best_model/test_output_%03d.tif --pre_processing_data ${PREPROC} --first 0 --last 29 --network ${NETWORK} --data_scale ${DATA_SCALE} ${TEST_ARGS}" > ${TEST_LOGFILE}

    $PYTHON test4.py \
      --input ${INPUT_SEQ} \
      --output ${OUTDIR}/best_model/test_output_%03d.tif \
      --pre_processing_data ${PREPROC} \
      --first 0 \
      --last 29 \
      --network ${NETWORK} \
      --data_scale ${DATA_SCALE} \
      ${TEST_ARGS} \
      &>> ${TEST_LOGFILE}

  done
done

echo "All experiments finished."
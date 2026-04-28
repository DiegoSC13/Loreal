#!/bin/bash

# Script para testear múltiples experimentos de un directorio base usando el mejor modelo (best_model.pth)
# Uso: ./test_experiments.sh <directorio_base> [input_seq] [preproc] [data_scale]

BASE_DIR=$1

if [ -z "$BASE_DIR" ]; then
    echo "Uso: $0 <directorio_base> [input_seq] [preproc] [data_scale] [first] [last]"
    echo "Ejemplo: $0 results/train_24-04-07_12-00-00_pure"
    exit 1
fi

# Guardar un registro de la ejecución para que no se te olvide cómo lo corriste
printf "bash " > "${BASE_DIR}/last_test_command.sh"
printf "%q " "$0" "$@" >> "${BASE_DIR}/last_test_command.sh"
echo "" >> "${BASE_DIR}/last_test_command.sh"
chmod +x "${BASE_DIR}/last_test_command.sh"

# --- CARGAR CONFIGURACIÓN LOCAL ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/config.sh" ]; then
    source "${SCRIPT_DIR}/config.sh"
fi

# Configuración (prioriza config.sh o argumentos)
PYTHON=${PYTHON_BIN:-"/mnt/bdisk/miniconda_envs/loreal_diego_cuda/bin/python"}

if [[ "$BASE_DIR" == *"fmdd"* ]]; then
    DEFAULT_INPUT="${FMDD_DIR}/WideField_BPAE_G/gt/12/avg50.png"
    DEFAULT_PREPROC=""
    DEFAULT_FIRST=0
    DEFAULT_LAST=0
    DEFAULT_EXTRA="--synthetic_test --gamma 1.0 --n_samples 25 --alpha 0.15 --loss r2r_p --save_noisy"
    DEFAULT_DATA_SCALE=255.0
else
    DEFAULT_INPUT="${SEQUENCE_DIR_BASE}/HF1_Bruite_1024pix_Ex780nm_10pc_LineAccu12.tif_dir/image_%03d.tif"
    DEFAULT_PREPROC="${SEQUENCE_DIR_BASE}/HF1_Bruite_1024pix_Ex780nm_10pc_LineAccu12.tif_dir/pre-processing.txt"
    DEFAULT_FIRST=0
    DEFAULT_LAST=29
    DEFAULT_EXTRA="--save_noisy"
    DEFAULT_DATA_SCALE=255.0
fi

INPUT_SEQ=${2:-"${DEFAULT_INPUT}"}
PREPROC=${3:-"${DEFAULT_PREPROC}"}
DATA_SCALE=${4:-${DEFAULT_DATA_SCALE}}
FIRST_FRAME=${5:-${DEFAULT_FIRST}}
LAST_FRAME=${6:-${DEFAULT_LAST}}
EXTRA_ARGS=${7:-"${DEFAULT_EXTRA}"}

echo "--- Iniciando Test por Lote ---"
echo "Directorio Base: $BASE_DIR"
echo "Secuencia: $INPUT_SEQ"
echo "-------------------------------"

# Verificar si PREPROC existe, de lo contrario no pasarlo
if [ -f "$PREPROC" ]; then
    PREPROC_ARG="--pre_processing_data ${PREPROC}"
else
    echo "[!] Aviso: pre_processing.txt no encontrado. Se usará a=1, b=0 por defecto."
    PREPROC_ARG=""
fi

# Iterar sobre cada subdirectorio de experimento
for exp in "$BASE_DIR"/*/; do
    # Eliminar slash final para basename
    exp=${exp%/}
    
    if [ -d "$exp" ]; then
        EXP_NAME=$(basename "$exp")
        
        # Omitir directorios que no sean experimentos (ej: aquellos que no tengan carpeta ckpts)
        if [ ! -d "$exp/ckpts" ]; then
            continue
        fi

        NETWORK="$exp/ckpts/best_model.pth"
        
        if [ -f "$NETWORK" ]; then
            echo "[*] Procesando experimento: $EXP_NAME"
            OUTDIR="$exp/best_model"
            mkdir -p "$OUTDIR"
            
            TEST_LOGFILE="$exp/test_experiments_output.log"
            
            # Registrar comando
            echo "Command: $PYTHON test4.py --input ${INPUT_SEQ} --output ${OUTDIR}/test_output_%03d.tif ${PREPROC_ARG} --first ${FIRST_FRAME} --last ${LAST_FRAME} --network ${NETWORK} --data_scale ${DATA_SCALE} ${EXTRA_ARGS}" > "${TEST_LOGFILE}"
            
            # Ejecutar test
            $PYTHON test4.py \
              --input "${INPUT_SEQ}" \
              --output "${OUTDIR}/test_output_%03d.tif" \
              ${PREPROC_ARG} \
              --first "${FIRST_FRAME}" \
              --last "${LAST_FRAME}" \
              --network "${NETWORK}" \
              --data_scale "${DATA_SCALE}" \
              ${EXTRA_ARGS} \
              &>> "${TEST_LOGFILE}"
            
            echo "    -> Completado. Resultados en: $OUTDIR"
        else
            echo "[!] Saltando $EXP_NAME: No se encontró 'best_model.pth' en ckpts/"
        fi
    fi
done

echo "--- Test por lote finalizado ---"

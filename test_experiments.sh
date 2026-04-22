#!/bin/bash

# Script para testear múltiples experimentos de un directorio base usando el mejor modelo (best_model.pth)
# Uso: ./test_experiments.sh <directorio_base> [input_seq] [preproc] [data_scale]

BASE_DIR=$1

if [ -z "$BASE_DIR" ]; then
    echo "Uso: $0 <directorio_base> [input_seq] [preproc] [data_scale] [first] [last]"
    echo "Ejemplo: $0 results/train_24-04-07_12-00-00_pure"
    exit 1
fi

# --- CARGAR CONFIGURACIÓN LOCAL ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/env_paths.sh" ]; then
    source "${SCRIPT_DIR}/env_paths.sh"
fi

# Configuración (prioriza env_paths.sh o argumentos)
PYTHON=${PYTHON_BIN:-"/mnt/bdisk/miniconda_envs/loreal_diego_cuda/bin/python"}
INPUT_SEQ=${2:-"${FMDD_DIR}/HF1_Bruite_1024pix_Ex780nm_10pc_LineAccu12.tif_dir/image_%03d.tif"}
PREPROC=${3:-"${FMDD_DIR}/HF1_Bruite_1024pix_Ex780nm_10pc_LineAccu12.tif_dir/pre-processing.txt"}
DATA_SCALE=${4:-255}
FIRST_FRAME=${5:-0}
LAST_FRAME=${6:-29}
EXTRA_ARGS=${7:-""}

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
            
            TEST_LOGFILE="$exp/test_best_batch.log"
            
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

#!/usr/bin/env bash

# for macOS
if command -v gdate &> /dev/null
then
    DATE_CMD=gdate
else
    DATE_CMD=date
fi

START_TIME="$(${DATE_CMD} +%s)"
START_TIME_STR="$(${DATE_CMD} -d @${START_TIME} +%Y%m%d-%H%M%S)"
BASENAME="$(basename "${BASH_SOURCE}" .sh)"
HOSTNAME_S="$(hostname -s)"
LOGFILE="${BASENAME}-${START_TIME_STR}-${HOSTNAME_S}.log"
echo "XXX LOGFILE ${LOGFILE}" | tee -a ${LOGFILE}
echo "XXX DATETIME ${START_TIME_STR}" | tee -a ${LOGFILE}

#ORIG="ibm-granite/granite-3.3-8b-instruct"
#TRAINED="experiments/sft_granite_example_granite-3.3-8b-instruct_teigaku-genzei-ibm-v6_20251005_080203/hf_format/samples_50253"
#TRAINED="experiments/osft_granite_example_granite-3.3-8b-instruct_teigaku-genzei-ibm-v6_20251005_080626/hf_format/samples_50253.0"
ORIG="ibm-granite/granite-4.0-h-small"
TRAINED="experiments/sft_granite4_example_teigaku-genzei-ibm-v6_20251104_022912/hf_format/samples_50253"

MODEL_PATH="${ORIG}"
TRAINED_MODEL_PATH="${TRAINED}"
TRAINED_MODEL_WEIGHT=0.5
if [[ "${TRAINED_MODEL_WEIGHT}" != "0.5" ]]; then
    INTERPOLATED="${TRAINED}_interp_${TRAINED_MODEL_WEIGHT}"
else
    INTERPOLATED="${TRAINED}_interp"
fi
OUTPUT_MODEL_PATH="${INTERPOLATED}"
TORCH_DTYPE="bfloat16"

ENV=""
#ENV="TOKENIZERS_PARALLELISM=false ${ENV}"
#ENV="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${ENV}"
cmd="${ENV}python ${BASENAME}.py --model-path ${MODEL_PATH} --trained-model-path ${TRAINED_MODEL_PATH} --trained-model-weight ${TRAINED_MODEL_WEIGHT} --output-model-path ${OUTPUT_MODEL_PATH} --torch-dtype ${TORCH_DTYPE}"
echo "$cmd" | tee -a ${LOGFILE}
eval "$cmd" 2>&1 | tee -a ${LOGFILE}

END_TIME="$(${DATE_CMD} +%s)"
END_TIME_STR="$(${DATE_CMD} -d @${END_TIME} +%Y%m%d-%H%M%S)"
echo "XXX DATETIME ${END_TIME_STR}" | tee -a ${LOGFILE}
echo "XXX ELAPSED_SECS $((END_TIME - START_TIME))" | tee -a ${LOGFILE}

#!/bin/bash
# Emu3-Chat Model - General Evaluation Script
#
# This script evaluates Emu3-Chat on any task
# Supports understanding tasks (image -> text)
#
# Usage:
#   bash emu3.sh [GPU_IDS] [TASK] [OUTPUT_PATH] [MODEL_PATH] [MASTER_PORT] [RESERVED] [LIMIT]
#
# Examples:
#   # ChartQA
#   bash /home/aiscuser/lmms-eval/g2u/emu3.sh "0" "chartqa100" "./logs/chartqa"
#
#   # MMBench
#   bash /home/aiscuser/lmms-eval/g2u/emu3.sh "0" "mmbench" "./logs/mmbench"
#
#   # MathVista
#   bash /home/aiscuser/lmms-eval/g2u/emu3.sh "0" "mathvista" "./logs/mathvista"
#
#   # Multiple tasks (comma-separated, no spaces)
#   bash /home/aiscuser/lmms-eval/g2u/emu3.sh "0" "chartqa100,mmbench" "./logs/multi"
#
#   # With custom model path and port
#   bash /home/aiscuser/lmms-eval/g2u/emu3.sh "0" "chartqa100" "./logs/chartqa" "BAAI/Emu3-Chat-hf" "29603"
#
#   # With limit (test only 100 samples)
#   bash /home/aiscuser/lmms-eval/g2u/emu3.sh "0" "geometry3k" "./logs/emu3/geometry3k" "BAAI/Emu3-Chat-hf" "29708" "" "100"

# ============ Configuration ============
GPU_IDS=${1:-"0"}
TASK=${2:-"mmbench"}
OUTPUT_PATH=${3:-"./logs/emu3_${TASK}"}
MODEL_PATH=${4:-"BAAI/Emu3-Chat-hf"}
MASTER_PORT=${5:-"29602"}
RESERVED=${6:-""}
LIMIT=${7:-""}
BATCH_SIZE=1

# Model arguments for Emu3-Chat (understanding mode)
# Note: using single device loading to avoid tensor parallel issues
# use_flash_attention_2=true to save memory on long sequences
MODEL_ARGS="pretrained=${MODEL_PATH},mode=understanding,max_new_tokens=512,do_sample=false,temperature=0.0,device=cuda,device_map=single,use_flash_attention_2=true"

# ============ Environment Setup ============
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export MASTER_PORT=${MASTER_PORT}
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export GLOO_USE_IPV6=0

# Fix libstdc++ version issue - use conda's libraries
export LD_LIBRARY_PATH=/opt/conda/envs/ptca/lib:$LD_LIBRARY_PATH

# ============ Print Configuration ============
echo "======================================"
echo "Emu3-Chat - General Evaluation"
echo "======================================"
echo "GPU(s):        ${GPU_IDS}"
echo "Model Path:    ${MODEL_PATH}"
echo "Task(s):       ${TASK}"
echo "Output Path:   ${OUTPUT_PATH}"
echo "Batch Size:    ${BATCH_SIZE}"
echo "Model Args:    ${MODEL_ARGS}"
echo "Master Port:   ${MASTER_PORT}"
if [ -n "${LIMIT}" ]; then
  echo "Limit:         ${LIMIT}"
fi
echo "======================================"
echo ""

# ============ Run Evaluation ============
# Build limit argument if provided
LIMIT_ARG=""
if [ -n "${LIMIT}" ]; then
  LIMIT_ARG="--limit ${LIMIT}"
fi

accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  -m lmms_eval \
  --model emu3 \
  --model_args ${MODEL_ARGS} \
  --tasks ${TASK} \
  --batch_size ${BATCH_SIZE} \
  --output_path ${OUTPUT_PATH} \
  --log_samples \
  --verbosity INFO \
  ${LIMIT_ARG}

echo ""
echo "======================================"
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_PATH}"
echo "======================================"

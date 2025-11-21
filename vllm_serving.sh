#!/usr/bin/env bash
set -e

# 1) 이 스크립트를 project_jeon 폴더에서 실행한다고 가정
cd "$(dirname "$0")"

# 2) WSL 안에서 만든 vLLM용 가상환경 활성화
source .venv_vllm/bin/activate

# 3) GPU 설정 (GPU 하나면 0)
export CUDA_VISIBLE_DEVICES="0"
export VLLM_SKIP_WARMUP="true"

# 4) 사용할 모델 (HuggingFace 공개 모델)
#    - 처음 실행할 때 자동으로 다운로드됨
MODEL="Qwen/Qwen2-0.5B-Instruct"
SERVED_NAME="qwen2-0_5b-instruct"

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size 1 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.7 \
  --dtype half \
  --host 0.0.0.0 \
  --port 1234

# openai_api.py

from typing import List, Dict, Any
from openai import OpenAI

# vLLM OpenAI 호환 서버 설정
VLLM_BASE_URL = "http://localhost:1234/v1"
VLLM_MODEL_NAME = "qwen2-1_5b-instruct"  # vllm_serving.sh의 --served-model-name과 동일하게 유지

# vLLM 클라이언트 (실제 OpenAI가 아니라 로컬 서버)
client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key="sk-no-auth-needed",  # 아무 문자열이나 가능 (vLLM에서는 실제로 사용 안 함)
)


def chat_completion(
    messages: List[Dict[str, Any]],
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 1024,
) -> str:
    """
    로컬 vLLM(OpenAI 호환 서버)에 chat.completions 요청을 보내고,
    assistant의 reply 텍스트만 반환하는 헬퍼 함수.
    """
    response = client.chat.completions.create(
        model=VLLM_MODEL_NAME,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

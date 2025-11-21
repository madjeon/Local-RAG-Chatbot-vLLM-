# app.py

import uuid
from typing import List, Tuple

import streamlit as st
from PyPDF2 import PdfReader

from rag import SimpleRAG
from openai_api import chat_completion

# -----------------------------
# 1) 기본 설정 & 세션 초기화
# -----------------------------
st.set_page_config(page_title="My Local RAG Chatbot", layout="wide")

# 간단 패스워드 잠금 (원하면 제거 가능)
PASSWORD = "demo"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    password = st.text_input("Enter password to access:", type="password")
    if password != PASSWORD:
        st.stop()
    else:
        st.session_state.authenticated = True

# 세션 상태 기본값
if "model_temperature" not in st.session_state:
    st.session_state.model_temperature = 0.7
if "model_top_p" not in st.session_state:
    st.session_state.model_top_p = 1.0
if "model_max_tokens" not in st.session_state:
    st.session_state.model_max_tokens = 1024
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        "You are a helpful, reliable, deeply knowledgeable AI assistant. "
        "Use the given context from documents when it is relevant. "
        "If the context is not relevant, ignore it."
    )
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_session" not in st.session_state:
    sid = str(uuid.uuid4())
    st.session_state.chat_sessions[sid] = []  # List[Tuple[role, content]]
    st.session_state.current_session = sid

# RAG용 객체
if "rag" not in st.session_state:
    st.session_state.rag = None  # type: ignore


# -----------------------------
# 2) 사이드바: 설정 + 세션 관리 + PDF 업로드
# -----------------------------
with st.sidebar:
    st.header("🔧 Configuration (Local vLLM)")

    st.slider(
        "Temperature",
        0.0,
        1.0,
        key="model_temperature",
    )
    st.slider(
        "Top-p",
        0.1,
        1.0,
        key="model_top_p",
    )
    st.slider(
        "Max Tokens",
        256,
        4096,
        key="model_max_tokens",
    )

    st.text_area("System Prompt", key="system_prompt", height=120)
    st.download_button("📥 Export Prompt", st.session_state.system_prompt, file_name="prompt.txt")

    st.divider()
    st.subheader("💬 Chat Sessions")

    # 세션 목록 버튼
    for session_id in list(st.session_state.chat_sessions.keys()):
        label = f"🔁 {session_id[:8]}"
        if st.button(label, key=f"switch_{session_id}"):
            st.session_state.current_session = session_id

    if st.button("➕ New Session"):
        new_id = str(uuid.uuid4())
        st.session_state.chat_sessions[new_id] = []
        st.session_state.current_session = new_id

    st.divider()
    st.subheader("📄 PDF for RAG")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file is not None:
        # 1) PDF 텍스트 추출
        reader = PdfReader(uploaded_file)
        full_text = ""
        for page in reader.pages:
            full_text += (page.extract_text() or "") + "\n"

        # 2) 간단한 chunking (문자 수 기준)
        def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
            text = text.replace("\r", " ").replace("\n", " ")
            tokens = list(text)
            chunks = []
            start = 0
            while start < len(tokens):
                end = start + chunk_size
                chunk = "".join(tokens[start:end]).strip()
                if chunk:
                    chunks.append(chunk)
                start = end - overlap
                if start < 0:
                    start = 0
            return chunks

        chunks = chunk_text(full_text, chunk_size=500, overlap=100)

        # 3) SimpleRAG에 문서 추가
        rag = SimpleRAG()
        added = rag.add_documents(chunks)

        st.session_state.rag = rag
        st.success(f"✅ PDF 로드 및 임베딩 완료! (추가된 청크 수: {added} 개)")


# -----------------------------
# 3) 메인 영역: 채팅 인터페이스
# -----------------------------
st.title("🤖 Local RAG Chatbot (vLLM)")

current_session_id = st.session_state.current_session
history: List[Tuple[str, str]] = st.session_state.chat_sessions[current_session_id]

# 과거 대화 표시
for role, msg in history:
    with st.chat_message(role):
        st.markdown(msg)

# 사용자 입력
user_input = st.chat_input("질문을 입력하세요...")

if user_input:
    # 1) 사용자 메시지 화면 & 세션에 기록
    st.session_state.chat_sessions[current_session_id].append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) (선택) RAG 컨텍스트 생성
    context_text = ""
    if st.session_state.rag is not None:
        context_text = st.session_state.rag.get_context(user_input, k=3)

    # 3) vLLM에 보낼 messages 구성
    messages = []

    # (1) 기본 system 프롬프트
    messages.append({"role": "system", "content": st.session_state.system_prompt})

    # (2) RAG 컨텍스트를 별도의 system 메시지로 전달
    if context_text:
        messages.append(
            {
                "role": "system",
                "content": (
                    "다음은 사용자가 질문할 때 참고해야 할 문서 컨텍스트입니다. "
                    "관련 있을 때만 활용하고, 관련이 없으면 무시하세요.\n\n"
                    f"{context_text}"
                ),
            }
        )

    # (3) 기존 대화 히스토리
    #    vLLM이 전체 대화 맥락을 이해하도록 user/assistant 메시지 모두 전달
    for role, msg in st.session_state.chat_sessions[current_session_id]:
        messages.append({"role": role, "content": msg})

    # 4) vLLM 호출
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = chat_completion(
                    messages=messages,
                    temperature=st.session_state.model_temperature,
                    top_p=st.session_state.model_top_p,
                    max_tokens=st.session_state.model_max_tokens,
                )
            except Exception as e:
                reply = f"❌ vLLM 호출 중 오류가 발생했습니다: {e}"

            st.markdown(reply)
            st.session_state.chat_sessions[current_session_id].append(("assistant", reply))

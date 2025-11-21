from typing import List, Tuple

class SimpleRAG:
    """
    간단하지만 실전용으로 쓸 수 있는 RAG 클래스.

    - SentenceTransformer로 문서 임베딩
    - FAISS IndexFlatIP (inner product) 기반 유사도 검색
    - add_documents: 문서 리스트 추가
    - retrieve: 쿼리와 유사한 top-k 문서 (idx, score, text) 반환
    - get_context: top-k 문서를 하나의 컨텍스트 문자열로 병합

    Streamlit 챗봇 / app.py 에서 바로 사용하기 좋게 설계됨.
    """

    def __init__(
        self,
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        embed_model_name : str
            SentenceTransformer 모델 이름.
        device : str | None
            "cuda", "cpu" 등. None이면 SentenceTransformer 기본값 사용.
        """
        from sentence_transformers import SentenceTransformer
        import faiss

        # SentenceTransformer 로더
        self.encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )

        # FAISS 관련
        self.faiss = faiss
        self.index = None  # type: ignore

        # 원본 문서 텍스트 저장
        self.docs: List[str] = []

    # -----------------------------
    # 내부 헬퍼
    # -----------------------------
    def _ensure_index(self, dim: int) -> None:
        """
        FAISS Index가 없으면 생성한다.
        Inner Product 기반 유사도 검색(IndexFlatIP) 사용.
        """
        if self.index is None:
            self.index = self.faiss.IndexFlatIP(dim)

    # -----------------------------
    # Public API
    # -----------------------------
    def add_documents(
        self,
        docs: List[str],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress_bar: bool = False,
    ) -> int:
        """
        문서 리스트를 받아 임베딩하고 FAISS 인덱스에 추가한다.

        Parameters
        ----------
        docs : List[str]
            문서 텍스트 리스트.
        batch_size : int
            임베딩 시 배치 크기.
        normalize : bool
            True면 임베딩을 L2 정규화하여 inner product = cosine 유사도와 비슷하게 사용.
        show_progress_bar : bool
            SentenceTransformer encode 진행바 출력 여부.

        Returns
        -------
        int
            실제로 추가된 문서 개수.
        """
        # 1) 빈 문자열 / None 등은 걸러낸다
        cleaned_docs = [d.strip() for d in docs if d and d.strip()]
        if not cleaned_docs:
            return 0

        # 2) 문서 임베딩
        embeddings = self.encoder.encode(
            cleaned_docs,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize,
        )

        # 3) FAISS 인덱스 준비
        dim = embeddings.shape[1]
        self._ensure_index(dim)

        # 4) float32로 캐스팅 후 인덱스에 추가
        embeddings = embeddings.astype("float32")
        self.index.add(embeddings)  # type: ignore

        # 5) 원본 문서 저장 (검색 결과에서 텍스트 반환용)
        self.docs.extend(cleaned_docs)

        return len(cleaned_docs)

    def retrieve(
        self,
        query: str,
        k: int = 3,
        normalize: bool = True,
    ) -> List[Tuple[int, float, str]]:
        """
        쿼리와 유사한 상위 k개의 문서를 반환.

        Parameters
        ----------
        query : str
            검색 쿼리 텍스트.
        k : int
            상위 몇 개의 문서를 가져올지.
        normalize : bool
            True면 쿼리 임베딩도 정규화.

        Returns
        -------
        List[Tuple[int, float, str]]
            (doc_index, score, doc_text) 튜플 리스트.
            인덱스가 없거나 문서가 없으면 빈 리스트.
        """
        if self.index is None or not self.docs:
            # 아직 문서가 하나도 추가되지 않은 상태
            return []

        if not query or not query.strip():
            return []

        # 1) 쿼리 임베딩
        q_vec = self.encoder.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        ).astype("float32")

        # 2) FAISS 검색
        k = min(k, len(self.docs))
        distances, indices = self.index.search(q_vec, k)  # type: ignore

        results: List[Tuple[int, float, str]] = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.docs):
                continue
            results.append((int(idx), float(score), self.docs[idx]))

        return results

    def get_context(
        self,
        query: str,
        k: int = 3,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """
        쿼리에 대해 retrieve 한 top-k 문서들을 하나의 컨텍스트 문자열로 합친다.

        Parameters
        ----------
        query : str
            사용자 쿼리.
        k : int
            top-k 문서 개수.
        separator : str
            문서 사이에 넣을 구분자 문자열.

        Returns
        -------
        str
            LLM 프롬프트에 그대로 붙여넣기 좋은 컨텍스트 문자열.
            검색 결과가 없으면 빈 문자열.
        """
        results = self.retrieve(query, k=k)
        if not results:
            return ""

        texts = [text for _, _, text in results]
        return separator.join(texts)

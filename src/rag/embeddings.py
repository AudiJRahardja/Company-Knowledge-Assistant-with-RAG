"""Turn text into vectors (embeddings) with a local multilingual model.

We use BGE-M3. It maps English and Indonesian text into the same vector space, so an Indonesian question
like "berapa hari cuti tahunan?" lands close to the English "Annual Leave Policy" chunk, and the other way
around. It runs locally, so no document text is sent to an outside service for search.
"""

from functools import cached_property


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "auto"):
        self.model_name = model_name
        self.device = device

    @cached_property
    def model(self):
        # Imported here so that modules which only need the types (and the tests) load quickly.
        import torch
        from sentence_transformers import SentenceTransformer

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(self.model_name, device=device)
        model.max_seq_length = 1024  # our chunks are well under this
        return model

    @property
    def dimension(self) -> int:
        model = self.model
        # sentence-transformers renamed this method in newer versions; support both.
        getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return getter()

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        vectors = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 32
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

"""Re-score search candidates with a cross-encoder (reranker).

The embedding model turns the question and each chunk into vectors *separately*, which is fast enough to
search everything but loses detail. A cross-encoder reads the question and one chunk *together* and outputs
a relevance score, which is much more accurate but too slow to run on every chunk. So we use it as a second
stage: search finds ~20 candidates, the reranker puts them in the best order, we keep the top 5.

Model: BAAI/bge-reranker-v2-m3, multilingual (works for English and Indonesian, and across them), runs
locally like the embedding model.
"""

from dataclasses import replace
from functools import cache, cached_property

from rag.store import Hit


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "auto"):
        self.model_name = model_name
        self.device = device

    @cached_property
    def model(self):
        import torch
        from sentence_transformers import CrossEncoder

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # On GPU use half precision: half the memory, practically the same ranking quality.
        kwargs = {"model_kwargs": {"torch_dtype": torch.float16}} if device == "cuda" else {}
        return CrossEncoder(self.model_name, device=device, max_length=512, **kwargs)

    def rerank(self, question: str, hits: list[Hit]) -> list[Hit]:
        """Return the hits sorted by reranker score (best first). Hit.score becomes that score (0..1)."""
        if not hits:
            return []
        scores = self.model.predict(
            [(question, f"{h.title}\n{h.text}") for h in hits], activation_fn=_sigmoid(), show_progress_bar=False
        )
        ranked = sorted(zip(hits, scores), key=lambda pair: float(pair[1]), reverse=True)
        return [replace(h, score=round(float(s), 4)) for h, s in ranked]


def _sigmoid():
    import torch

    return torch.nn.Sigmoid()


@cache
def get_reranker(model_name: str, device: str = "auto") -> Reranker:
    """One shared instance per model, so the app's two assistants don't load it twice."""
    return Reranker(model_name, device)

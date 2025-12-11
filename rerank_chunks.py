from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from typing import List
from typing import Tuple

from timer import timer


@timer
def rerank_chunks(
        cross_encoder: CrossEncoder,
        query: str,
        chunks: List[Document],
        top_k_chunks: int) -> List[Tuple[Document, float]]:
    """
    Rerank and trim a list of Document based on contextual relevance with the query.

    Args:
        cross_encoder: instance of the Cross-Encoder model.
        query: user query string.
        chunks: list of candidate documents retrieved from the vector store.
        top_k_chunks: number of the most relevant documents that must be selected.

    Returns:
        List of the top_k_chunks most relevant chunks, sorted by decreasing order of relevance with the query.
    """

    # Associate each chunk to the query to create (query, content) pairs
    query_content_pairs = [[query, chunk.page_content] for chunk in chunks]

    # Score each (query, content) pair based on contextual relevance
    scores = cross_encoder.predict(query_content_pairs)

    # Convert the scores to Python floats, to avoid unnecessary "np.float32(...)"
    scores = [float(s) for s in scores]

    # Create a list of tuples (chunk, score)
    reranked_chunks = list(zip(chunks, scores))

    # Sort the list by decreasing scores
    reranked_chunks = sorted(reranked_chunks, key=lambda x: x[1], reverse=True)

    # Trims the list to preserve only the most relevant chunks
    selected_chunks = [{"chunk": chunk, "score": score} for chunk, score in reranked_chunks[:top_k_chunks]]

    return selected_chunks


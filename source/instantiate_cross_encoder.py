from sentence_transformers import CrossEncoder


def instantiate_cross_encoder() -> CrossEncoder:
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2")
    return cross_encoder

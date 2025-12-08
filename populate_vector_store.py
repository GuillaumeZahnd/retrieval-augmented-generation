from instantiate_embedding_model import instantiate_embedding_model
from instantiate_vector_store import instantiate_vector_store


def populate_vector_store(chunks) -> None:
    embeddings_model = instantiate_embedding_model()
    vector_store = instantiate_vector_store(embeddings_model=embeddings_model)
    _ = vector_store.add_documents(documents=chunks)

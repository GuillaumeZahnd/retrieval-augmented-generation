import asyncio
import os
from langchain.tools import tool
from langchain.agents import create_agent

from source.instantiate_cross_encoder import instantiate_cross_encoder
from source.instantiate_embedding_model import instantiate_embedding_model
from source.vector_database import VectorDatabase
from source.large_language_model import LargeLanguageModel
from source.rerank_chunks import rerank_chunks
from source.query_expansion import QueryExpansion
from source.retrieve_candidate_chunks import retrieve_candidate_chunks


async def rag(query: str, pdf_url: str) -> str:

    # Stage 0: Instantiation
    cross_encoder = instantiate_cross_encoder()
    embedding_model = instantiate_embedding_model()
    vector_database = VectorDatabase(embedding_model=embedding_model)
    llm = LargeLanguageModel(temperature=0.0)
    query_expansion = QueryExpansion(nb_variants=3, temperature=0.1)

    # Stage 1: Populate vector database
    vector_database.populate_vector_store(pdf_url=pdf_url)

    # Stage 2: Query expansion
    QUERY_EXPANSION = True
    if QUERY_EXPANSION:
        top_k_chunks_for_mini_retrieval = 5
        retrieved_chunks = await retrieve_candidate_chunks(
            query=user_query, vector_database=vector_database, top_k_chunks=top_k_chunks_for_mini_retrieval)
        alternative_queries, _ = query_expansion.expand_query(query=user_query, chunks=retrieved_chunks)

        retrieval_query = [user_query] + alternative_queries
    else:
        retrieval_query = user_query

    # Stage 3: Bi-encoder to pre-select candidate chunks (high recall)
    top_k_chunks_for_bi_encoder = 100
    retrieved_chunks = await retrieve_candidate_chunks(
        query=retrieval_query, vector_database=vector_database, top_k_chunks=top_k_chunks_for_bi_encoder)

    # Stage 4: Cross-encoder to refine the selection (high precision)
    top_k_chunks_for_cross_encoder = 10
    reranked_chunks = rerank_chunks(
        cross_encoder=cross_encoder,
        query=user_query,
        chunks=retrieved_chunks,
        top_k_chunks=top_k_chunks_for_cross_encoder)

    # Stage 5: Formulate an answer to the query using the most relevant chunks
    answer = llm.get_answer_from_query(query=user_query, chunks=reranked_chunks)

    return(answer)


if __name__ == "__main__":

    user_query = "how does the working class score victory points (VPs)?"

    pdf_url = "https://hegemonicproject.com/wp-content/uploads/2023/04/Hegemony-English-Rulebook-v1.2.pdf"

    answer = asyncio.run(rag(query=user_query, pdf_url=pdf_url))

    print("-"*64)
    print("Query:  {}".format(user_query))
    print("Answer:  {}".format(answer))

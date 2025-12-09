import os
from langchain_mistralai import ChatMistralAI


def instantiate_language_model():

    os.environ.get("MISTRAL_API_KEY")

    model_name = "magistral-medium-2506"

    language_model = ChatMistralAI(
        model=model_name,
        temperature=0,
        max_retries=0)

    return language_model

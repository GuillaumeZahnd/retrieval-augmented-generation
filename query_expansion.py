import os
import re
from langchain_community.llms import LlamaCpp

from timer import timer

class QueryExpansion():
    def __init__(self, nb_variants: int):
        super().__init__()

        self.nb_variants = nb_variants
        self._language_model = self._instantiate_language_model()

    @timer
    def expand_query(self, user_query: str, context: str):
        prompt = self._build_expansion_prompt(
                    user_query=user_query,
                    nb_variants=self.nb_variants,
                    context=context)
        return self._language_model.invoke(prompt)


    def _build_expansion_prompt(self, user_query: str, context: str, nb_variants: int) -> str:

        instructions = f"""
            Your role:
            - You are a query expansion engine.

            Your task:
            - Generate {nb_variants} alternative queries by rephrasing the user query.
            - The alternative queries must have the same meaning as the user query.
            - Do not ask about anything diverging from the user query.
            """

        prompt = f"""
            {instructions}\n\n
            User query: {user_query}\n\n
            Context: {context}\n\n
            """

        return prompt


    def format_llm_output(self, raw_output: str) -> str:
        query_pattern = re.compile(r'^\s*\d+\.\s*(.*?)$', re.MULTILINE)
        clean_output = query_pattern.findall(raw_output)
        return clean_output


    def _instantiate_language_model(self):

        model_name = "gemma-2b-it.Q5_K_M.gguf"
        nb_tokens_per_variant = 25
        max_tokens = self.nb_variants * nb_tokens_per_variant

        language_model = LlamaCpp(
            model_path=os.path.join("models", model_name),
            n_ctx=8192,
            n_threads=8,
            n_gpu_layers=35,
            n_batch=64,
            temperature=0.1,
            top_p=0.9,
            max_tokens=max_tokens,
            verbose=False)

        return language_model

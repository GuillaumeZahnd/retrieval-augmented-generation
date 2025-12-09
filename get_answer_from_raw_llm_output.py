import json
import ast


def get_answer_from_raw_llm_output(raw_output: str) -> str:

    raw_output = ast.literal_eval(raw_output)

    final_output = raw_output[-1]

    final_text = final_output['text']

    stripped_text = final_text.strip()

    if stripped_text.startswith("```json"):
        stripped_text = stripped_text[len("```json"):].strip()

    if stripped_text.endswith("```"):
        stripped_text = stripped_text[:-len("```"):].strip()

    answer_field = json.loads(stripped_text)

    answer = answer_field["answer"]

    return answer


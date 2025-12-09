import re
import fitz


def _join_broken_lines(text: str) -> str:

    # 1. Handle Hyphenated Words:
    # Find a pattern where a word is broken with a hyphen and a newline.
    # Pattern: [a-z] (lowercase letter) followed by - and a newline.
    # The \s* (any whitespace) ensures we catch breaks with trailing spaces.
    # Replacement: Remove the hyphen and the newline/whitespace.
    text = re.sub(r'([a-z])-\s*\n\s*', r'\1', text, flags=re.IGNORECASE)

    # 2. Handle General Mid-sentence Breaks:
    # Find a newline character that is not preceded by a sentence-endind punctuation (. ? !).
    # Replace it with a single space.
    # Negative lookbehind: (?<![.?!\u2026]) ensures the break is not after (. ? ! ...).
    # Replacement: A single space to rejoin the sentence.
    text = re.sub(r'(?<![.?!\u2026])\s*\n\s*', ' ', text)

    # 3. Clean up extra spacing:
    # Replace multiple spaces with a single space and remove leading/trailing spaces.
    text = re.sub(r'\s{2,}', ' ', text).strip()

    return text


def extract_text_from_pdf(path_to_pdf: str) -> str:
    text = []
    with fitz.open(path_to_pdf) as document:
        for page in document:
            text.append(page.get_text("text").strip())
    text = "\n".join(text)

    text = _join_broken_lines(text=text)

    return text

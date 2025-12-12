import os
import re
import fitz
import requests


def extract_text_from_pdf(url: str) -> str:

    path_to_pdf = "data"
    filename = "tmp.pdf"
    _download_pdf(url=url, path_to_pdf=path_to_pdf, filename=filename)
    text = _read_pdf(path_to_pdf=path_to_pdf, filename=filename)

    return text


def _read_pdf(path_to_pdf: str, filename: str) -> str:

    text = []
    with fitz.open(os.path.join(path_to_pdf, filename)) as document:
        for page in document:
            text.append(page.get_text("text").strip())
    text = "\n".join(text)
    text = _join_broken_lines(text=text)

    return text


def _download_pdf(url: str, path_to_pdf: str, filename: str) -> None:

    if not os.path.exists(path_to_pdf):
        os.makedirs(path_to_pdf)

    pdf_data = requests.get(url).content
    with open(os.path.join(path_to_pdf, filename), "wb") as fid:
        fid.write(pdf_data)


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

import re


def clean_llm_output(text: str) -> str:
    """
    Remove <think></think> tags and their content from the text.

    Args:
        text (str): The text containing think tags

    Returns:
        str: Cleaned text with think tags and their content removed
    """
    # Pattern explanation:
    # <think> - matches the opening tag
    # .*? - matches any characters (non-greedy)
    # </think> - matches the closing tag
    # re.DOTALL - allows .* to match newlines
    pattern = r"<think>.*?</think>"

    # Remove the tags and their content
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)

    # Optional: Remove any extra whitespace that might be left
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

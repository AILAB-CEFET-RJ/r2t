import json
from openai import OpenAI


def segment_special_appeal(
    appeal_text: str,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.0
) -> dict:
    """
    Segments a Brazilian special appeal into rhetorical/legal sections.

    Parameters
    ----------
    appeal_text : str
        Full text of the special appeal.
    model : str
        OpenAI model to use.
    temperature : float
        Sampling temperature (keep at 0 for determinism).

    Returns
    -------
    dict
        JSON-like dict with segmented sections.
    """

    client = OpenAI()

    system_prompt = (
        "You are a legal text analysis assistant. "
        "Your task is to segment Brazilian legal documents based on their rhetorical and functional structure. "
        "Do NOT summarize, classify, interpret legal outcomes, or infer themes. "
        "Only segment the text into structurally meaningful sections."
    )

    user_prompt = f"""
Segment the following Brazilian special appeal into the sections listed below.

Sections:
- identification
- procedural_history
- facts
- legal_issues
- legal_arguments
- requests
- other

Instructions:
- Preserve the original text verbatim in each section.
- Do NOT rewrite, paraphrase, or summarize.
- If a section is not present, return it as an empty string.
- Return the output strictly as a valid JSON object.
- Do not include any explanation or commentary.

TEXT:
\"\"\"
{appeal_text}
\"\"\"
"""

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    raw_output = response.choices[0].message.content

    try:
        segmented = json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise ValueError("Model output is not valid JSON") from e

    return segmented


if __name__ == "__main__":
    # Example usage (for testing)
    with open("recurso_especial.txt", "r", encoding="utf-8") as f:
        text = f.read()

    result = segment_special_appeal(text)

    with open("recurso_segmentado.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Segmentation completed. Output saved to recurso_segmentado.json")

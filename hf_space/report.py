import os
from google import genai

_client = None

REPORT_PROMPT = """
You are an expert radiologist AI assistant. Based on the following chest X-ray analysis results,
generate a concise, professional medical report.

Analysis Results:
- Diagnosis: {diagnosis}
- Confidence: {confidence}%
- Normal Probability: {normal_prob}%
- Pneumonia Probability: {pneumonia_prob}%

Write a structured medical report with these sections:
1. **Clinical Impression** (1-2 sentences - state the primary finding clearly)
2. **Radiological Findings** (2-3 sentences - describe what was detected)
3. **Assessment & Recommendation** (2-3 sentences - clinical next steps)

Keep the language professional but accessible. Do NOT mention AI or machine learning in the report -
frame it as a radiological assessment. Use appropriate medical terminology.
"""


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    return _client


def generate_report(diagnosis, confidence, normal_prob, pneumonia_prob):
    prompt = REPORT_PROMPT.format(
        diagnosis=diagnosis, confidence=confidence,
        normal_prob=normal_prob, pneumonia_prob=pneumonia_prob,
    )
    try:
        resp = _get_client().models.generate_content(
            model="gemini-flash-latest", contents=prompt
        )
        return resp.text.strip()
    except Exception as e:
        return (f"_Report generation unavailable ({str(e)[:120]}). "
                "The diagnosis and Grad-CAM above are fully functional._")

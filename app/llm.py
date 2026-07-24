import google.generativeai as genai
import os

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

_model = genai.GenerativeModel("gemini-1.5-flash")

REPORT_PROMPT = """
You are an expert radiologist AI assistant. Based on the following chest X-ray analysis results,
generate a concise, professional medical report.

Analysis Results:
- Diagnosis: {diagnosis}
- Confidence: {confidence}%
- Normal Probability: {normal_prob}%
- Pneumonia Probability: {pneumonia_prob}%

Write a structured medical report with these sections:
1. **Clinical Impression** (1-2 sentences — state the primary finding clearly)
2. **Radiological Findings** (2-3 sentences — describe what the AI model detected)
3. **Assessment & Recommendation** (2-3 sentences — clinical next steps)

Keep the language professional but accessible. Do NOT mention AI or machine learning in the report —
frame it as a radiological assessment. Use appropriate medical terminology.
"""


def generate_report(diagnosis: str, confidence: float, normal_prob: float, pneumonia_prob: float) -> str:
    prompt = REPORT_PROMPT.format(
        diagnosis=diagnosis,
        confidence=confidence,
        normal_prob=normal_prob,
        pneumonia_prob=pneumonia_prob,
    )
    try:
        response = _model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Report generation failed: {str(e)}. Please ensure GEMINI_API_KEY is set."

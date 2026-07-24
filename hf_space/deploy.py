import os
import sys
from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
GEMINI_KEY = os.environ["GEMINI_API_KEY"]
REPO_ID = "bhaveshkhaple/ai-medical-intelligence-platform"
HERE = os.path.dirname(__file__)

api = HfApi(token=HF_TOKEN)

# 1. Create the Space (Streamlit SDK)
api.create_repo(
    repo_id=REPO_ID,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
)
print(f"Space ready: https://huggingface.co/spaces/{REPO_ID}")

# 2. Set the Gemini API key as a Space secret
api.add_space_secret(repo_id=REPO_ID, key="GEMINI_API_KEY", value=GEMINI_KEY)
print("Secret GEMINI_API_KEY set")

# 3. Upload files
for fname in ["app.py", "inference.py", "report.py", "database.py",
              "requirements.txt", "README.md", "model.pth", "Dockerfile"]:
    api.upload_file(
        path_or_fileobj=os.path.join(HERE, fname),
        path_in_repo=fname,
        repo_id=REPO_ID,
        repo_type="space",
    )
    print(f"  uploaded {fname}")

print("\nDONE. Space building at:")
print(f"https://huggingface.co/spaces/{REPO_ID}")

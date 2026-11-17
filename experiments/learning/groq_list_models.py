"""Phase 11 diagnostic — list Groq's currently available models.
Helps confirm whether a model name in .env is still valid."""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

response = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
)
response.raise_for_status()

for model in response.json()["data"]:
    print(model["id"])
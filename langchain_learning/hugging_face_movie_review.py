
import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv(Path(__file__).with_name(".env"))
hf_token = os.getenv("HF_API_KEY")

# STEP 1: Load dataset
dataset = load_dataset("stanfordnlp/imdb", token=hf_token)

# STEP 2: Read one customer review
review = dataset["train"][0]["text"]

print("Customer Review:")
print(review)

# STEP 3: Load pre-trained sentiment model
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    token=hf_token,
)

# STEP 4: Perform inference
result = classifier(review)

# STEP 5: Display prediction
print("Prediction:")
print(result)

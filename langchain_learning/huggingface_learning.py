from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

reviews = [
    "This mobile is amazing",
    "This mobile is terrible",
    "The camera quality is excellent"
]

results = classifier(reviews)

for review, result in zip(reviews, results):
    print(review)
    print(result)
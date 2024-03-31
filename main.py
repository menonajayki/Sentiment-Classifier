from transformers import LongformerTokenizer, LongformerForSequenceClassification
from transformers import pipeline


model_name = "allenai/longformer-base-4096"
tokenizer = LongformerTokenizer.from_pretrained(model_name)
model = LongformerForSequenceClassification.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

def classify_sentiment(text):
    result = classifier(text)
    return result[0]['label']

text = "I loved the movie! The acting was fantastic."
print("Sentiment:", classify_sentiment(text))

text = "The film was a complete disaster. Terrible acting and a weak plot."
print("Sentiment:", classify_sentiment(text))

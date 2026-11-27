import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")
MAX_LEN = 64

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()


def predict(texts):
    enc = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = model(**enc).logits.softmax(-1).cpu()
    return [(t, model.config.id2label[int(i)], float(p)) for t, i, p in zip(texts, probs.argmax(1), probs.max(1).values)]


if __name__ == "__main__":
    for text, label, conf in predict(sys.argv[1:]):
        print(f"{label} ({conf:.2f}): {text}")

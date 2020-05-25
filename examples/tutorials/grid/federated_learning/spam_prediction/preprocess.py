import numpy as np
import pandas as pd
import re

from nltk.corpus import stopwords  # noqa: F401

STOPWORDS = {}  # {stopwords.words('english')}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text


def tokenize(text, word_to_idx):
    tokens = []
    for word in text.split():
        tokens.append(word_to_idx[word])
    return tokens


def pad_and_truncate(messages, max_length=30):
    features = np.zeros((len(messages), max_length), dtype=int)
    for i, sms in enumerate(messages):
        if len(sms):
            features[i, -len(sms) :] = sms[:max_length]
    return features


def preprocess_spam():
    data = pd.read_csv("./data/SMSSpamCollection", sep="\t", header=None, names=["label", "sms"])
    data.sms = data.sms.apply(clean_text)
    words = set((" ".join(data.sms)).split())
    word_to_idx = {word: i for i, word in enumerate(words, 1)}
    tokens = data.sms.apply(lambda x: tokenize(x, word_to_idx))
    inputs = pad_and_truncate(tokens)

    labels = np.array((data.label == "spam").astype(int))

    np.save("./data/labels.npy", labels)
    np.save("./data/inputs.npy", inputs)

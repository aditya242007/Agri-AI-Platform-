import re
import joblib
import nltk
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


nltk.download("punkt", quiet=True)


def preprocess(text):
    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z0-9 ]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    tokens = nltk.word_tokenize(text)

    return " ".join(tokens)

print("LOADING CHATBOT DATASET")

data = pd.read_csv(
    "chatbot_dataset_500.csv"
)

required_columns = [
    "Question",
    "Answer",
    "Intent"
]

for column in required_columns:

    if column not in data.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )

data = data.dropna(
    subset=[
        "Question",
        "Answer",
        "Intent"
    ]
).copy()

data["Question"] = data[
    "Question"
].astype(str)

data["Answer"] = data[
    "Answer"
].astype(str)

data["Intent"] = data[
    "Intent"
].astype(str)

data["Processed_Question"] = data[
    "Question"
].apply(preprocess)

print(
    "\nDataset Size:",
    len(data)
)

print(
    "Number of Intents:",
    data["Intent"].nunique()
)

print("\nIntent Distribution:\n")

print(
    data["Intent"].value_counts()
)

print("SPLITTING DATASET")

X = data["Processed_Question"]
y = data["Intent"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    "\nTraining Questions:",
    len(X_train)
)

print(
    "Testing Questions:",
    len(X_test)
)

print("CREATING TF-IDF FEATURES")

evaluation_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_vectorized = (
    evaluation_vectorizer.fit_transform(
        X_train
    )
)

X_test_vectorized = (
    evaluation_vectorizer.transform(
        X_test
    )
)

print(
    "\nTF-IDF Feature Count:",
    X_train_vectorized.shape[1]
)

print("TRAINING INTENT MODEL")

evaluation_model = MultinomialNB(
    alpha=0.3
)

evaluation_model.fit(
    X_train_vectorized,
    y_train
)

print(
    "\nIntent model trained successfully!"
)

print("EVALUATING MODEL")

predictions = evaluation_model.predict(
    X_test_vectorized
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    f"\nTest Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

print("TRAINING FINAL MODEL")

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

all_question_vectors = (
    vectorizer.fit_transform(
        data["Processed_Question"]
    )
)

intent_model = MultinomialNB(
    alpha=0.3
)

intent_model.fit(
    all_question_vectors,
    data["Intent"]
)

print(
    "\nFinal model trained on entire dataset."
)

print("SAVING CHATBOT MODELS")

joblib.dump(
    vectorizer,
    "chatbot_vectorizer.pkl"
)

joblib.dump(
    intent_model,
    "chatbot_intent_model.pkl"
)

joblib.dump(
    all_question_vectors,
    "chatbot_question_vectors.pkl"
)

joblib.dump(
    data,
    "chatbot_data.pkl"
)

print(
    "\nSaved: chatbot_vectorizer.pkl"
)

print(
    "Saved: chatbot_intent_model.pkl"
)

print(
    "Saved: chatbot_question_vectors.pkl"
)

print(
    "Saved: chatbot_data.pkl"
)

print("CHATBOT TRAINING COMPLETE")

print(
    f"\nFinal Dataset Size: {len(data)}"
)


print(
    f"Number of Intents: {data['Intent'].nunique()}"
)


print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)
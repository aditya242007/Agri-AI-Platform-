import base64
import re

import cv2
import joblib
import nltk
import numpy as np
import pandas as pd

from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State


print("Loading models...")

crop_yield_model = joblib.load("crop_yield_model.pkl")
svm_model = joblib.load("maize_svm_model.pkl")
label_encoder = joblib.load("maize_label_encoder.pkl")

chatbot_vectorizer = joblib.load("chatbot_vectorizer.pkl")
chatbot_intent_model = joblib.load("chatbot_intent_model.pkl")
chatbot_question_vectors = joblib.load("chatbot_question_vectors.pkl")
chatbot_data = joblib.load("chatbot_data.pkl")

print("Models loaded successfully!")

nltk.download("punkt", quiet=True)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    tokens = nltk.word_tokenize(text)
    return " ".join(tokens)


def get_chatbot_response(question):

    if question is None or question.strip() == "":
        return "Please enter a question."

    processed_question = preprocess_text(question)

    user_vector = chatbot_vectorizer.transform(
        [processed_question]
    )

    predicted_intent = chatbot_intent_model.predict(
        user_vector
    )[0]

    intent_probabilities = chatbot_intent_model.predict_proba(
        user_vector
    )[0]

    intent_confidence = np.max(
        intent_probabilities
    )

    print("\nUser Question:", question)
    print("Predicted Intent:", predicted_intent)
    print(
        "Intent Confidence:",
        round(intent_confidence, 3)
    )

    INTENT_THRESHOLD = 0.30

    if intent_confidence < INTENT_THRESHOLD:
        return (
            "I'm not confident that I understand your question. "
            "Please ask me something related to agriculture, "
            "such as crops, irrigation, rainfall, soil, "
            "fertilizer, weather, or crop yield."
        )

    intent_indices = chatbot_data.index[
        chatbot_data["Intent"] == predicted_intent
    ].tolist()

    if len(intent_indices) == 0:
        return (
            "I understand the topic of your question, "
            "but I don't currently have enough information "
            "to answer it."
        )

    intent_vectors = chatbot_question_vectors[
        intent_indices
    ]

    similarities = cosine_similarity(
        user_vector,
        intent_vectors
    ).flatten()

    best_local_index = similarities.argmax()
    best_score = similarities[best_local_index]
    best_index = intent_indices[best_local_index]

    matched_question = chatbot_data.loc[
        best_index,
        "Question"
    ]

    answer = chatbot_data.loc[
        best_index,
        "Answer"
    ]

    print(
        "Matched Question:",
        matched_question
    )

    print(
        "Similarity Score:",
        round(best_score, 3)
    )

    SIMILARITY_THRESHOLD = 0.15

    if best_score < SIMILARITY_THRESHOLD:
        return (
            "I understand the topic of your question, "
            "but I don't have a sufficiently relevant "
            "answer in my knowledge base."
        )

    return answer


IMAGE_SIZE = (128, 128)


def classify_image(contents):

    if contents is None:
        return None

    try:
        content_type, content_string = contents.split(",")

        decoded = base64.b64decode(
            content_string
        )

        image_array = np.frombuffer(
            decoded,
            np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:
            print("Unable to decode image")
            return None

        image = cv2.resize(
            image,
            IMAGE_SIZE
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )

        features = features.reshape(
            1,
            -1
        )

        prediction = svm_model.predict(
            features
        )[0]

        disease = label_encoder.inverse_transform(
            [prediction]
        )[0]

        probabilities = svm_model.predict_proba(
            features
        )[0]

        confidence = np.max(
            probabilities
        ) * 100

        print("Prediction:", disease)
        print("Confidence:", confidence)

        return disease, confidence

    except Exception as e:
        print(
            "Image classification error:",
            e
        )
        return None


def stat_card(icon, label, value, note):

    return html.Div(
        className="stat-card",
        children=[
            html.Div(
                icon,
                className="stat-icon"
            ),

            html.Div(
                label,
                className="stat-label"
            ),

            html.Div(
                value,
                className="stat-value"
            ),

            html.Div(
                note,
                className="stat-note"
            )
        ]
    )


def hero_section(
    label,
    title,
    description,
    badge_label=None,
    badge_value=None
):

    children = [

        html.Div(
            className="hero-content",
            children=[
                html.P(
                    label,
                    className="hero-label"
                ),

                html.H1(
                    title,
                    className="hero-title"
                ),

                html.P(
                    description,
                    className="hero-description"
                )
            ]
        )

    ]

    if badge_label and badge_value:

        children.append(

            html.Div(
                className="hero-badge",
                children=[
                    html.Div(
                        badge_label,
                        className="hero-badge-label"
                    ),

                    html.Div(
                        badge_value,
                        className="hero-badge-value"
                    )
                ]
            )

        )

    return html.Div(
        className="hero",
        children=children
    )


def dashboard_page():

    return html.Div([

        hero_section(
            "AI-DRIVEN AGRICULTURE",
            "Crop Intelligence System",
            (
                "Machine learning tools for crop yield "
                "prediction, maize disease detection "
                "and agricultural assistance."
            ),
            "Integrated ML Models",
            "3 Models"
        ),

        html.Div(
            className="page-content",
            children=[

                html.H2(
                    "System Overview",
                    className="section-title"
                ),

                html.P(
                    (
                        "An integrated agricultural machine "
                        "learning platform built using "
                        "regression, image classification "
                        "and natural language processing."
                    ),
                    className="section-description"
                ),

                html.Div(
                    className="stats-grid",
                    children=[

                        stat_card(
                            "📈",
                            "Yield Model R²",
                            "91.32%",
                            "Linear Regression"
                        ),

                        stat_card(
                            "🍃",
                            "Disease Accuracy",
                            "86.41%",
                            "HOG + SVM"
                        ),

                        stat_card(
                            "🔬",
                            "Disease Classes",
                            "4",
                            "Maize leaf conditions"
                        ),

                        stat_card(
                            "💬",
                            "Assistant",
                            "NLP",
                            "TF-IDF + Naive Bayes"
                        )

                    ]
                ),

                html.Div(
                    className="content-card",
                    children=[

                        html.H3(
                            "🌾 Crop Intelligence",
                            className="card-title"
                        ),

                        html.P(
                            (
                                "Use the navigation menu to access "
                                "each machine learning component."
                            ),
                            className="card-subtitle"
                        ),

                        html.Div(
                            className="analytics-grid",
                            children=[

                                html.Div(
                                    className="model-card",
                                    children=[

                                        html.Div(
                                            "📈 Yield Prediction",
                                            className="model-name"
                                        ),

                                        html.Div(
                                            "REGRESSION",
                                            className="model-type"
                                        ),

                                        html.P(
                                            (
                                                "Estimate crop yield using "
                                                "region, soil, crop, rainfall, "
                                                "temperature, fertilizer, "
                                                "irrigation, weather and "
                                                "harvest information."
                                            )
                                        )

                                    ]
                                ),

                                html.Div(
                                    className="model-card",
                                    children=[

                                        html.Div(
                                            "🍃 Disease Detection",
                                            className="model-name"
                                        ),

                                        html.Div(
                                            "IMAGE CLASSIFICATION",
                                            className="model-type"
                                        ),

                                        html.P(
                                            (
                                                "Upload a maize leaf image "
                                                "to classify its condition "
                                                "using HOG image features "
                                                "and an SVM classifier."
                                            )
                                        )

                                    ]
                                ),

                                html.Div(
                                    className="model-card",
                                    children=[

                                        html.Div(
                                            "💬 Agriculture Assistant",
                                            className="model-name"
                                        ),

                                        html.Div(
                                            "NATURAL LANGUAGE PROCESSING",
                                            className="model-type"
                                        ),

                                        html.P(
                                            (
                                                "Ask agriculture-related "
                                                "questions using an NLP "
                                                "intent classification and "
                                                "similarity-based chatbot."
                                            )
                                        )

                                    ]
                                )

                            ]
                        )

                    ]
                )

            ]
        )

    ])


def yield_page():

    return html.Div([

        hero_section(
            "REGRESSION MODEL",
            "Crop Yield Prediction",
            (
                "Estimate crop yield from agricultural "
                "and environmental conditions."
            ),
            "Model R²",
            "91.32%"
        ),

        html.Div(
            className="page-content",
            children=[

                html.H2(
                    "Yield Predictor",
                    className="section-title"
                ),

                html.P(
                    (
                        "Enter the field conditions below. "
                        "All inputs are used by the trained "
                        "regression pipeline."
                    ),
                    className="section-description"
                ),

                html.Div(
                    className="content-card",
                    children=[

                        html.H3(
                            "Field Information",
                            className="card-title"
                        ),

                        html.P(
                            "Complete all fields before predicting.",
                            className="card-subtitle"
                        ),

                        html.Div(
                            className="form-grid",
                            children=[

                                html.Div(
                                    className="form-group",
                                    children=[
                                        html.Label(
                                            "Region",
                                            className="form-label"
                                        ),

                                        dcc.Dropdown(
                                            id="region",
                                            options=[
                                                {
                                                    "label": "North",
                                                    "value": "North"
                                                },
                                                {
                                                    "label": "South",
                                                    "value": "South"
                                                },
                                                {
                                                    "label": "East",
                                                    "value": "East"
                                                },
                                                {
                                                    "label": "West",
                                                    "value": "West"
                                                }
                                            ],
                                            placeholder="Select region"
                                        )
                                    ]
                                ),

                                html.Div(
                                    className="form-group",
                                    children=[
                                        html.Label(
                                            "Soil Type",
                                            className="form-label"
                                        ),

                                        dcc.Dropdown(
                                            id="soil",
                                            options=[
                                                {
                                                    "label": "Sandy",
                                                    "value": "Sandy"
                                                },
                                                {
                                                    "label": "Clay",
                                                    "value": "Clay"
                                                },
                                                {
                                                    "label": "Loam",
                                                    "value": "Loam"
                                                },
                                                {
                                                    "label": "Silt",
                                                    "value": "Silt"
                                                }
                                            ],
                                            placeholder="Select soil type"
                                        )
                                    ]
                                ),

                                html.Div(
                                    className="form-group",
                                    children=[
                                        html.Label(
                                            "Crop",
                                            className="form-label"
                                        ),

                                        dcc.Dropdown(
                                            id="crop",
                                            options=[
                                                {
                                                    "label": "Wheat",
                                                    "value": "Wheat"
                                                },
                                                {
                                                    "label": "Rice",
                                                    "value": "Rice"
                                                },
                                                {
                                                    "label": "Maize",
                                                    "value": "Maize"
                                                },
                                                {
                                                    "label": "Barley",
                                                    "value": "Barley"
                                                },
                                                {
                                                    "label": "Soybean",
                                                    "value": "Soybean"
                                                },
                                                {
                                                    "label": "Cotton",
                                                    "value": "Cotton"
                                                }
                                            ],
                                            placeholder="Select crop"
                                        )
                                    ]
                                ),

                                html.Div(
                                    className="form-group",
                                    children=[
                                        html.Label(
                                            "Weather Condition",
                                            className="form-label"
                                        ),

                                        dcc.Dropdown(
                                            id="weather",
                                            options=[
                                                {
                                                    "label": "Sunny",
                                                    "value": "Sunny"
                                                },
                                                {
                                                    "label": "Rainy",
                                                    "value": "Rainy"
                                                },
                                                {
                                                    "label": "Cloudy",
                                                    "value": "Cloudy"
                                                }
                                            ],
                                            placeholder="Select weather"
                                        )
                                    ]
                                ),

                                html.Div(
                                    className="form-group",
                                    children=[
                                        html.Label(
                                            "Rainfall (mm)",
                                            className="form-label"
                                        ),

                                        dcc.Input(
                                            id="rainfall",
                                            type="number",
                                            placeholder="Example: 750",
                                            className="input-field"
                                        )
                                    ]
                                ),

                                html.Div(
                                    className="form-group",
                                    children=[
                                        html.Label(
                                            "Temperature (°C)",
                                            className="form-label"
                                        ),

                                        dcc.Input(
                                            id="temperature",
                                            type="number",
                                            placeholder="Example: 25",
                                            className="input-field"
                                        )
                                    ]
                                ),

                                html.Div(
                                    className="form-group",
                                    children=[
                                        html.Label(
                                            "Fertilizer Used",
                                            className="form-label"
                                        ),

                                        dcc.Dropdown(
                                            id="fertilizer",
                                            options=[
                                                {
                                                    "label": "Yes",
                                                    "value": True
                                                },
                                                {
                                                    "label": "No",
                                                    "value": False
                                                }
                                            ],
                                            placeholder="Select option"
                                        )
                                    ]
                                ),

                                html.Div(
                                    className="form-group",
                                    children=[
                                        html.Label(
                                            "Irrigation Used",
                                            className="form-label"
                                        ),

                                        dcc.Dropdown(
                                            id="irrigation",
                                            options=[
                                                {
                                                    "label": "Yes",
                                                    "value": True
                                                },
                                                {
                                                    "label": "No",
                                                    "value": False
                                                }
                                            ],
                                            placeholder="Select option"
                                        )
                                    ]
                                ),

                                html.Div(
                                    className="form-group",
                                    children=[
                                        html.Label(
                                            "Days to Harvest",
                                            className="form-label"
                                        ),

                                        dcc.Input(
                                            id="harvest-days",
                                            type="number",
                                            placeholder="Example: 120",
                                            className="input-field"
                                        )
                                    ]
                                )

                            ]
                        ),

                        html.Br(),

                        html.Button(
                            "Predict Crop Yield",
                            id="predict-yield-button",
                            n_clicks=0,
                            className="primary-button"
                        ),

                        html.Div(
                            id="yield-output"
                        )

                    ]
                )

            ]
        )

    ])


def disease_page():

    return html.Div([

        hero_section(
            "IMAGE CLASSIFICATION",
            "Maize Disease Detection",
            (
                "Upload a maize leaf image and classify "
                "its condition using HOG features and SVM."
            ),
            "Test Accuracy",
            "86.41%"
        ),

        html.Div(
            className="page-content",
            children=[

                html.H2(
                    "Leaf Analysis",
                    className="section-title"
                ),

                html.P(
                    (
                        "The classifier recognizes four maize "
                        "leaf conditions."
                    ),
                    className="section-description"
                ),

                html.Div(
                    className="stats-grid",
                    children=[

                        stat_card(
                            "🟤",
                            "Disease",
                            "Cercospora",
                            "Leaf Spot"
                        ),

                        stat_card(
                            "🟠",
                            "Disease",
                            "Common Rust",
                            "Maize disease"
                        ),

                        stat_card(
                            "🌿",
                            "Condition",
                            "Healthy",
                            "Healthy leaf"
                        ),

                        stat_card(
                            "🍂",
                            "Disease",
                            "N. Leaf Blight",
                            "Northern Leaf Blight"
                        )

                    ]
                ),

                html.Div(
                    className="content-card",
                    children=[

                        html.H3(
                            "Upload Maize Leaf",
                            className="card-title"
                        ),

                        html.P(
                            (
                                "For the most reliable result, use "
                                "a clear maize leaf image similar "
                                "to the PlantVillage dataset."
                            ),
                            className="card-subtitle"
                        ),

                        dcc.Upload(
                            id="upload-image",

                            className="upload-box",

                            children=html.Div([
                                html.Div(
                                    "🍃",
                                    className="upload-icon"
                                ),

                                html.Div(
                                    "Upload a maize leaf image",
                                    className="upload-title"
                                ),

                                html.Div(
                                    (
                                        "Drag and drop an image "
                                        "or click to browse"
                                    ),
                                    className="upload-subtitle"
                                )
                            ]),

                            multiple=False
                        ),

                        html.Div(
                            className="image-result-grid",
                            children=[

                                html.Div(
                                    id="image-preview"
                                ),

                                html.Div(
                                    id="disease-output"
                                )

                            ]
                        )

                    ]
                )

            ]
        )

    ])


def chatbot_page():

    return html.Div([

        hero_section(
            "NATURAL LANGUAGE PROCESSING",
            "Agriculture Assistant",
            (
                "Ask questions about crop yield, rainfall, "
                "irrigation, soil, fertilizer and farming."
            ),
            "Assistant",
            "Online"
        ),

        html.Div(
            className="page-content",
            children=[

                html.Div(
                    className="chat-container",
                    children=[

                        html.Div(
                            className="chat-window",
                            children=[

                                html.Div(
                                    className="chat-header",
                                    children=[

                                        html.Div(
                                            "🌾",
                                            className="chat-avatar"
                                        ),

                                        html.Div([
                                            html.Div(
                                                "CropIntel Assistant",
                                                className="chat-name"
                                            ),

                                            html.Div(
                                                "● Online",
                                                className="chat-status"
                                            )
                                        ])

                                    ]
                                ),

                                html.Div(
                                    id="chatbot-output",
                                    className="chat-body",
                                    children=[

                                        html.Div(
                                            (
                                                "Hello! I'm your agriculture "
                                                "assistant. Ask me about crops, "
                                                "rainfall, irrigation, soil, "
                                                "fertilizer, weather or "
                                                "crop yield."
                                            ),
                                            className="bot-message"
                                        )

                                    ]
                                ),

                                html.Div(
                                    className="chat-input-area",
                                    children=[

                                        dcc.Textarea(
                                            id="chatbot-input",
                                            placeholder=(
                                                "Ask about agriculture..."
                                            ),
                                            className="chat-input"
                                        ),

                                        html.Button(
                                            "➤",
                                            id="chatbot-button",
                                            n_clicks=0,
                                            className="send-button"
                                        )

                                    ]
                                )

                            ]
                        )

                    ]
                )

            ]
        )

    ])


def analytics_page():

    return html.Div([

        hero_section(
            "MODEL PERFORMANCE",
            "Model Analytics",
            (
                "Overview of the machine learning techniques "
                "used throughout the Crop Intelligence System."
            ),
            "Models",
            "3"
        ),

        html.Div(
            className="page-content",
            children=[

                html.H2(
                    "Machine Learning Models",
                    className="section-title"
                ),

                html.P(
                    (
                        "Performance and architecture of the "
                        "models integrated into the application."
                    ),
                    className="section-description"
                ),

                html.Div(
                    className="analytics-grid",
                    children=[

                        html.Div(
                            className="model-card",
                            children=[

                                html.Div(
                                    "📈 Crop Yield Model",
                                    className="model-name"
                                ),

                                html.Div(
                                    "LINEAR REGRESSION",
                                    className="model-type"
                                ),

                                html.Div(
                                    className="metric-row",
                                    children=[
                                        html.Span(
                                            "R² Score",
                                            className="metric-name"
                                        ),
                                        html.Span(
                                            "0.9132",
                                            className="metric-value"
                                        )
                                    ]
                                ),

                                html.Div(
                                    className="metric-row",
                                    children=[
                                        html.Span(
                                            "MAE",
                                            className="metric-name"
                                        ),
                                        html.Span(
                                            "0.3983",
                                            className="metric-value"
                                        )
                                    ]
                                ),

                                html.Div(
                                    className="metric-row",
                                    children=[
                                        html.Span(
                                            "MSE",
                                            className="metric-name"
                                        ),
                                        html.Span(
                                            "0.2493",
                                            className="metric-value"
                                        )
                                    ]
                                ),

                                html.Div(
                                    className="metric-row",
                                    children=[
                                        html.Span(
                                            "RMSE",
                                            className="metric-name"
                                        ),
                                        html.Span(
                                            "0.4993",
                                            className="metric-value"
                                        )
                                    ]
                                )

                            ]
                        ),

                        html.Div(
                            className="model-card",
                            children=[

                                html.Div(
                                    "🍃 Maize Disease Model",
                                    className="model-name"
                                ),

                                html.Div(
                                    "HOG + SUPPORT VECTOR MACHINE",
                                    className="model-type"
                                ),

                                html.Div(
                                    className="metric-row",
                                    children=[
                                        html.Span(
                                            "Test Accuracy",
                                            className="metric-name"
                                        ),
                                        html.Span(
                                            "86.41%",
                                            className="metric-value"
                                        )
                                    ]
                                ),

                                html.Div(
                                    className="metric-row",
                                    children=[
                                        html.Span(
                                            "Image Size",
                                            className="metric-name"
                                        ),
                                        html.Span(
                                            "128 × 128",
                                            className="metric-value"
                                        )
                                    ]
                                ),

                                html.Div(
                                    className="metric-row",
                                    children=[
                                        html.Span(
                                            "HOG Features",
                                            className="metric-name"
                                        ),
                                        html.Span(
                                            "8100",
                                            className="metric-value"
                                        )
                                    ]
                                ),

                                html.Div(
                                    className="metric-row",
                                    children=[
                                        html.Span(
                                            "Classes",
                                            className="metric-name"
                                        ),
                                        html.Span(
                                            "4",
                                            className="metric-value"
                                        )
                                    ]
                                )

                            ]
                        ),

                        html.Div(
                            className="model-card",
                            children=[

                                html.Div(
                                    "💬 Agriculture Chatbot",
                                    className="model-name"
                                ),

                                html.Div(
                                    "TF-IDF + MULTINOMIAL NAIVE BAYES",
                                    className="model-type"
                                ),

                                html.Div(
                                    className="metric-row",
                                    children=[
                                        html.Span(
                                            "Task",
                                            className="metric-name"
                                        ),
                                        html.Span(
                                            "Intent Classification",
                                            className="metric-value"
                                        )
                                    ]
                                ),

                                html.Div(
                                    className="metric-row",
                                    children=[
                                        html.Span(
                                            "Text Features",
                                            className="metric-name"
                                        ),
                                        html.Span(
                                            "TF-IDF",
                                            className="metric-value"
                                        )
                                    ]
                                ),

                                html.Div(
                                    className="metric-row",
                                    children=[
                                        html.Span(
                                            "Classifier",
                                            className="metric-name"
                                        ),
                                        html.Span(
                                            "Naive Bayes",
                                            className="metric-value"
                                        )
                                    ]
                                ),

                                html.Div(
                                    className="metric-row",
                                    children=[
                                        html.Span(
                                            "Answer Selection",
                                            className="metric-name"
                                        ),
                                        html.Span(
                                            "Cosine Similarity",
                                            className="metric-value"
                                        )
                                    ]
                                )

                            ]
                        )

                    ]
                )

            ]
        )

    ])


app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True
)

app.title = "CropIntel"


app.layout = html.Div(
    className="app-container",
    children=[

        dcc.Store(
            id="current-page",
            data="dashboard"
        ),

        html.Div(
            className="sidebar",
            children=[

                html.Div(
                    className="brand",
                    children=[

                        html.Div(
                            "🌿",
                            className="brand-icon"
                        ),

                        html.Div([
                            html.H2(
                                "CropIntel",
                                className="brand-name"
                            ),

                            html.P(
                                "AI-Driven Crop Intelligence",
                                className="brand-subtitle"
                            )
                        ])

                    ]
                ),

                html.Div(
                    className="nav-menu",
                    children=[

                        html.Button(
                            [
                                html.Span("▦"),
                                html.Span(
                                    "Dashboard",
                                    className="nav-text"
                                )
                            ],
                            id="nav-dashboard",
                            n_clicks=0,
                            className="nav-button"
                        ),

                        html.Button(
                            [
                                html.Span("📈"),
                                html.Span(
                                    "Yield Prediction",
                                    className="nav-text"
                                )
                            ],
                            id="nav-yield",
                            n_clicks=0,
                            className="nav-button"
                        ),

                        html.Button(
                            [
                                html.Span("🍃"),
                                html.Span(
                                    "Disease Detection",
                                    className="nav-text"
                                )
                            ],
                            id="nav-disease",
                            n_clicks=0,
                            className="nav-button"
                        ),

                        html.Button(
                            [
                                html.Span("💬"),
                                html.Span(
                                    "Assistant",
                                    className="nav-text"
                                )
                            ],
                            id="nav-chatbot",
                            n_clicks=0,
                            className="nav-button"
                        ),

                        html.Button(
                            [
                                html.Span("📊"),
                                html.Span(
                                    "Model Analytics",
                                    className="nav-text"
                                )
                            ],
                            id="nav-analytics",
                            n_clicks=0,
                            className="nav-button"
                        )

                    ]
                ),

                html.Div(
                    className="sidebar-footer",
                    children=[

                        html.Div(
                            "🌱 Crop Intelligence System",
                            className="sidebar-footer-title"
                        ),

                        html.Div(
                            (
                                "Regression • Computer Vision "
                                "• Natural Language Processing"
                            ),
                            className="sidebar-footer-text"
                        )

                    ]
                )

            ]
        ),

        html.Div(
            className="main-content",
            children=[

                html.Div(
                    id="page-content"
                )

            ]
        )

    ]
)


@app.callback(
    Output("current-page", "data"),
    [
        Input("nav-dashboard", "n_clicks"),
        Input("nav-yield", "n_clicks"),
        Input("nav-disease", "n_clicks"),
        Input("nav-chatbot", "n_clicks"),
        Input("nav-analytics", "n_clicks")
    ],
    prevent_initial_call=True
)
def change_page(
    dashboard_clicks,
    yield_clicks,
    disease_clicks,
    chatbot_clicks,
    analytics_clicks
):

    triggered = dash.ctx.triggered_id

    if triggered == "nav-yield":
        return "yield"

    if triggered == "nav-disease":
        return "disease"

    if triggered == "nav-chatbot":
        return "chatbot"

    if triggered == "nav-analytics":
        return "analytics"

    return "dashboard"


@app.callback(
    Output("page-content", "children"),
    Input("current-page", "data")
)
def render_page(page):

    if page == "yield":
        return yield_page()

    if page == "disease":
        return disease_page()

    if page == "chatbot":
        return chatbot_page()

    if page == "analytics":
        return analytics_page()

    return dashboard_page()


@app.callback(
    [
        Output("nav-dashboard", "className"),
        Output("nav-yield", "className"),
        Output("nav-disease", "className"),
        Output("nav-chatbot", "className"),
        Output("nav-analytics", "className")
    ],
    Input("current-page", "data")
)
def update_navigation(page):

    base = "nav-button"
    active = "nav-button active"

    return [
        active if page == "dashboard" else base,
        active if page == "yield" else base,
        active if page == "disease" else base,
        active if page == "chatbot" else base,
        active if page == "analytics" else base
    ]


@app.callback(
    Output(
        "yield-output",
        "children"
    ),
    Input(
        "predict-yield-button",
        "n_clicks"
    ),
    State("region", "value"),
    State("soil", "value"),
    State("crop", "value"),
    State("rainfall", "value"),
    State("temperature", "value"),
    State("fertilizer", "value"),
    State("irrigation", "value"),
    State("weather", "value"),
    State("harvest-days", "value"),
    prevent_initial_call=True
)
def predict_yield(
    n_clicks,
    region,
    soil,
    crop,
    rainfall,
    temperature,
    fertilizer,
    irrigation,
    weather,
    harvest_days
):

    values = [
        region,
        soil,
        crop,
        rainfall,
        temperature,
        fertilizer,
        irrigation,
        weather,
        harvest_days
    ]

    if any(
        value is None
        for value in values
    ):
        return html.Div(
            "⚠️ Please fill in all fields.",
            style={
                "color": "#c84b4b",
                "marginTop": "20px",
                "fontWeight": "600"
            }
        )

    input_data = pd.DataFrame({
        "Region": [region],
        "Soil_Type": [soil],
        "Crop": [crop],
        "Rainfall_mm": [rainfall],
        "Temperature_Celsius": [temperature],
        "Fertilizer_Used": [fertilizer],
        "Irrigation_Used": [irrigation],
        "Weather_Condition": [weather],
        "Days_to_Harvest": [harvest_days]
    })

    prediction = crop_yield_model.predict(
        input_data
    )[0]

    prediction = max(
        0,
        prediction
    )

    return html.Div(
        className="result-card",
        children=[

            html.Div(
                "Predicted Crop Yield",
                className="result-label"
            ),

            html.Div(
                f"{prediction:.2f}",
                className="result-value"
            ),

            html.Div(
                "tons / hectare",
                className="result-unit"
            )

        ]
    )


@app.callback(
    [
        Output(
            "image-preview",
            "children"
        ),
        Output(
            "disease-output",
            "children"
        )
    ],
    Input(
        "upload-image",
        "contents"
    ),
    prevent_initial_call=True
)
def process_uploaded_image(contents):

    if contents is None:
        return None, None

    result = classify_image(
        contents
    )

    preview = html.Img(
        src=contents,
        className="preview-image"
    )

    if result is None:

        output = html.Div(
            "⚠️ Unable to process this image.",
            style={
                "color": "#c84b4b",
                "fontWeight": "600"
            }
        )

        return preview, output

    disease, confidence = result

    output = html.Div(
        className="result-card",
        children=[

            html.Div(
                "Classification Result",
                className="result-label"
            ),

            html.Div(
                disease,
                className="result-value",
                style={
                    "fontSize": "25px"
                }
            ),

            html.Div(
                f"Model confidence: {confidence:.2f}%",
                className="result-unit"
            )

        ]
    )

    return preview, output


@app.callback(
    Output(
        "chatbot-output",
        "children"
    ),
    Input(
        "chatbot-button",
        "n_clicks"
    ),
    State(
        "chatbot-input",
        "value"
    ),
    prevent_initial_call=True
)
def chatbot_response(
    n_clicks,
    question
):

    if question is None or question.strip() == "":

        return html.Div(
            (
                "Please enter an agriculture-related "
                "question below."
            ),
            className="bot-message"
        )

    response = get_chatbot_response(
        question
    )

    return [

        html.Div(
            (
                "Hello! I'm your agriculture assistant. "
                "Ask me about crops, rainfall, irrigation, "
                "soil, fertilizer, weather or crop yield."
            ),
            className="bot-message"
        ),

        html.Div(
            question,
            className="user-message"
        ),

        html.Div(
            response,
            className="bot-message"
        )

    ]


if __name__ == "__main__":
    app.run(debug=True)
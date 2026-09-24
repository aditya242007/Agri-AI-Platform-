# 📚 Project Submission Structure

This repository represents the combination of two Machine Learning projects into one integrated application.

The original project requirements consisted of:

### Project 1 — Image Classification

A maize leaf disease classification system was developed using:

- Image preprocessing with OpenCV
- HOG (Histogram of Oriented Gradients) feature extraction
- Support Vector Machine (SVM) classification

The classifier identifies:

- Cercospora Leaf Spot
- Common Rust
- Healthy
- Northern Leaf Blight

### Project 2 — NLP Chatbot

An agriculture-focused chatbot was developed using Natural Language Processing and Machine Learning.

The chatbot uses:

- Text preprocessing
- TF-IDF vectorization
- Multinomial Naive Bayes for intent classification
- Cosine similarity for retrieving relevant responses

It answers questions related to agriculture, crop yield, irrigation, rainfall, soil, fertilizer, weather, and other topics included in its knowledge base.

## Combined Project

Instead of keeping these two projects as separate applications, they were integrated into a single system called **CropIntel-AI**.

A third Machine Learning component — **Crop Yield Prediction using Linear Regression** — was also developed and integrated into the same application.

Therefore, the final system consists of:

| Original Component | Type | Technique |
|---|---|---|
| Project 1 | Image Classification | HOG + SVM |
| Project 2 | NLP Chatbot | TF-IDF + Naive Bayes + Cosine Similarity |
| Additional Component | Crop Yield Prediction | Linear Regression |

All components are connected through a single Dash-based web interface.

This approach demonstrates how Computer Vision, Natural Language Processing, and Regression can be integrated into one practical Machine Learning application.

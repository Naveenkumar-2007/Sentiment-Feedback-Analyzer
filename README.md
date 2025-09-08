# 📝 Review Sentiment Analysis API (FastAPI)

This project is a **sentiment analysis API** built with **FastAPI**.  
It predicts whether a given text review is **Positive 😀** or **Negative 😞** based on a machine learning model trained on Amazon product reviews.

---

## 📌 Features
- Preprocesses raw review text (cleaning, stopword removal, lemmatization).
- Uses a saved **CountVectorizer** and **Naive Bayes model** (`vector.pkl`, `model.pkl`).
- Provides a **REST API** endpoint `/predict` for predictions.
- Returns both **binary prediction (0/1)** and **human-readable sentiment**.

---

## 🛠️ Tech Stack
- **Python 3.12**
- **FastAPI** (Backend Framework)
- **Scikit-learn** (Machine Learning)
- **NLTK** (Stopwords & Lemmatization)
- **BeautifulSoup** (HTML cleaning)
- **Pickle** (Model persistence)
- **Uvicorn** (Server)

---


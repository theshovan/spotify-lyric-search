# 🎵 Spotify Lyric Search Engine

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Similarity--Model-green.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-Spotify--50k-orange.svg)]()

> **Project 5: Technical Assessment** > A high-performance text-identification algorithm designed to retrieve Song Titles and Artists from short lyric snippets using the Spotify 50k+ Songs Dataset.

---

## 📖 Overview
This project addresses the challenge of searching through vast musical libraries using natural language. By implementing **NLP (Natural Language Processing)** techniques, the model can "listen" to a few words of a song and accurately identify its metadata (Title & Artist) from a database of over 57,000 tracks.

## ✨ Key Features
* **Smart Preprocessing:** Custom pipeline for tokenization, lowercase normalization, and regex-based noise removal.
* **Stop-word Filtering:** Automatically ignores common English words to focus on the unique "fingerprint" of the lyrics.
* **Vectorized Search:** Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into mathematical vectors.
* **Similarity Engine:** Employs **Cosine Similarity** to calculate the distance between a user's snippet and the global dataset.
* **Accuracy Demo:** Includes an automated evaluation script to test prediction reliability.

## 🛠️ Tech Stack
* **Language:** Python
* **ML Libraries:** Scikit-Learn (TfidfVectorizer, Cosine Similarity)
* **Data Handling:** Pandas, NumPy
* **Notebook:** Jupyter Notebook for interactive demonstration

### 🔗 [Download Dataset](https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs)



## 🚀 The Process
1. **Data Preprocessing:** Standardized raw lyrics through tokenization and noise reduction (regex filtering).
2. **Feature Engineering:** Converted text into numerical significance using **TF-IDF Vectorization**, effectively weighting unique words over common ones.
3. **Similarity Modeling:** Applied **Cosine Similarity** metrics to perform high-speed identification across the entire dataset.

## 📊 Model Performance
The engine demonstrates high precision in identifying tracks even from generic snippets. 
- **Input:** "She holds me and squeezes my hand"
- **Output:** `Ahe's My Kind Of Girl` by `ABBA`

---

## 🚀 Getting Started


### 1. Clone the Repository
```bash
git clone [https://github.com/theshovan/spotify-lyric-search.git](https://github.com/theshovan/spotify-lyric-search.git)
cd spotify-lyric-search
```

## 🛠 Setup
1. `pip install -r requirements.txt`
2. Run `python scripts/model.py` to start the engine.

---

**Developed by:** [Shovan Bera](https://github.com/theshovan)

# Magazine Subscription Recommendation System (Hybrid Recommender)

A hybrid recommendation system for magazine subscriptions combining:
- **Collaborative Filtering** using **Truncated SVD** on a user–item rating matrix
- **Content-Based Filtering** using **TF-IDF** vectorization over magazine descriptions + **cosine similarity**
- A **Streamlit** UI for generating recommendations and collecting user feedback (ratings)

This project demonstrates an end-to-end applied ML workflow: data ingestion, cleaning, feature engineering, modeling, recommendation generation, and simple evaluation.

---

## Features

### 1) Data Processing (CSV + JSONL)
- Loads user interactions (ratings) from a CSV dataset
- Loads magazine metadata (titles, descriptions) from a JSONL file
- Cleans data, removes duplicates, and handles missing values
- Converts list-style descriptions into a text field for NLP processing

### 2) Content-Based Recommendations (TF-IDF)
- Builds a TF-IDF matrix from magazine description text
- Computes cosine similarity to recommend magazines similar to a selected item

### 3) Collaborative Filtering (SVD)
- Builds a **user × item** interaction matrix from ratings
- Applies **TruncatedSVD** to learn latent factors
- Generates Top-N recommendations for a selected user

### 4) Hybrid Recommendations
- Combines collaborative and content-based recommendations
- Outputs a merged Top-N list

### 5) Streamlit Application
- Select a user and a magazine to generate recommendations
- Displays **Collaborative**, **Content-Based**, and **Hybrid** recommendations
- Collects user feedback via rating sliders and saves rating history to CSV

### 6) Basic Evaluation
- Includes simple Precision/Recall/F1 scoring logic (baseline-style evaluation)
- Intended as a starting point for more rigorous recommender evaluation

---

## Tech Stack
- **Python**
- **pandas**, **NumPy**
- **scikit-learn** (TF-IDF, cosine similarity, train/test split, TruncatedSVD, metrics)
- **SciPy** (sparse matrices)
- **Streamlit**

---

## Project Structure (Typical)
> Filenames may vary based on your local setup.

- `Magazine_Subscriptions_Mini.csv` — user ratings / interactions  
- `meta_Magazine_Subscriptions.jsonl` — magazine metadata (titles, descriptions, ASINs)  
- `user_history.csv` — saved user ratings from the Streamlit UI  
- `app.py` (or similar) — main script with preprocessing + modeling + Streamlit UI  

---

## How It Works (High-Level)

### Content-Based
1. Convert descriptions to text
2. TF-IDF vectorize
3. Cosine similarity between items
4. Recommend items most similar to the selected magazine

### Collaborative Filtering (SVD)
1. Create user–item rating matrix
2. Apply TruncatedSVD to learn latent representations
3. Recommend items from learned latent factors

### Hybrid
- Union of the two recommendation lists and return Top-N

---

## Setup & Run

### 1) Create environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

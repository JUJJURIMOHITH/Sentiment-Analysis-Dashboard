# Sentiment Analysis Dashboard

Classify movie/product reviews as **positive**, **negative**, or **neutral** using an LSTM trained on the IMDB dataset. Flask API + dashboard with confidence score visualization.

## How to run

### 1. Create a virtual environment (recommended)

```bash
cd C:\Users\mohit\Desktop\TT2\TTT\ml_model
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (first time only)

This downloads the IMDB dataset and trains the LSTM. Takes a few minutes.

```bash
python train.py
```

### 4. Start the Flask app

```bash
python app.py
```

### 5. Open the dashboard

- In your browser go to: **http://127.0.0.1:5000**
- Paste a review and click **Analyze sentiment** to see the label and confidence bars.

## API

- **POST** `/api/predict`  
  Body: `{ "text": "Your review here..." }`  
  Response: `{ "sentiment": "positive"|"negative"|"neutral", "confidence": 0.92, "scores": { "positive": 0.92, "negative": 0.05, "neutral": 0.03 } }`

## Project layout

- `train.py` – Train LSTM on IMDB; saves model and config under `model/`
- `inference.py` – Load model and run prediction from text
- `app.py` – Flask server (dashboard + `/api/predict`)
- `templates/dashboard.html` – Dashboard UI

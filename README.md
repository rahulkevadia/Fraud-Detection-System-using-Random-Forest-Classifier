# Fraud Detection System

A machine learning-based web application for detecting fraudulent bank transactions.

## Features

- Real-time fraud detection using Random Forest model
- User-friendly web interface
- Client-side validation
- Automatic balance calculations
- Probability scores for predictions

## Tech Stack

- Python 3.7
- Flask
- Scikit-learn
- Pandas
- NumPy
- Bootstrap 5
- JavaScript

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd fraud-detection-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

## Model Information

- Algorithm: Random Forest Classifier
- Accuracy: 98.25%
- ROC-AUC Score: 94.58%
- Features: Transaction type, amount, account balances, and account information

## Deployment

This application is configured for deployment on Render. The deployment configuration can be found in `render.yaml`.

## Project Structure

```
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── gunicorn.conf.py      # Gunicorn configuration
├── render.yaml           # Render deployment configuration
├── static/
│   ├── css/             # CSS styles
│   └── js/              # JavaScript files
├── templates/
│   └── home.html        # HTML template
└── model/
    ├── fraud_detection_model.pkl
    ├── scaler.pkl
    └── label_encoders.pkl
```

## License

MIT License 
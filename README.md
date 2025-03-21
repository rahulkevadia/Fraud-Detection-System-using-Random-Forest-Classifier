# Fraud Detection System using Random Forest Classifier

This project implements a fraud detection system using Random Forest Classifier to identify fraudulent transactions in financial data.

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

## Dataset

The dataset used in this project is too large to be included in the repository. To use this project:

1. Create a `Dataset` directory in the project root if it doesn't exist
2. Download the "Transactions Dataset.csv" file
3. Place the downloaded file in the `Dataset` directory

## Project Structure

- `Dataset/`: Directory for storing the transaction dataset (not included in repository due to size)
- Other project files and directories containing the implementation

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/YOUR-USERNAME/Fraud-Detection-System-using-Random-Forest-Classifier.git
cd Fraud-Detection-System-using-Random-Forest-Classifier
```

2. Set up the dataset as described in the Dataset section above

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Run the project according to the implementation instructions in the code

## Note

The dataset file is not included in this repository due to its large size (470.67 MB). Please ensure you have the dataset file properly placed in the Dataset directory before running the code.

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
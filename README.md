#  Real Estate Price Predictor

This project is a **machine learning web application** built with **Streamlit** that predicts real estate prices based on user-inputted property features. The model uses linear regression and provides insight into which features most influence price.

---

##  Features
- Predict housing prices using features like year sold, size, beds, baths, and lot size.
- Modularized codebase (training, saving, preprocessing, logging, and app UI).
- Streamlit web interface for user input and predictions.
- Automatically generates feature importance chart.
- Logging and error handling throughout.

---

##  Model Info
- Algorithm: `LinearRegression`
- Features used: `year_sold`, `sqft`, `beds`, `baths`, `lot_size`
- Evaluation: MSE & R² score logged
- Visual: `feature_importance.png`

---

##  How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run everything (train + launch UI)
```bash
python main.py
```

This will:
- Load and preprocess data
- Train and save the model
- Generate a feature importance chart
- Launch the Streamlit web app

Or run the app only:
```bash
streamlit run app/streamlit_app.py
```

---

##  Folder Structure
```
real_estate_app/
├── app/
│   └── streamlit_app.py        # Web UI for predictions
├── data/
│   └── real_estate.csv         # Dataset
├── data_prep/
│   └── data_loader.py          # Preprocessing functions
├── model/
│   ├── train_model.py          # Training + chart
│   ├── evaluate_model.py       # (Optional evaluation)
│   └── save_model.py           # Save model as .pickle
├── utils/
│   └── logger.py               # Logging utility
├── main.py                     # Entry point for full pipeline
├── real_estate_model.pickle    # Saved ML model
├── feature_importance.png      # Feature chart
├── requirements.txt
└── README.md
```

---

##  Author
- Project for `CST2216 Machine Learning`
- Student: Junjin Chen
- Year: 2025 Winter

---

# Y-Knot Fraud Detective

End-to-end credit card fraud detection system built with XGBoost and an interactive Streamlit dashboard.

## Project Overview
This project detects fraudulent credit card transactions using machine learning. It effectively handles severe class imbalance (only 0.17% fraud cases) and provides a user-friendly web interface for batch predictions.

## Key Features
- Exploratory Data Analysis with visualizations
- XGBoost classifier with proper imbalance handling using `scale_pos_weight`
- Interactive Streamlit web application for batch prediction
- Fraud probability scores and downloadable CSV results

## Tech Stack
- Python 3
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib, Seaborn

## Dataset
Credit Card Fraud Detection dataset from Kaggle  
(284,807 transactions with 492 fraud cases)

[Download Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/y-knot-fraud-detector.git
   cd y-knot-fraud-detector
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the dataset:
Download creditcard.csv from Kaggle
Place it inside the data/ folder

4. Train the model:
    ```bash
   python train_model.py
   ```

5. Run the web app:
   ```bash
   streamlit run streamlit_app.py
   ```


Project Structure 
```text
y-knot-fraud-detector/
├── data/                    # creditcard.csv
├── models/                  # fraud_model.pkl and scaler.pkl
├── images/                  # class_distribution.png
├── train_model.py           # Model training + EDA
├── streamlit_app.py         # Interactive Streamlit dashboard
├── requirements.txt
└── README.md
```

Results

Successfully handles highly imbalanced dataset
Provides clear fraud probability for each transaction
Ready for batch processing of new transaction files




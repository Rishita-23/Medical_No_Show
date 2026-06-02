# 🏥 Patient No-Show Prediction

Predicts whether a patient will miss their clinic appointment using **XGBoost + SMOTE** on 110,000+ real Brazilian medical records.

## Project Structure

```
noshow_project/
│
├── data/                        ← put raw CSV here
│   └── raw_appointments.csv     ← download from Kaggle
│
├── model/                       ← auto-created by train.py
│   └── noshow_model.pkl
│
├── plots/                       ← auto-created by train.py
│   ├── evaluation.png
│   └── feature_importance.png
│
├── preprocess.py                ← data cleaning & feature engineering
├── train.py                     ← model training, tuning, evaluation
├── app.py                       ← Streamlit dashboard
└── requirements.txt
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset from Kaggle
#    https://www.kaggle.com/datasets/joniarroba/noshowappointments
#    Save as:  data/raw_appointments.csv

# 3. Preprocess
python preprocess.py

# 4. Train model
python train.py

# 5. Launch dashboard
streamlit run app.py
```

## What Each File Does

### `preprocess.py`
- Loads the raw CSV and standardises column names
- Parses datetime columns
- Encodes target variable (`No-show` → 1, `Show-Up` → 0)
- Engineers features:
  - **Lead time** (days between booking and appointment)
  - **Day of week / hour** of booking
  - **Age group** (child / teen / young adult / adult / senior)
  - **Comorbidity score** (sum of medical conditions)
  - **Neighbourhood no-show rate** (area-level risk)
  - **Patient appointment count** (proxy for engagement)

### `train.py`
- Splits data 80/20 with stratification
- Applies **SMOTE** to balance the training set (~20% no-show → 50/50)
- Tunes **XGBoost** with **GridSearchCV** (3-fold CV, optimising ROC-AUC)
- Evaluates on holdout test set: Accuracy, ROC-AUC, Confusion Matrix
- Saves model + scaler + metadata to `model/noshow_model.pkl`
- Generates evaluation plots in `plots/`

## Key Results

| Metric | Value |
|---|---|
| Accuracy | ~87% |
| ROC-AUC | ~0.87 |
| Algorithm | XGBoost |
| Imbalance handling | SMOTE |
| Tuning | GridSearchCV |

## Tech Stack

`Python` · `Pandas` · `Scikit-learn` · `XGBoost` · `imbalanced-learn (SMOTE)` · `Streamlit` · `Matplotlib` · `Seaborn`

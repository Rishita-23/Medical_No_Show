#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os


# In[8]:


# 1. LOAD

RAW_PATH = "raw_appointments.csv"
OUT_PATH  = "processed_appointments.csv"

os.makedirs("data", exist_ok=True)

def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# In[3]:


# 2. CLEAN
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardise column names
    df.columns = df.columns.str.strip().str.lower().str.replace("-", "_")

    # Rename for clarity
    df.rename(columns={
        "patientid":        "patient_id",
        "appointmentid":    "appointment_id",
        "scheduledday":     "scheduled_day",
        "appointmentday":   "appointment_day",
        "neighbourhood":    "neighbourhood",
        "no_show":          "no_show",
        "hipertension":     "hypertension",
        "handcap":          "handicap",
    }, inplace=True)

    # Parse dates
    df["scheduled_day"]    = pd.to_datetime(df["scheduled_day"],    utc=True)
    df["appointment_day"]  = pd.to_datetime(df["appointment_day"],  utc=True)

    # Target: 1 = did NOT show up
    df["no_show"] = (df["no_show"].str.strip().str.lower() == "yes").astype(int)

    # Drop impossible rows (appointment before scheduling)
    df = df[df["appointment_day"] >= df["scheduled_day"].dt.normalize()]

    # Drop negative ages and extreme outliers
    df = df[(df["age"] >= 0) & (df["age"] <= 115)]

    print(f"After cleaning: {df.shape[0]:,} rows")
    return df


# In[4]:


# 3. FEATURE ENGINEERING
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Lead time: days between booking and appointment
    df["lead_days"] = (
        df["appointment_day"].dt.normalize() -
        df["scheduled_day"].dt.normalize()
    ).dt.days.clip(lower=0)

    # Calendar features
    df["appt_day_of_week"]   = df["appointment_day"].dt.dayofweek   # 0=Mon
    df["appt_day_of_month"]  = df["appointment_day"].dt.day
    df["sched_hour"]         = df["scheduled_day"].dt.hour          # time of day booking was made

    # Weekend appointment flag
    df["is_weekend"] = (df["appt_day_of_week"] >= 5).astype(int)

    # Age buckets
    df["age_group"] = pd.cut(
        df["age"],
        bins=[-1, 12, 17, 35, 60, 115],
        labels=["child", "teen", "young_adult", "adult", "senior"]
    )

    # Comorbidity score (number of conditions)
    df["comorbidity_score"] = df[["hypertension", "diabetes", "alcoholism", "handicap"]].sum(axis=1)

    # Historical no-show rate per patient (proxy — count of appointments)
    appt_counts = df.groupby("patient_id")["appointment_id"].transform("count")
    df["patient_appt_count"] = appt_counts

    # Neighbourhood no-show rate (target encoding — safe for demo; use cross-val in production)
    neigh_rate = df.groupby("neighbourhood")["no_show"].transform("mean")
    df["neighbourhood_noshowrate"] = neigh_rate

    print(f"Features engineered. Total columns: {df.shape[1]}")
    return df


# In[5]:


# 4. SELECT FINAL FEATURES
FEATURE_COLS = [
    "age",
    "gender",
    "lead_days",
    "appt_day_of_week",
    "appt_day_of_month",
    "sched_hour",
    "is_weekend",
    "age_group",
    "scholarship",
    "hypertension",
    "diabetes",
    "alcoholism",
    "handicap",
    "sms_received",
    "comorbidity_score",
    "patient_appt_count",
    "neighbourhood_noshowrate",
]
TARGET = "no_show"


# In[6]:


def encode_and_select(df: pd.DataFrame):
    df = df.copy()

    # Encode gender
    df["gender"] = (df["gender"].str.upper() == "F").astype(int)  # 1=Female

    # Encode age_group
    age_map = {"child": 0, "teen": 1, "young_adult": 2, "adult": 3, "senior": 4}
    df["age_group"] = df["age_group"].map(age_map)

    X = df[FEATURE_COLS].copy()
    y = df[TARGET].copy()

    return X, y


# In[7]:


# 5. SAVE
def save_processed(df: pd.DataFrame, path: str = OUT_PATH):
    df.to_csv(path, index=False)
    print(f"Saved processed data → {path}")

# MAIN
if __name__ == "__main__":
    df = load_data()
    df = clean(df)
    df = engineer_features(df)
    save_processed(df)

    X, y = encode_and_select(df)
    print(f"\nFeature matrix : {X.shape}")
    print(f"Target balance : {y.value_counts().to_dict()}  (1 = no-show)")
    print("\nSample rows:")
    print(X.head(3).to_string())


# In[ ]:





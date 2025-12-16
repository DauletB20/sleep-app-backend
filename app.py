# ============================================================
# app.py — Sleep Prediction Backend (with plots + LLM)
# ============================================================

import os
import json
import uuid
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fastapi.staticfiles import StaticFiles


from fastapi import FastAPI, UploadFile, File, HTTPException
from openai import OpenAI


# ------------------------------------------------------------
# Init
# ------------------------------------------------------------
app = FastAPI(title="Sleep Quality Prediction API")

app.mount("/plots", StaticFiles(directory="plots"), name="plots")

from fastapi.middleware.cors import CORSMiddleware

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


os.makedirs("plots", exist_ok=True)

reg_model = joblib.load("sleep_regression_pipeline.joblib")
cls_model = joblib.load("sleep_classification_pipeline.joblib")

with open("sleep_feature_columns.json") as f:
    FEATURE_COLS = json.load(f)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def save_plot(fig_name, plot_func, run_id):
    filename = f"{fig_name}_{run_id}.png"
    path = f"plots/{filename}"
    plt.figure()
    plot_func()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return filename

def assign_night_date(df):
    df["ts_start"] = pd.to_datetime(df["ts_start"], unit="ms")
    df["ts_end"] = pd.to_datetime(df["ts_end"], unit="ms")

    df["night_date"] = df["ts_start"].dt.date
    df.loc[df["ts_start"].dt.hour < 12, "night_date"] = (
        df["ts_start"] - pd.Timedelta(days=1)
    ).dt.date
    return df


def low_motion_ratio(x, thr=0.05):
    return np.mean(np.abs(np.asarray(x)) < thr)


def aggregate_night(df):
    agg = {
        "HR": ["mean", "std", "min", "max"],
        "rmssd": ["mean", "std"],
        "sdnn": ["mean", "std"],
        "lf/hf": ["mean"],
        "acc_x_avg": ["std", low_motion_ratio],
        "acc_y_avg": ["std", low_motion_ratio],
        "acc_z_avg": ["std", low_motion_ratio],
        "gyr_x_avg": ["std"],
        "gyr_y_avg": ["std"],
        "gyr_z_avg": ["std"],
        "steps": ["sum"],
        "calories": ["sum"],
        "missingness_score": ["mean"],
    }

    night = df.groupby(["deviceId", "night_date"]).agg(agg)
    night.columns = [
        f"{c[0]}_{c[1] if isinstance(c[1], str) else c[1].__name__}"
        for c in night.columns
    ]
    return night.reset_index()


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def save_plot(fig_name, plot_func, run_id):
    filename = f"{fig_name}_{run_id}.png"
    path = f"plots/{filename}"

    plt.figure()
    plot_func()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return filename



def generate_plots(df, run_id):
    urls = {}

    df["acc_magnitude"] = np.sqrt(
        df["acc_x_avg"]**2 + df["acc_y_avg"]**2 + df["acc_z_avg"]**2
    )
    df["gyro_energy"] = np.sqrt(
        df["gyr_x_avg"]**2 + df["gyr_y_avg"]**2 + df["gyr_z_avg"]**2
    )

    urls["heart_rate"] = save_plot(
        "heart_rate",
        lambda: (
            plt.plot(df["ts_start"], df["HR"], color="crimson"),
            plt.grid(True, alpha=0.3)
        ),
        run_id
    )

    urls["rmssd"] = save_plot(
        "rmssd",
        lambda: (
            plt.plot(df["ts_start"], df["rmssd"], color="darkgreen"),
            plt.grid(True, alpha=0.3)
        ),
        run_id
    )

    urls["lf_hf"] = save_plot(
        "lf_hf",
        lambda: (
            plt.plot(df["ts_start"], df["lf/hf"], color="purple"),
            plt.grid(True, alpha=0.3)
        ),
        run_id
    )

    urls["accelerometer"] = save_plot(
        "accelerometer",
        lambda: (
            plt.plot(df["ts_start"], df["acc_x_avg"], label="X", color="steelblue"),
            plt.plot(df["ts_start"], df["acc_y_avg"], label="Y", color="orange"),
            plt.plot(df["ts_start"], df["acc_z_avg"], label="Z", color="green"),
            plt.legend(),
            plt.grid(True, alpha=0.3)
        ),
        run_id
    )

    urls["movement_intensity"] = save_plot(
        "movement_intensity",
        lambda: (
            plt.plot(df["ts_start"], df["acc_magnitude"], color="teal"),
            plt.grid(True, alpha=0.3)
        ),
        run_id
    )

    urls["body_rotation"] = save_plot(
        "body_rotation",
        lambda: (
            plt.plot(df["ts_start"], df["gyro_energy"], color="darkorange"),
            plt.grid(True, alpha=0.3)
        ),
        run_id
    )

    urls["calories"] = save_plot(
        "calories",
        lambda: (
            plt.plot(df["ts_start"], df["calories"], color="brown"),
            plt.grid(True, alpha=0.3)
        ),
        run_id
    )

    return {k: f"/plots/{v}" for k, v in urls.items()}




# ------------------------------------------------------------
# LLM
# ------------------------------------------------------------
def build_prompt(features, preds):
    return f"""
You are a sleep health assistant.

INPUT DATA (smartwatch, one night):
- Average heart rate: {features['HR_mean']:.1f} bpm
- HRV RMSSD: {features['rmssd_mean']:.1f}
- HRV SDNN: {features['sdnn_mean']:.1f}
- LF/HF ratio: {features['lf/hf_mean']:.2f}
- Motion variability:
  acc_x_std={features['acc_x_avg_std']:.3f},
  acc_y_std={features['acc_y_avg_std']:.3f},
  acc_z_std={features['acc_z_avg_std']:.3f}
- Night steps: {features['steps_sum']}
- Calories burned: {features['calories_sum']}

MODEL PREDICTIONS (these are predictions, talk about them in future future):
- Sleep efficiency: {preds['pred_sleep_efficiency']:.2f}
- Sleep duration: {preds['pred_sleep_duration']:.2f} hours
- Sleep latency: {preds['pred_sleep_latency']:.2f} hours
- Sleep quality: {"Good" if preds['pred_sleep_quality']==1 else "Poor"}
- Fragmented sleep: {"Yes" if preds['pred_fragmented']==1 else "No"}

TASK:
Return your response in EXACTLY the following format:

ANALYSIS:
<...>

RECOMMENDATIONS:
<...>

Rules:
- Do NOT mention AI or models
- Use simple, clear language
- Do NOT add extra sections
"""


def call_llm(prompt):
    r = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    text = r.choices[0].message.content

    analysis = ""
    recs = ""

    if "ANALYSIS:" in text and "RECOMMENDATIONS:" in text:
        analysis = text.split("ANALYSIS:")[1].split("RECOMMENDATIONS:")[0].strip()
        recs = text.split("RECOMMENDATIONS:")[1].strip()
    else:
        analysis = text

    return analysis, recs


# ------------------------------------------------------------
# API endpoint
# ------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    df = assign_night_date(df)

    run_id = str(uuid.uuid4())[:8]  # short unique ID
    plot_urls = generate_plots(df, run_id)

    night = aggregate_night(df)
    if len(night) != 1:
        raise HTTPException(400, "CSV must contain exactly one night")

    X = night[FEATURE_COLS]

    reg = reg_model.predict(X)[0]
    cls = cls_model.predict(X)[0]

    preds = {
        "pred_sleep_duration": float(reg[0]),
        "pred_sleep_latency": float(reg[1]),
        "pred_sleep_efficiency": float(reg[2]),
        "pred_sleep_quality": int(cls[0]),
        "pred_fragmented": int(cls[1]),
    }

    features = night.iloc[0].to_dict()
    analysis, recs = call_llm(build_prompt(features, preds))

    return {
        "predictions": preds,
        "analysis": analysis,
        "recommendations": recs,
        "plots": plot_urls
    }

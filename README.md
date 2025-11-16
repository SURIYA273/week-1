# 🚗 EV Range Predictor — Internship Project

**Project:** Predict driving range (km) of electric vehicles using performance and battery specs.  

**Author :** SURIYA D          

**Dataset :** `electric_vehicle_raw_data.csv` (source: Kaggle).

---


## Project Overview

This project trains a regression model to predict the driving range of electric vehicles from numeric features such as battery capacity, torque, efficiency, dimensions, and charging power. The trained model is exposed via a Streamlit app for single and batch predictions.

---


## Repository Structure
```
EV-Range-Project/
├── models/
│ ├── ev_range_predictor.joblib # trained pipeline (model + preprocessing)
│ └── metadata.json # features & target metadata
├── electric_vehicle_cleaned_data.csv
├── train_model.py
├── predict_example.py
├── streamlit_app.py
├── requirements.txt
└── README.md

```

---

## How to run locally

**1. Create & activate a virtual environment (recommended):**
```
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```


**2.Install dependencies:**
```
pip install -r requirements.txt
```

**3.Ensure the `models/ev_range_predictor.joblib` and `models/metadata.json` files exist and `electric_vehicle_cleaned_data.csv` is present (optional for sample evaluation).**



**4.Start the Streamlit app:**
```
streamlit run streamlit_app.py
```
Open `http://localhost:8501` in your browser.

---

## How to use the app

  **Single input:** Fill the form with vehicle specs and click Predict range.

  **Batch input:** Upload a CSV that contains the exact feature columns defined in models/metadata.json, click Predict, and download the result CSV.

  **Model evaluation:** If the cleaned CSV is present, the app shows a quick RMSE on a sample.
  
---

## Feature order (must match metadata.json):
```
top_speed_kmh
battery_capacity_kWh
number_of_cells
torque_nm
efficiency_wh_per_km
acceleration_0_100_s
fast_charging_power_kw_dc
length_mm
width_mm
height_mm
```

---


## Notes & best practices

Keep the model and metadata files consistent; if retraining, update metadata.json.

If the model file is large, store it in cloud storage (Google Drive / S3) and modify the app to download at startup.

For production, use HTTPS hosting and do not store sensitive info in the repo.

---

### 🧑‍💻 Author

SURIYA  [ B.E Student | Data & AI Enthusiast | Web Developer]

📧 Contact: suriyamail273@gmail.com .

💼 GitHub: https://github.com/SURIYA273.




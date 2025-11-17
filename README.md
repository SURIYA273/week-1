# 🚗 EV Range Predictor — Internship Project

**Project:** Predict driving range (km) of electric vehicles using performance and battery specs.  

**Author :** SURIYA D          

**Dataset :** `electric_vehicle_raw_data.csv` (source: Kaggle).  

**Technologies:** Python, Streamlit, Scikit-Learn, Flask, HTML/CSS/JS, OpenAI API

---


## 📌 Project Overview

This project contains two integrated modules:

**✅ 1. EV Range Predictor**

A machine-learning model that predicts the driving range (km) of an electric vehicle based on specifications such as:

Battery capacity                    
Torque                 
Efficiency                           
Dimensions                    
Fast-charging power                            
Acceleration                          
Top speed   

The ML model is deployed using Streamlit for:                  

Single prediction                           
Batch CSV prediction                                  
Quick evaluation (RMSE)       

<p align="center">
<img width="506" height="463" alt="Screenshot 2025-11-17 143739" src="https://github.com/user-attachments/assets/e85d9c7d-27bb-41e3-89aa-58f56f45e89d" />
</p>

---

**🤖 2. EV Chatbot**

An AI-powered chatbot that:                                                     
Answers common EV-related questions                                            
Explains EV features (battery, range, charging, specs...)                                            
Helps users understand predictions                                             
Can simulate conversations about EV performance                            
Uses Flask backend + OpenAI API                                                          
Connected to a responsive HTML/CSS/JS frontend                                                         

This makes the project interactive and suitable for real-world applications.                                 
<p align="center">
<img width="600" height="695" alt="ChatBot result" src="https://github.com/user-attachments/assets/a886a8c9-54ec-429d-be89-aadbc1d4b941" />
</p>



---


## Repository Structure
```

Week-1/
├── EV-range-Chatbot-project
│    ├── models/
│    │ ├── ev_range_predictor.joblib                 # trained pipeline (model + preprocessing)
│    │ └── metadata.json                             # features & target metadata
│    ├── backend/
│    │ └── chat_openai.py                            # Flask backend with OpenAI integration
│    ├── data/
│    │ └── electric_vehicle_cleaned_data.csv         # cleaned dataset
│    ├── static
│    │ └── chat_ui.html                              # chatbot UI page
│    └── requirements.txt                            # main ML environment packages
│
├── EV-Range-streamlit-Project/
│    ├── models/
│    │ ├── ev_range_predictor.joblib                 # trained pipeline (model + preprocessing)
│    │ └── metadata.json                             # features & target metadata
│    ├── electric_vehicle_cleaned_data.csv           # cleaned dataset
│    ├── streamlit_app.py                            # UI for predictions
│    └── requirements.txt                            # main ML environment packages
│
├── electric_vehicles_raw_data_2025.csv              # raw dataset for week 1 task
├── electric_vehicles_cleaned_data_2025.csv          # cleaned dataset for week 1 task
├── ev_range_predictor.joblib                        # trained pipeline (model + preprocessing) for week 2 task
├── report document.doc
├── week 1 output.png                                # Week 1 task output
├── streamlit_app_result.png                         # streamlit_app output
├── ChatBot result.png                               # chatbot output
│
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


**5.Start the Chatbot :**
```
python backend/chat_openai.py
```

Open: `http://127.0.0.1:5000` in your browser.

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




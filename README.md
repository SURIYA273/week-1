# 🚗 Electric Vehicle Battery Capacity Prediction using Machine Learning

## 📘 Project Overview
This project aims to **predict the battery capacity or battery degradation of electric vehicles (EVs)** using their performance and technical features such as range, motor power, charge time, and efficiency.

The model helps in understanding how different EV specifications influence battery performance — a valuable insight for EV manufacturers, researchers, and consumers looking to evaluate efficiency and lifespan.

---

## 🎯 Problem Statement
> Develop a Machine Learning model to predict the **battery capacity (kWh)** or **battery degradation** of an electric vehicle based on its specifications and performance metrics.

---

## 🧠 Objectives
- Collect and preprocess data from the Electric Car Performance and Battery dataset.
- Perform **Exploratory Data Analysis (EDA)** to identify correlations between features.
- Train multiple **regression models** to predict battery capacity.
- Evaluate and compare model performance using accuracy metrics (MAE, RMSE, R²).
- Provide insights into the most influential factors affecting EV battery capacity.

---

## 📊 Dataset
**Source:** [Kaggle – Electric Car Performance and Battery Dataset](https://www.kaggle.com/datasets/afnansaifafnan/electric-car-performance-and-battery-dataset)  
**Dataset ID:** `afnansaifafnan/electric-car-performance-and-battery-dataset`

**Key Features:**
| Feature | Description |
|----------|--------------|
| Range (km) | Maximum distance the EV can travel on a full charge |
| Charge Time (hours) | Time required to fully charge the EV |
| Efficiency (km/kWh) | Distance covered per unit of energy |
| Top Speed (km/h) | Maximum speed of the EV |
| Motor Power (kW) | Power output of the electric motor |
| Acceleration (0–100 km/h) | Time taken to reach 100 km/h |
| Battery Capacity (kWh) | Target variable to be predicted |

---

## ⚙️ Project Workflow

### 🗓 Week 1 – Data Preparation
1. Define the problem statement.  
2. Load dataset using `kagglehub`.  
3. Perform data cleaning and preprocessing:
   - Handle missing values  
   - Convert data types  
   - Remove duplicates and irrelevant columns  
4. Conduct **EDA** with correlation heatmaps and visualizations.  
5. Split data into training and testing sets.  

### 🧮 Week 2 – Model Building
1. Implement regression algorithms:
   - Linear Regression  
   - Random Forest Regressor  
   - XGBoost Regressor  
2. Evaluate models using metrics:
   - Mean Absolute Error (MAE)  
   - Root Mean Squared Error (RMSE)  
   - R² Score  
3. Compare results and choose the best-performing model.  
4. Save the trained model using `joblib` or `pickle`.  

---

## 🧰 Tech Stack
- **Language:** Python  
- **Libraries:**  
  - `pandas`, `numpy` – Data manipulation  
  - `matplotlib`, `seaborn` – Data visualization  
  - `scikit-learn` – Machine learning models and metrics  
  - `xgboost` – Advanced regression model  
  - `kagglehub` – Dataset import  

---

## 📈 Expected Output
- A trained ML model that accurately predicts EV battery capacity.  
- Visualizations showing relationships between features.  
- A project report detailing preprocessing, model performance, and insights.  

---

## 📂 Project Structure

```
EV-Battery-Capacity-Prediction/
│
├── data/
│ └── electric_car_dataset.csv
│
├── notebooks/
│ ├── week1_preprocessing.ipynb
│ └── week2_model_building.ipynb
│
├── model/
│ └── battery_model.pkl
│
├── README.md
└── requirements.txt

```
---

## 🧾 Results and Insights
- Higher motor power and longer range typically correlate with higher battery capacity.  
- Efficiency (km/kWh) and charge time significantly influence the overall prediction accuracy.  
- Random Forest and XGBoost models tend to outperform simple linear models in predicting complex EV behavior.  

---

## 👨‍💻 Author
**Suriya D (Demon)**  
AI/ML Intern — ACIT Sponsored Internship  
📧 [suriyamail273@gmail.com](mailto:suriyamail273@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/) *(Add your profile link here)*

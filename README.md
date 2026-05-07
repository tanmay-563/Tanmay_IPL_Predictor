# 🏏 Tanmay's IPL Predictor

An AI-powered IPL match winner prediction platform built using Machine Learning, historical IPL analytics, and explainable AI concepts.

---

## 🚀 Live Features

* 🤖 IPL Winner Prediction using Machine Learning
* 📊 Confidence Score Prediction
* 🧠 Explainable AI Reasoning System
* 🏟️ Venue-Based Historical Analytics
* ⚔️ Head-to-Head Rivalry Insights
* 🎨 Modern Streamlit UI
* 🌙 Premium Dark Theme Interface
* 📈 XGBoost-Based Prediction Model

---

## 🧠 Project Overview

This project predicts the probable winner of an IPL match using:

* Team combinations
* Match venue
* Historical venue win rates
* Head-to-head statistics
* XGBoost machine learning model

Unlike basic prediction systems, this project also explains *why* a team is predicted to win.

Example:

> "RCB historically performs strongly at this venue and has a better rivalry record against CSK."

---

# 🖼️ Preview

## Home Page

(Add screenshot here)

## Prediction Result

(Add screenshot here)

---

# 🛠️ Tech Stack

| Technology   | Purpose                 |
| ------------ | ----------------------- |
| Python       | Core Programming        |
| Streamlit    | Frontend Web App        |
| XGBoost      | Machine Learning Model  |
| Scikit-learn | Encoding & ML Utilities |
| Pandas       | Data Processing         |
| Joblib       | Model Serialization     |
| Git & GitHub | Version Control         |

---

# 📂 Project Structure

```bash
Tanmay_IPL_Predictor/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── ipl_model.pkl
├── team_encoder.pkl
├── venue_encoder.pkl
├── winner_encoder.pkl
├── venue_win_rates.pkl
├── head_to_head.pkl
│
├── ipl-logo.jpg
├── ipl-all-team-logo.jpg
├── Virat-kohli-signing.gif
```

---

# ⚙️ Machine Learning Workflow

## 1. Data Collection

Used IPL datasets from 2022–2026 containing:

* Ball-by-ball data
* Venue information
* Team information
* Match details

---

## 2. Feature Engineering

Converted ball-by-ball data into match-level analytics:

* Team 1
* Team 2
* Venue
* Winner

Additional features engineered:

* Venue win rates
* Head-to-head statistics

---

## 3. Model Training

Used:

* XGBoost Classifier

Hyperparameter tuning performed using:

* GridSearchCV

---

## 4. Explainable AI Layer

The project combines:

* Machine Learning predictions
* Historical analytics
* Statistical reasoning

To provide transparent match predictions.

---

# 📊 Model Performance

| Model            | Accuracy |
| ---------------- | -------- |
| Baseline XGBoost | ~49%     |
| Tuned XGBoost    | ~56%     |

> Cricket is highly unpredictable, so explainability and analytics are equally important alongside prediction accuracy.

---

# 💻 Installation & Setup

## 1. Clone Repository

```bash
git clone https://github.com/tanmay-563/Tanmay_IPL_Predictor.git
```

## 2. Navigate Into Project

```bash
cd Tanmay_IPL_Predictor
```

## 3. Create Virtual Environment

```bash
python -m venv venv
```

## 4. Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## 6. Run Streamlit App

```bash
streamlit run app.py
```

---

# ☁️ Deployment

This project can be deployed easily using:

* Streamlit Community Cloud

Deployment Steps:

1. Push project to GitHub
2. Open Streamlit Cloud
3. Connect GitHub repository
4. Select `app.py`
5. Deploy

---

# ✨ Future Improvements

* Toss-based prediction
* Recent team form analysis
* Player-level analytics
* Live IPL score integration
* Dynamic team logos
* Match simulation engine
* Win probability graphs

---

# 👨‍💻 Author

## Tanmay

> "Where cricket intelligence meets machine learning."

Built with ❤️ using AI, analytics, and crick

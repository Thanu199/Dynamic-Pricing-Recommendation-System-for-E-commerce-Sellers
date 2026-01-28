# Dynamic Pricing Recommendation System for E-commerce Sellers

## Overview
This project implements a Dynamic Pricing Recommendation System that predicts optimal product prices for e-commerce sellers using supervised machine learning and a free open-source LLM is integrated as an explainability layer to generate human-readable pricing insights and business recommendations. The system analyzes historical sales data and provides intelligent price recommendations along with business-friendly explanations.

## Features
- Price prediction using **XGBoost Regressor (XGBRegressor)**   
- Data preprocessing and feature engineering  
- Handling of categorical and numerical features   
- Interactive Streamlit web application  
- AI-based explanation layer for pricing insights  

## Project Structure
- `models.py` – Model training and evaluation  
- `app2.py` – Streamlit application for price prediction  
- `pricing_model.pkl` – Trained machine learning model  
- `data/` – Dataset folder  

## Tech Stack
- Python, Pandas, NumPy  
- Scikit-learn, XGBoost  
- Streamlit  

## How to Run
1. Train the model:
   ```bash
   python models.py
2. Run the Streamlit app:
   ```bash
     python -m streamlit run app2.py

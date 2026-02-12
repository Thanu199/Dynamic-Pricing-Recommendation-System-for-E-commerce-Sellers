import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import time
import requests

st.set_page_config(
    page_title="Dynamic Pricing Recommendation System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .prediction-box {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        text-align: center;
        margin: 20px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .llm-response-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 20px 0;
        color: #333;
        line-height: 1.6;
    }
    h1, h2, h3 {
        color: white;
    }
    .stSelectbox label, .stNumberInput label, .stTextInput label {
        color: white !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = joblib.load('pricing_model.pkl')
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'pricing_model.pkl' not found in the current directory!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

def query_llm(prompt: str) -> str:
    try:
        api_url = "https://router.huggingface.co/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer HF_TOKEN",
        }
        
        payload = {
            "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                    ],
            "model": "meta-llama/Llama-3.1-8B-Instruct:novita"

        }
        
        response = requests.post(api_url, headers=headers, json=payload)
        # response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]['content']
        # if isinstance(result, list) and len(result) > 0:
        #     return result[0].get("generated_text", "")
        # elif isinstance(result, dict):
        #     return result.get("generated_text", "")
        # else:
        #     return ""
            
    except requests.exceptions.Timeout:
        return "⚠️ The AI analysis is taking longer than expected. Please try again."
    except requests.exceptions.HTTPError as e:
        return f"⚠️ Unable to generate AI insights at this moment. Please try again later."
    except requests.exceptions.RequestException:
        return "⚠️ Network error occurred. Please check your connection and try again."
    except Exception:
        return "⚠️ Unable to generate AI insights. The pricing recommendation is still valid."

model = load_model()

st.title("💰 Dynamic Pricing Recommendation System for E-commerce Sellers")
st.markdown("### Optimize your product pricing with AI-powered recommendations")

with st.sidebar:
    st.header("📊 Input Parameters")
    
    status_options = ["Pending", "Shipped", "Cancelled", "Shipped - Delivered to Buyer", 
                      "Cancelled - Customer Request", "Shipped - Returning to Seller",
                      "Shipped - Rejected by Buyer", "Shipped - Lost in Transit",
                      "Shipped - Out for Delivery", "Shipped - Picked Up", "Unknown"]
    status = st.selectbox("Order Status", status_options)
    
    fulfillment_options = ["Amazon", "Merchant", "Unknown"]
    fulfillment = st.selectbox("Fulfillment Type", fulfillment_options)
    
    sales_channel_options = ["Amazon.in", "Non-Amazon", "Unknown"]
    sales_channel = st.selectbox("Sales Channel", sales_channel_options)
    
    ship_service_options = ["Expedited", "Standard", "Second Class", "Unknown"]
    ship_service = st.selectbox("Ship Service Level", ship_service_options)
    
    category_options = ["T-shirt", "Shirt", "Blazer", "Trousers", "Kurta", "Top", 
                        "Western Dress", "Saree", "Set", "Ethnic Dress", "Unknown"]
    category = st.selectbox("Product Category", category_options)
    
    size_options = ["M", "L", "XL", "XXL", "S", "3XL", "4XL", "5XL", "6XL", "XS", 
                    "Free", "Unknown"]
    size = st.selectbox("Product Size", size_options)
    
    qty = st.number_input("Quantity", min_value=0, max_value=100, value=1, step=1)
    
    courier_status_options = ["Shipped", "Cancelled", "Unshipped", "Pending", "Unknown"]
    courier_status = st.selectbox("Courier Status", courier_status_options)
    
    currency_options = ["INR", "USD", "Unknown"]
    currency = st.selectbox("Currency", currency_options)
    
    ship_state_options = ["MAHARASHTRA", "KARNATAKA", "TAMIL NADU", "DELHI", "UTTAR PRADESH", 
                          "TELANGANA", "GUJARAT", "RAJASTHAN", "WEST BENGAL", "KERALA", 
                          "PUNJAB", "HARYANA", "MADHYA PRADESH", "ANDHRA PRADESH", "Unknown"]
    ship_state = st.selectbox("Ship State", ship_state_options)
    
    style_options = ["Unknown"]
    style = st.selectbox("Style", style_options)
    
    sku = st.text_input("SKU", value="Unknown")
    
    asin = st.text_input("ASIN", value="Unknown")
    
    ship_city = st.text_input("Ship City", value="Unknown")
    
    fulfilled_by = st.text_input("Fulfilled By", value="Unknown")
    
    b2b_options = [False, True]
    b2b = st.selectbox("B2B Transaction", b2b_options)
    
    st.markdown("---")
    st.subheader("📅 Order Date")
    order_date = st.date_input("Select Date", value=datetime.now())
    
    order_year = order_date.year
    order_month = order_date.month
    order_day = order_date.day

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button("🚀 Generate Price Recommendation")

if predict_button:
    with st.spinner("🔮 Analyzing market data and generating optimal price..."):
        time.sleep(1.5)
        
        try:
            input_data = pd.DataFrame({
                'Status': [status],
                'Fulfilment': [fulfillment],
                'Sales Channel ': [sales_channel],
                'ship-service-level': [ship_service],
                'Category': [category],
                'Size': [size],
                'Qty': [qty],
                'currency': [currency],
                'ship-state': [ship_state],
                'ship-city': [ship_city],
                'Courier Status': [courier_status],
                'Style': [style],
                'SKU': [sku],
                'ASIN': [asin],
                'fulfilled-by': [fulfilled_by],
                'B2B': [b2b],
                'order_year': [order_year],
                'order_month': [order_month],
                'order_day': [order_day]
            })
            
            prediction = model.predict(input_data)[0]
            
            st.success("✅ Price Recommendation Generated Successfully!")
            
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="color: #667eea;">Recommended Price</h2>
                <h1 style="color: #764ba2; font-size: 48px;">₹ {prediction:.2f}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Min Price Range</h3>
                    <h2>₹ {prediction * 0.9:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Optimal Price</h3>
                    <h2>₹ {prediction:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Max Price Range</h3>
                    <h2>₹ {prediction * 1.1:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("🤖 AI-Powered Pricing Insights")
            
            with st.spinner("Generating personalized insights..."):
                prompt = f"""You are a pricing strategy expert for e-commerce sellers. Provide a clear, professional analysis in 3 paragraphs:

1. Explain in simple business language why ₹{prediction:.2f} is the recommended price for this {category} product.

2. Identify 2-3 key factors that influenced this price recommendation:
   - Product Category: {category}
   - Size: {size}
   - Quantity: {qty}
   - Fulfillment: {fulfillment}
   - Sales Channel: {sales_channel}
   - Ship State: {ship_state}
   - B2B Transaction: {'Yes' if b2b else 'No'}

3. Provide 3 practical, actionable tips for the seller to improve revenue or optimize their pricing strategy.

Keep the response clear, concise, and actionable for business owners."""

                llm_response = query_llm(prompt)
                
                if llm_response and not llm_response.startswith("⚠️"):
                    st.markdown(f"""
                    <div class="llm-response-card">
                        {llm_response.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(llm_response if llm_response else "Unable to generate AI insights at this time.")
            
            st.markdown("---")
            st.subheader("📈 Pricing Insights")
            
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                st.info(f"""
                **Category Analysis:**
                - Product: {category}
                - Size: {size}
                - Quantity: {qty}
                """)
            
            with insights_col2:
                st.info(f"""
                **Order Details:**
                - Status: {status}
                - Channel: {sales_channel}
                - B2B: {'Yes' if b2b else 'No'}
                """)
            
        except ValueError as ve:
            st.error(f"❌ Input validation error: {str(ve)}")
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.error("Please ensure all inputs match the expected format.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white;'>
    <p>Powered by Machine Learning | Dynamic Pricing Recommendation System</p>
</div>

""", unsafe_allow_html=True)

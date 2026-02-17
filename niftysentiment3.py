import requests
import pandas as pd
import time
import streamlit as st
import plotly.express as px
import os
from datetime import datetime
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from collections import deque
from scipy.stats import norm
#import talib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import sys
from requests.exceptions import RequestException
import json
import pickle


# ====== PERSISTENCE FUNCTIONS ======
# These functions save/load session state to files so data persists across app restarts

def save_session_state():
    """Save all session state data to files for persistence"""
    try:
        # Save previous_data (dictionary)
        with open("previous_data.pkl", "wb") as f:
            pickle.dump(st.session_state.get("previous_data", {}), f)
        
        # Save rolling_data (dictionary with deques - need special handling)
        rolling_data_serializable = {}
        for key, deque_obj in st.session_state.get("rolling_data", {}).items():
            rolling_data_serializable[key] = list(deque_obj)
        with open("rolling_data.json", "w") as f:
            json.dump(rolling_data_serializable, f)
        
        # Save alerts (list)
        with open("alerts.json", "w") as f:
            json.dump(st.session_state.get("alerts", []), f)
        
        # Save sent_alerts (set - convert to list for JSON)
        with open("sent_alerts.json", "w") as f:
            json.dump(list(st.session_state.get("sent_alerts", set())), f)
        
        # Save sentiment_history (list)
        with open("sentiment_history.json", "w") as f:
            json.dump(st.session_state.get("sentiment_history", []), f)
        
        # Save previous_avg_iv (float)
        if "previous_avg_iv" in st.session_state:
            with open("previous_avg_iv.json", "w") as f:
                json.dump(st.session_state.previous_avg_iv, f)
        
        # Save strike_sentiment_log (already saved via save_sentiment_log, but ensure it's saved)
        if "strike_sentiment_log" in st.session_state:
            save_sentiment_log()
            
    except Exception as e:
        # Silently fail - don't break the app if saving fails
        pass

def load_session_state():
    """Load all session state data from files on app startup"""
    try:
        # Load previous_data
        if os.path.exists("previous_data.pkl"):
            with open("previous_data.pkl", "rb") as f:
                st.session_state.previous_data = pickle.load(f)
        
        # Load rolling_data
        if os.path.exists("rolling_data.json"):
            with open("rolling_data.json", "r") as f:
                rolling_data_dict = json.load(f)
                # Convert lists back to deques
                st.session_state.rolling_data = {
                    key: deque(data_list, maxlen=5) 
                    for key, data_list in rolling_data_dict.items()
                }
        
        # Load alerts
        if os.path.exists("alerts.json"):
            with open("alerts.json", "r") as f:
                st.session_state.alerts = json.load(f)
        
        # Load sent_alerts (convert list back to set)
        if os.path.exists("sent_alerts.json"):
            with open("sent_alerts.json", "r") as f:
                st.session_state.sent_alerts = set(json.load(f))
        
        # Load sentiment_history
        if os.path.exists("sentiment_history.json"):
            with open("sentiment_history.json", "r") as f:
                st.session_state.sentiment_history = json.load(f)
        
        # Load previous_avg_iv
        if os.path.exists("previous_avg_iv.json"):
            with open("previous_avg_iv.json", "r") as f:
                st.session_state.previous_avg_iv = json.load(f)
        
        # Load strike_sentiment_log (already handled in enhanced_analyze_data, but ensure it's loaded)
        if "strike_sentiment_log" not in st.session_state:
            try:
                if os.path.exists("sentiment_log_backup.csv"):
                    st.session_state.strike_sentiment_log = pd.read_csv("sentiment_log_backup.csv").to_dict("records")
                else:
                    st.session_state.strike_sentiment_log = []
            except:
                st.session_state.strike_sentiment_log = []
                
    except Exception as e:
        # If loading fails, initialize with defaults
        pass


def save_sentiment_log():
    log_df = pd.DataFrame(st.session_state.strike_sentiment_log)
    log_df.to_csv("sentiment_log_backup.csv", index=False)


@st.cache_data(ttl=10800)
def get_expiry_dates_cached():
    return get_expiry_dates()


sys.setrecursionlimit(5000)  # Default is ~1000
# 🔹 Set IST Timezone
IST = pytz.timezone("Asia/Kolkata")

# Streamlit Page Configuration
st.set_page_config(page_title="Nifty Options IV Spike Dashboard", layout="wide")

# 🔹 Load persisted session state data on startup
load_session_state()

# 🔹 Dhan API Credentials (Replace with your own)
# ====== Dhan API Config ======
CLIENT_ID = '1100244268'
ACCESS_TOKEN= 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzcxMzkwMDk4LCJpYXQiOjE3NzEzMDM2OTgsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTAwMjQ0MjY4In0.0KMLJ1IQS1GMwTOes-h1uqm5eoC3c9d1sbtr3HDYSPCKgSdMQPR9wONx4n01RvN6CLpS_VzYxYcJXewCTv1YjA'  # Replace with your Access Token

HEADERS = {
    'client-id': CLIENT_ID,
    'access-token': ACCESS_TOKEN,
    'Content-Type': 'application/json'
}

# 🔹 Telegram Bot Credentials (Replace with your own)
TELEGRAM_BOT_TOKEN = "7967747029:AAFyMl5zF1XvRqrhY5CIoR_1_EJwiEyrAqw"
TELEGRAM_CHAT_ID = "-470480347"

# 🔹 API Endpoints
OPTION_CHAIN_URL = "https://api.dhan.co/v2/optionchain"
EXPIRY_LIST_URL = "https://api.dhan.co/v2/optionchain/expirylist"

# 🔹 Nifty Index Code
NIFTY_SCRIP_ID = 13
NIFTY_SEGMENT = "IDX_I"

# CSV File Path
CSV_FILE = "nifty_option_chain.csv"



# Store rolling IV/OI history
if "rolling_data" not in st.session_state:
    st.session_state.rolling_data = {}

# Store previous data for comparison
if "previous_data" not in st.session_state:
    st.session_state.previous_data = {}
previous_data = st.session_state.previous_data  # Use session state storage


# Store last cycle alerts to prevent duplicates
sent_alerts = {}

# Streamlit session state for alerts
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# Function to send Telegram Alerts
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=data)

# Function to fetch expiry dates
def get_expiry_dates():
    url = "https://api.dhan.co/v2/optionchain/expirylist"
    payload = {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I"}
    try:
        response = requests.post(url, json=payload, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()['data']
    except RequestException as e:
        st.error(f"Connection error while fetching expiry list: {e}")
        st.stop()


def fetch_option_chain(expiry):
    url = "https://api.dhan.co/v2/optionchain"
    payload = {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I", "Expiry": expiry}
    response = requests.post(url, json=payload, headers=HEADERS)
    time.sleep(3)
    if response.status_code != 200:
        st.error(f"Failed to fetch option chain: {response.text}")
        st.stop()
    return response.json()

# Function to analyze IV, OI, and Greeks
# Define Alert Thresholds
IV_SPIKE_THRESHOLD = 5  # IV increase threshold (%)
IV_CRASH_THRESHOLD = 5   # IV drop threshold (%)
OI_SPIKE_THRESHOLD = 10  # OI increase threshold (%)
GAMMA_THRESHOLD = 0.02   # Gamma exposure threshold
THETA_DECAY_THRESHOLD = -20  # Theta erosion threshold
PRICE_STABILITY_THRESHOLD = 0.5  # Price stability for IV-based alerts

# 🔹 Add these constants after your existing thresholds
EXPIRY_DAY_THETA_ACCELERATION = -50  # Theta acceleration threshold on expiry
PIN_RISK_THRESHOLD = 100000  # OI concentration threshold for pin risk
VOLATILITY_CRUSH_THRESHOLD = 15  # IV drop threshold on expiry day
GAMMA_SQUEEZE_THRESHOLD = 0.05  # High gamma threshold for squeeze detection

# 🔹 ENHANCED CONSTANTS - Add these after your existing thresholds
MOMENTUM_WINDOW = 10  # Data points for momentum calculation
VWAP_DEVIATION_THRESHOLD = 0.5  # VWAP deviation alert threshold
LIQUIDITY_THRESHOLD = 50000  # Minimum OI for liquid options
SMART_MONEY_THRESHOLD = 1000000  # Large trade detection
VOLATILITY_REGIME_PERIODS = [5, 10, 20]  # Multi-timeframe vol analysis
MARKET_TIMING_ZONES = {
    "opening": (9, 30, 10, 30),  # Opening hour volatility
    "mid_session": (11, 0, 14, 0),  # Calmer period
    "power_hour": (14, 0, 15, 30)  # Final 1.5 hours
}

def score_option_sentiment(row):
    """
    Scores sentiment based on OI_Change, LTP_Change, IV_Change, Theta, Vega.
    """
    score = 0
    
    # OI change scoring
    if row['OI_Change'] > 0:
        score += 1
    elif row['OI_Change'] < 0:
        score -= 1

    # LTP change scoring
    if row['LTP_Change'] > 0:
        score += 1
    elif row['LTP_Change'] < 0:
        score -= 1

    # IV change scoring
    if row['IV_Change'] > 0:
        score += 1
    elif row['IV_Change'] < 0:
        score -= 1

    # Theta (positive favors writers, negative favors buyers)
    if row['Theta'] < 0:
        score += 1
    elif row['Theta'] > 0:
        score -= 1

    # Vega (positive favors buyers, negative favors sellers)
    if row['Vega'] > 0:
        score += 1
    elif row['Vega'] < 0:
        score -= 1
    # IV vs LTP Divergence (Detect Buyer or Writer Intent)
    if row['IV_Change'] > 0 and row['LTP_Change'] < 0:
        score -= 1  # Likely writing (increased IV but price weak)
    elif row['IV_Change'] > 0 and row['LTP_Change'] > 0:
        score += 1  # Likely buying (willing to pay IV premium)

    # Bias label
    if score >= 3:
        bias = "Aggressive Buying"
    elif score >= 1:
        bias = "Mild Buying"
    elif score == 0:
        bias = "Neutral"
    elif score <= -3:
        bias = "Aggressive Writing"
    else:
        bias = "Mild Writing"

    return pd.Series([score, bias], index=['SentimentScore', 'SentimentBias'])


# 🔹 Add this function to check if today is expiry day
def is_expiry_day(expiry_date_str):
    """Check if today is the expiry day"""
    try:
        expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
        return expiry_date == datetime.now(IST).date()
    except:
        return False


def sentiment_color(score):
    if score >= 3:
        return '#006400'  # Dark Green
    elif score >= 1:
        return '#90EE90'  # Light Green
    elif score <= -3:
        return '#8B0000'  # Dark Red
    elif score <= -1:
        return '#FF6347'  # Light Red
    else:
        return 'gray'



def enhanced_analyze_data(option_chain):
    # Ensure previous data persists across Streamlit reruns
    if "previous_data" not in st.session_state:
        st.session_state.previous_data = {}  

    previous_data = st.session_state.previous_data  # Use session state storage

    if "data" not in option_chain or "oc" not in option_chain["data"]:
        st.error("Invalid option chain data received!")
        return pd.DataFrame()

    option_chain_data = option_chain["data"]["oc"]
    data_list = []
    underlying_price = option_chain["data"]["last_price"]  # Fetch underlying price

    # Determine ATM Strike
    atm_strike = min(option_chain_data.keys(), key=lambda x: abs(float(x) - underlying_price))
    atm_strike = float(atm_strike)

    # Define range for ATM ± 4 strikes
    min_strike = atm_strike - 5 * 50  # Assuming 50-point strike intervals
    max_strike = atm_strike + 5 * 50
    
    
    for strike, contracts in option_chain_data.items():
        strike_price = float(strike)  # Convert key to float

        # Filter only ATM ± 4 strikes
        if strike_price < min_strike or strike_price > max_strike:
            continue

        ce_data = contracts.get("ce", {})
        pe_data = contracts.get("pe", {})

        ce_iv = ce_data.get("implied_volatility", 0)
        ce_oi = ce_data.get("oi", 0)
        ce_ltp = ce_data.get("last_price", 0)
        ce_delta = ce_data.get("greeks", {}).get("delta", 0)
        ce_gamma = ce_data.get("greeks", {}).get("gamma", 0)
        ce_theta = ce_data.get("greeks", {}).get("theta", 0)
        ce_vega = ce_data.get("greeks", {}).get("vega", 0)
        ce_volume = ce_data.get("volume", 0) 

        pe_iv = pe_data.get("implied_volatility", 0)
        pe_oi = pe_data.get("oi", 0)
        pe_ltp = pe_data.get("last_price", 0)
        pe_delta = pe_data.get("greeks", {}).get("delta", 0)
        pe_gamma = pe_data.get("greeks", {}).get("gamma", 0)
        pe_theta = pe_data.get("greeks", {}).get("theta", 0)
        pe_vega = pe_data.get("greeks", {}).get("vega", 0)
        pe_volume = pe_data.get("volume", 0)
        
        # Add data to list
        data_list.append({
            "StrikePrice": strike_price,
            "Type": "CE",
            "IV": ce_iv,
            "OI": ce_oi,
            "LTP": ce_ltp,
            "Delta": ce_delta,
            "Gamma": ce_gamma,
            "Theta": ce_theta,
            "Vega": ce_vega,
            "Volume": ce_volume})

        data_list.append({
            "StrikePrice": strike_price,
            "Type": "PE",
            "IV": pe_iv,
            "OI": pe_oi,
            "LTP": pe_ltp,
            "Delta": pe_delta,
            "Gamma": pe_gamma,
            "Theta": pe_theta,
            "Vega": pe_vega,
            "Volume": pe_volume})

    df = pd.DataFrame(data_list)

    df["Timestamp"] = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    df = df.sort_values(by=["StrikePrice", "Type"])

    alerts = []
    prev_underlying_price = previous_data.get("underlying_price", underlying_price)

    
    # Update rolling data
    rolling_data = st.session_state.rolling_data
    for _, row in df.iterrows():
        key = f"{row['StrikePrice']}_{row['Type']}"
        if key not in rolling_data:
            rolling_data[key] = deque(maxlen=5)
        rolling_data[key].append({"IV": row["IV"], "OI": row["OI"], "LTP": row["LTP"]})

    # Calculate Net OI Imbalance
    net_oi_df = df.pivot(index='StrikePrice', columns='Type', values='OI').fillna(0)
    net_oi_df['NetOI'] = net_oi_df['PE'] - net_oi_df['CE']

    # Add OI Change % to DataFrame
    df["OI_Change"] = df.apply(lambda row: (
        ((row["OI"] - previous_data.get(f'{row["StrikePrice"]}_{row["Type"]}', {}).get("OI", row["OI"])) / row["OI"]) * 100
    ) if row["OI"] else 0, axis=1)

    df["IV_Change"] = df.apply(lambda row: (
        ((row["IV"] - previous_data.get(f'{row["StrikePrice"]}_{row["Type"]}', {}).get("IV", row["IV"])) / row["IV"]) * 100
    ) if row["IV"] else 0, axis=1)

    df["LTP_Change"] = df.apply(lambda row: (
        ((row["LTP"] - previous_data.get(f'{row["StrikePrice"]}_{row["Type"]}', {}).get("LTP", row["LTP"])) / row["LTP"]) * 100
    ) if row["LTP"] else 0, axis=1)

    df[['SentimentScore', 'SentimentBias']] = df.apply(score_option_sentiment, axis=1)

    # Detect Straddle Writing at Same Strike
    straddle_writes = []
    for strike in df['StrikePrice'].unique():
        ce_bias = df[(df['StrikePrice'] == strike) & (df['Type'] == 'CE')]['SentimentBias'].values
        pe_bias = df[(df['StrikePrice'] == strike) & (df['Type'] == 'PE')]['SentimentBias'].values
        if "Aggressive Writing" in ce_bias and "Aggressive Writing" in pe_bias:
            straddle_writes.append(strike)

    if straddle_writes:
        for strike in straddle_writes:
            st.warning(f"🔗 **Straddle Writing Detected at {int(strike)}**")
            send_telegram_alert(f"🔗 STRADDLE WRITING: Both CE/PE aggressively written at {int(strike)}")

    
    # Initialize strike-level sentiment log
    if "strike_sentiment_log" not in st.session_state:
        try:
            st.session_state.strike_sentiment_log = pd.read_csv("sentiment_log_backup.csv").to_dict("records")
        except FileNotFoundError:
            st.session_state.strike_sentiment_log = []


    # Append current timestamped strike-level sentiment
    timestamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    for _, row in df.iterrows():
        st.session_state.strike_sentiment_log.append({
            "Timestamp": timestamp,
            "StrikePrice": row["StrikePrice"],
            "Type": row["Type"],
            "LTP": row["LTP"],
            "OI": row["OI"],
            "LTP_Change": row["LTP_Change"],
            "OI_Change": row["OI_Change"],
            "Volume": row["Volume"],
            "IV": row["IV"],
            "IV_Change": row["IV_Change"],
            "Theta": row["Theta"],
            "Vega": row["Vega"],
            "SentimentScore": row["SentimentScore"],  # ✅ Correct
            "SentimentBias": row["SentimentBias"],    # ✅ Correct
            "UnderlyingValue": underlying_price
        })

    save_sentiment_log()



    # Limit to entire session
    st.session_state.strike_sentiment_log = st.session_state.strike_sentiment_log[-20500:]


    log_df = pd.DataFrame(st.session_state.strike_sentiment_log)

    # Telegram for strong signals
    top_signals = df[abs(df['SentimentScore']) >= 3]
    if not top_signals.empty:
        msg = "📣 *Market Sentiment Alerts*\n"
        for _, row in top_signals.iterrows():
            msg += f"• {row['Type']} {int(row['StrikePrice'])} → *{row['SentimentBias']}* (Score: {int(row['SentimentScore'])})\n"
        send_telegram_alert(msg)

    # ATM ±5 strike sentiment
    atm_range_df = df[(df['StrikePrice'] >= atm_strike - 5 * 50) & (df['StrikePrice'] <= atm_strike + 5 * 50)]
    avg_sentiment_ce = atm_range_df[atm_range_df['Type'] == 'CE']['SentimentScore'].mean()
    avg_sentiment_pe = atm_range_df[atm_range_df['Type'] == 'PE']['SentimentScore'].mean()

    # Initialize sentiment history if not already
    if "sentiment_history" not in st.session_state:
        st.session_state.sentiment_history = []

    # Append new sentiment snapshot
    st.session_state.sentiment_history.append({
        "Timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "CE_Avg_Sentiment": avg_sentiment_ce,
        "PE_Avg_Sentiment": avg_sentiment_pe
    })
    
    # Save sentiment history periodically (limit size to prevent file bloat)
    if len(st.session_state.sentiment_history) > 1000:
        st.session_state.sentiment_history = st.session_state.sentiment_history[-1000:]
    save_session_state()

    
    # Show in UI
    st.markdown("### 📊 Average Sentiment (ATM ±5 Strikes)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Call Avg Sentiment", f"{avg_sentiment_ce:+.2f}")
    with col2:
        st.metric("Put Avg Sentiment", f"{avg_sentiment_pe:+.2f}")

    with st.expander("🧭 CE/PE Zone Sentiment Gauges (ATM ±5)", expanded=False):
        fig_gauge = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}]],
            subplot_titles=("CE Sentiment", "PE Sentiment")
        )

        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=avg_sentiment_ce,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Calls"},
            gauge={
                'axis': {'range': [-5, 5]},
                'bar': {'color': "green" if avg_sentiment_ce > 0 else "red"},
                'steps': [
                    {'range': [-5, -3], 'color': "red"},
                    {'range': [-3, -1], 'color': "orange"},
                    {'range': [-1, 1], 'color': "gray"},
                    {'range': [1, 3], 'color': "lightgreen"},
                    {'range': [3, 5], 'color': "green"}
                ],
            }
        ), row=1, col=1)

        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=avg_sentiment_pe,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Puts"},
            gauge={
                'axis': {'range': [-5, 5]},
                'bar': {'color': "green" if avg_sentiment_pe > 0 else "red"},
                'steps': [
                    {'range': [-5, -3], 'color': "red"},
                    {'range': [-3, -1], 'color': "orange"},
                    {'range': [-1, 1], 'color': "gray"},
                    {'range': [1, 3], 'color': "lightgreen"},
                    {'range': [3, 5], 'color': "green"}
                ],
            }
        ), row=1, col=2)

        fig_gauge.update_layout(height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Enhanced HFT Algo Scanner with Advanced Features

    # Prepare DataFrame
    log_df = pd.DataFrame(st.session_state.strike_sentiment_log)

    if not log_df.empty:
        # Multi-strike analysis for "big money" detection
        st.subheader("🎯 HFT Algo Scanner - Multi-Strike Analysis")
        
        # Enhanced calculations for institutional flow detection
        log_df['Exposure'] = log_df['OI'] * log_df['LTP']
        log_df['Volume_Weighted_Exposure'] = log_df.get('Volume', 1) * log_df['Exposure']
        log_df['Notional_Value'] = log_df['OI'] * log_df['LTP'] * 75  # Assuming lot size of 75
        
        # Calculate percentage changes with better handling
        log_df['OI_Change_Pct'] = log_df.get('OI_Change', 0) / (log_df['OI'] + 1) * 100
        log_df['IV_Change_Pct'] = log_df.get('IV_Change', 0) / (log_df.get('IV', 1) + 0.01) * 100
        log_df['LTP_Change_Pct'] = log_df.get('LTP_Change', 0) / (log_df['LTP'] + 0.01) * 100
        
        # NEW: Add Money Flow Index (MFI) calculation
        def calculate_money_flow_index(df, window=14):
            """Calculate Money Flow Index for options"""
            typical_price = (log_df['LTP'] + log_df.get('High', log_df['LTP']) + log_df.get('Low', log_df['LTP'])) / 3
            money_flow = typical_price * log_df.get('Volume', 1)
            
            positive_flow = money_flow.where(log_df['LTP_Change_Pct'] > 0, 0)
            negative_flow = money_flow.where(log_df['LTP_Change_Pct'] < 0, 0)
            
            positive_mf = positive_flow.rolling(window=window).sum()
            negative_mf = negative_flow.rolling(window=window).sum()
            
            mfi = 100 - (100 / (1 + (positive_mf / (negative_mf + 1))))
            return mfi.fillna(50)  # Neutral starting point
        
        log_df['MFI'] = calculate_money_flow_index(log_df)
        
        # Enhanced Activity Metric with MFI
        def calculate_enhanced_activity_metric(row):
            """
            Enhanced algorithm with MFI and more sophisticated scoring
            """
            base_exposure = row['Volume_Weighted_Exposure']
            oi_change = abs(row['OI_Change_Pct'])
            volume = row.get('Volume', 1)
            ltp_change = row['LTP_Change_Pct']
            iv_change = row['IV_Change_Pct']
            mfi = row.get('MFI', 50)
            
            # Volume surge factor (improved scaling)
            volume_factor = np.log1p(volume / 50)  # Logarithmic scaling
            
            # OI change momentum with acceleration
            oi_momentum = min(oi_change / 3, 4)
            
            # Price action factor with momentum consideration
            price_momentum = abs(ltp_change) / 8
            
            # IV factor for institutional detection  
            iv_factor = abs(iv_change) / 15
            
            # MFI factor (extreme values indicate institutional activity)
            mfi_factor = abs(mfi - 50) / 25  # Distance from neutral
            
            # Composite activity score with weights
            activity_score = base_exposure * (
                1 + 
                (volume_factor * 0.3) + 
                (oi_momentum * 0.25) + 
                (price_momentum * 0.2) + 
                (iv_factor * 0.15) +
                (mfi_factor * 0.1)
            )
            
            return activity_score

        # Apply enhanced activity metric
        log_df['Activity_Metric'] = log_df.apply(calculate_enhanced_activity_metric, axis=1)
        
        # NEW: Dark Pool Activity Detection
        def detect_dark_pool_activity(row):
            """Detect potential dark pool or iceberg orders"""
            volume = row.get('Volume', 1)
            oi_change = row['OI_Change_Pct']
            ltp_change = row['LTP_Change_Pct']
            
            # Large OI increase with minimal price impact suggests stealth trading
            if oi_change > 5 and abs(ltp_change) < 2 and volume > 100:
                return True
            return False
        
        log_df['Dark_Pool_Activity'] = log_df.apply(detect_dark_pool_activity, axis=1)
        
        # Sentiment analysis with enhanced logic
        spot_prices = log_df.groupby('Timestamp')['UnderlyingValue'].first().reset_index()
        spot_prices['Price_Change'] = spot_prices['UnderlyingValue'].pct_change() * 100
        spot_prices['Price_Velocity'] = spot_prices['Price_Change'].diff()  # Rate of change acceleration
        
        # Enhanced sentiment with velocity consideration
        def enhanced_sentiment(row):
            price_change = row['Price_Change']
            velocity = row.get('Price_Velocity', 0)
            
            if price_change > 0.1 and velocity > 0:
                return 'Strong Bullish'
            elif price_change > 0.05:
                return 'Bullish'
            elif price_change < -0.1 and velocity < 0:
                return 'Strong Bearish'
            elif price_change < -0.05:
                return 'Bearish'
            else:
                return 'Neutral'
        
        spot_prices['Enhanced_Sentiment'] = spot_prices.apply(enhanced_sentiment, axis=1)
        
        # Merge enhanced sentiment
        log_df = log_df.merge(spot_prices[['Timestamp', 'Enhanced_Sentiment']], on='Timestamp', how='left')
        
        # Enhanced classification with more nuanced logic
        def classify_enhanced_big_money_flow(row):
            """
            Enhanced classification with dark pool detection and MFI
            """
            activity = row['Activity_Metric']
            oi_change_pct = row['OI_Change_Pct']
            ltp_change_pct = row['LTP_Change_Pct']
            iv_change_pct = row['IV_Change_Pct']
            volume = row.get('Volume', 1)
            sentiment = row.get('Enhanced_Sentiment', 'Neutral')
            option_type = row['Type']
            mfi = row.get('MFI', 50)
            dark_pool = row.get('Dark_Pool_Activity', False)
            
            # Dynamic thresholds based on market conditions
            activity_percentile = np.percentile(log_df['Activity_Metric'], 65)
            # Calculate dynamic OI change threshold based on current data
            oi_change_values = log_df['OI_Change_Pct'].replace([np.inf, -np.inf], np.nan).dropna()
            significant_oi_change = np.percentile(oi_change_values, 75) if not oi_change_values.empty else 5

            significant_volume = 75
            
            # Dark pool activity gets special treatment
            if dark_pool:
                return f"Dark Pool {option_type}", "#800080", activity * 1.5  # Purple for stealth
            
            # Only classify if activity exceeds threshold
            if activity < activity_percentile:
                return "Low Activity", "#333333", activity * 0.1
            
            if option_type == "CE":  # Call Options
                # Aggressive Call Buying (MFI > 70)
                if (mfi > 70 and volume > significant_volume and 
                    ltp_change_pct > 2 and sentiment in ['Bullish', 'Strong Bullish']):
                    return "Aggressive Call Buy", "#006400", activity * 1.2  # Dark Green
                
                # Call Short with high conviction
                elif (oi_change_pct > significant_oi_change and 
                    ltp_change_pct < -2 and mfi < 30 and
                    volume > significant_volume):
                    return "Heavy Call Short", "#DC143C", activity * 1.1  # Crimson
                
                # Standard Call Buy
                elif (volume > significant_volume and 
                    ltp_change_pct > 1 and 
                    (sentiment in ['Bullish', 'Strong Bullish'] or iv_change_pct > 3)):
                    return "Call Buy", "#2E8B57", activity
                
                # Standard Call Short
                elif (oi_change_pct > significant_oi_change and 
                    ltp_change_pct < -1 and volume > significant_volume):
                    return "Call Short", "#FF6B6B", activity
                
                else:
                    return "Call Activity", "#90EE90", activity * 0.7
                    
            elif option_type == "PE":  # Put Options
                # Aggressive Put Buying (Fear-driven)
                if (mfi < 30 and volume > significant_volume and 
                    ltp_change_pct > 2 and sentiment in ['Bearish', 'Strong Bearish']):
                    return "Aggressive Put Buy", "#8B0000", activity * 1.2  # Dark Red
                
                # Put Write with high conviction
                elif (oi_change_pct > significant_oi_change and 
                    ltp_change_pct < -2 and mfi > 70 and
                    volume > significant_volume):
                    return "Heavy Put Write", "#228B22", activity * 1.1  # Forest Green
                
                # Standard Put Buy
                elif (volume > significant_volume and 
                    ltp_change_pct > 1 and 
                    (sentiment in ['Bearish', 'Strong Bearish'] or iv_change_pct > 3)):
                    return "Put Buy", "#8B0000", activity
                
                # Standard Put Write
                elif (oi_change_pct > significant_oi_change and 
                    ltp_change_pct < -1 and volume > significant_volume):
                    return "Put Write", "#90EE90", activity
                
                else:
                    return "Put Activity", "#FFB6C1", activity * 0.7
            
            return "Neutral", "#666666", activity * 0.3

        # Apply enhanced classification
        classification = log_df.apply(lambda row: pd.Series(classify_enhanced_big_money_flow(row)), axis=1)
        log_df[['Flow_Type', 'Color', 'Weighted_Activity']] = classification

        # Time-based aggregation with better granularity
        log_df['TimeSlot'] = pd.to_datetime(log_df['Timestamp']).dt.floor('1min').dt.strftime('%H:%M')
        
        # Enhanced aggregation
        flow_agg = log_df.groupby(['TimeSlot', 'Flow_Type', 'Color']).agg({
            'Weighted_Activity': 'sum',
            'UnderlyingValue': 'last',
            'Volume': 'sum',
            'MFI': 'mean'
        }).reset_index()
        
    
        
        # Price data with MFI
        price_data = log_df.groupby('TimeSlot').agg({
            'UnderlyingValue': 'last',
            'MFI': 'mean'
        }).reset_index()

        # Create enhanced TradingView-style triple pane chart
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.1, 0.3],  # More height for Spot Price
            subplot_titles=(
                "📈 Spot Price Movement", 
                "💹 Money Flow Index (MFI)",
                "🔍 Enhanced HFT Algo Scanner"
            )
        )

        # Upper pane: Price chart
        fig.add_trace(
            go.Scatter(
                x=price_data['TimeSlot'],
                y=price_data['UnderlyingValue'],
                mode='lines',
                name='Spot Price',
                line=dict(color='#00BFFF', width=2),
                hovertemplate="<b>%{x}</b><br>Price: ₹%{y:,.2f}<extra></extra>"
            ),
            row=1, col=1
        )

        # Middle pane: MFI
        fig.add_trace(
            go.Scatter(
                x=price_data['TimeSlot'],
                y=price_data['MFI'],
                mode='lines',
                name='MFI',
                line=dict(color='#FFD700', width=2),
                fill='tonexty',
                fillcolor='rgba(255, 215, 0, 0.1)',
                hovertemplate="<b>%{x}</b><br>MFI: %{y:.1f}<extra></extra>"
            ),
            row=2, col=1
        )

        # Add MFI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

        # Lower pane: Enhanced flow bars
        flow_types = flow_agg['Flow_Type'].unique()

        excluded_flows = ['Call Activity', 'Put Activity', 'Call Buy', 'Put Buy', 'Low Activity']

        for flow_type in flow_types:
            if flow_type in excluded_flows:
                continue  # Skip excluded flow types

            flow_data = flow_agg[flow_agg['Flow_Type'] == flow_type]
            if not flow_data.empty:
                color = flow_data['Color'].iloc[0]
                fig.add_trace(
                    go.Bar(
                        x=flow_data['TimeSlot'],
                        y=flow_data['Weighted_Activity'],
                        name=flow_type,
                        marker_color=color,
                        opacity=0.8,
                        hovertemplate=f"<b>{flow_type}</b><br>Time: %{{x}}<br>Flow: %{{y:,.0f}}<br>Avg MFI: %{{customdata:.1f}}<extra></extra>",
                        customdata=flow_data['MFI']
                    ),
                    row=3, col=1
                )


        # Apply no scroll bar, allow pan
        for row in [1, 2, 3]:
            fig.update_xaxes(
                rangeslider_visible=False,  # ❌ remove scroll
                fixedrange=False,           # ✅ allow drag scroll
                showgrid=False,
                zeroline=False,
                color='black',
                row=row, col=1
            )
            fig.update_yaxes(
                showgrid=False,
                zeroline=False,
                color='black',
                row=row, col=1
            )

        # Specific Y axis titles
        fig.update_yaxes(title_text="Price (₹)", title_font=dict(size=12), row=1, col=1)
        fig.update_yaxes(title_text="MFI", title_font=dict(size=12), row=2, col=1)
        fig.update_yaxes(title_text="Big Money Flow", title_font=dict(size=12), row=3, col=1)

        # Specific X axis title
        fig.update_xaxes(title_text="Time", title_font=dict(size=12), row=3, col=1)

        # Layout config
        fig.update_layout(
            title=dict(
                text="🚀 Enhanced HFT Algo Scanner – Institutional Flow Detection with MFI",
                x=0.5,
                font=dict(size=18, family="Arial Black", color="black")
            ),
            barmode='stack',
            height=900,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=11, family='Arial'),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.95,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=11)
            )
        )

        # Final rendering
        st.plotly_chart(fig, use_container_width=True, config={
            "scrollZoom": True,
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["zoomIn2d", "zoomOut2d", "autoScale2d"],
            "modeBarButtonsToAdd": ["pan2d"]
        })

        
        # 🎯 Writing Flows Only Visualization
        
        # Enhanced Analysis Dashboard
        st.subheader("📊 Enhanced Big Money Flow Analysis")
        
        # Calculate enhanced metrics
        flow_totals = flow_agg.groupby('Flow_Type')['Weighted_Activity'].sum().reset_index()
        dark_pool_count = len(log_df[log_df['Dark_Pool_Activity'] == True])
        avg_mfi = log_df['MFI'].mean()
        
        # Top row metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🎯 Dark Pool Signals", dark_pool_count)
        
        with col2:
            mfi_status = "Overbought" if avg_mfi > 70 else "Oversold" if avg_mfi < 30 else "Neutral"
            st.metric("💹 Avg MFI", f"{avg_mfi:.1f}", delta=mfi_status)
        
        with col3:
            total_volume = log_df.get('Volume', 0).sum()
            st.metric("📊 Total Volume", f"{total_volume:,.0f}")
        
        with col4:
            aggressive_flows = len(flow_agg[flow_agg['Flow_Type'].str.contains('Aggressive|Heavy', na=False)])
            st.metric("⚡ Aggressive Flows", aggressive_flows)
        
        with col5:
            latest_sentiment = log_df['Enhanced_Sentiment'].iloc[-1] if not log_df.empty else 'Neutral'
            st.metric("🎭 Market Sentiment", latest_sentiment)

        # Flow type breakdown
        st.subheader("🔍 Flow Type Breakdown")
        
        # Exclude unwanted flow types
        excluded_flows = ['Call Activity', 'Put Activity', 'Call Buy', 'Put Buy' , 'Low Activity']
        filtered_totals = flow_totals[~flow_totals['Flow_Type'].isin(excluded_flows)]

        # Display top flow types
        top_flows = filtered_totals.nlargest(6, 'Weighted_Activity')

        
        cols = st.columns(3)
        for i, (_, row) in enumerate(top_flows.iterrows()):
            if i < 6:
                col_idx = i % 3
                with cols[col_idx]:
                    # Determine emoji based on flow type
                    if 'Call' in row['Flow_Type'] and 'Buy' in row['Flow_Type']:
                        emoji = '🟢'
                    elif 'Put' in row['Flow_Type'] and 'Buy' in row['Flow_Type']:
                        emoji = '🔴'
                    elif 'Dark Pool' in row['Flow_Type']:
                        emoji = '🟣'
                    elif 'Aggressive' in row['Flow_Type'] or 'Heavy' in row['Flow_Type']:
                        emoji = '⚡'
                    else:
                        emoji = '📊'
                    
                    st.metric(
                        f"{emoji} {row['Flow_Type']}", 
                        f"{row['Weighted_Activity']:,.0f}"
                    )

        # Advanced Analytics
        with st.expander("🔬 Advanced Analytics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Market Microstructure Analysis:**")
                
                # Calculate institutional vs retail bias
                institutional_flows = ['Aggressive Call Buy', 'Heavy Call Short', 'Aggressive Put Buy', 'Heavy Put Write', 'Dark Pool CE', 'Dark Pool PE']
                institutional_activity = flow_totals[flow_totals['Flow_Type'].isin(institutional_flows)]['Weighted_Activity'].sum()
                total_activity = flow_totals['Weighted_Activity'].sum()
                
                institutional_pct = (institutional_activity / total_activity * 100) if total_activity > 0 else 0
                
                st.write(f"- Institutional Activity: {institutional_pct:.1f}%")
                st.write(f"- Retail Activity: {100-institutional_pct:.1f}%")
                st.write(f"- Dark Pool Penetration: {(dark_pool_count/len(log_df)*100):.1f}%")
            
            with col2:
                st.write("**Risk Indicators:**")
                
                # Calculate risk metrics
                put_call_ratio = flow_totals[flow_totals['Flow_Type'].str.contains('Put', na=False)]['Weighted_Activity'].sum() / \
                            (flow_totals[flow_totals['Flow_Type'].str.contains('Call', na=False)]['Weighted_Activity'].sum() + 1)
                
                volatility_demand = len(log_df[abs(log_df['IV_Change_Pct']) > 5]) / len(log_df) * 100
                
                st.write(f"- Put/Call Ratio: {put_call_ratio:.2f}")
                st.write(f"- Volatility Demand: {volatility_demand:.1f}%")
                
                if put_call_ratio > 1.2:
                    st.error("⚠️ High Fear Index")
                elif put_call_ratio < 0.8:
                    st.success("😊 Low Fear Index")
                else:
                    st.info("😐 Moderate Fear Index")

    else:
        st.error("❌ No data available. Please ensure strike_sentiment_log contains options data.")
    
   
    # Telegram for ATM zone
    atm_summary = f"📊 *ATM ±5 Avg Sentiment:*\n• Call: {avg_sentiment_ce:+.2f}\n• Put: {avg_sentiment_pe:+.2f}"
    send_telegram_alert(atm_summary)

    with st.expander("📊 Net Writer vs Buyer Strength by Strike (Split by CE/PE)"):
        grouped = df.groupby(['StrikePrice', 'Type', 'SentimentBias'])['SentimentScore'].sum().reset_index()

        def filter_bias(df, bias_keyword):
            return df[df['SentimentBias'].str.contains(bias_keyword, case=False, na=False)]

        ce_buy = filter_bias(grouped[grouped['Type'] == 'CE'], 'Buying')
        ce_write = filter_bias(grouped[grouped['Type'] == 'CE'], 'Writing')
        pe_buy = filter_bias(grouped[grouped['Type'] == 'PE'], 'Buying')
        pe_write = filter_bias(grouped[grouped['Type'] == 'PE'], 'Writing')

        fig = go.Figure()
        fig.add_trace(go.Bar(x=ce_buy['StrikePrice'], y=ce_buy['SentimentScore'], name="CE Buy", marker_color="green"))
        fig.add_trace(go.Bar(x=ce_write['StrikePrice'], y=ce_write['SentimentScore'], name="CE Write", marker_color="red"))
        fig.add_trace(go.Bar(x=pe_buy['StrikePrice'], y=pe_buy['SentimentScore'], name="PE Buy", marker_color="lightgreen"))
        fig.add_trace(go.Bar(x=pe_write['StrikePrice'], y=pe_write['SentimentScore'], name="PE Write", marker_color="orange"))

        fig.update_layout(
            title="CE/PE Buyer and Writer Strength by Strike",
            xaxis_title="Strike Price",
            yaxis_title="Sentiment Score",
            barmode="group",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)
 
    # Compare with previous data and apply indicators
    for _, row in df.iterrows():
        strike_price = row["StrikePrice"]
        opt_type = row["Type"]
        iv = row["IV"]
        oi = row["OI"]
        ltp = row["LTP"]
        delta = row["Delta"]
        gamma = row["Gamma"]
        theta = row["Theta"]
        vega = row["Vega"]
        key = f"{strike_price}_{opt_type}"

        # Fetch previous data
        prev = previous_data.get(key, {})
        prev_iv = prev.get("IV", iv)
        prev_oi = prev.get("OI", oi)

        # IV and OI Change
        iv_change = ((iv - prev_iv) / prev_iv) * 100 if prev_iv else 0
        oi_change = ((oi - prev_oi) / prev_oi) * 100 if prev_oi else 0
        price_change = abs((underlying_price - prev_underlying_price) / prev_underlying_price * 100) if prev_underlying_price else 0

        # STRONG BREAKOUT ALERT
        if opt_type == "CE" and iv_change > IV_SPIKE_THRESHOLD and oi_change > OI_SPIKE_THRESHOLD and delta > 0.75:
            alerts.append(f"🔥 STRONG BREAKOUT (CALLS): IV + OI + Delta Confirmed!\nStrike: {strike_price} | IV: {iv_change:.2f}% | OI: {oi_change:.2f}% | Delta: {delta:.2f}")

        if opt_type == "PE" and iv_change > IV_SPIKE_THRESHOLD and oi_change > OI_SPIKE_THRESHOLD and delta < -0.75:
            alerts.append(f"🔥 STRONG BREAKOUT (PUTS): IV + OI + Delta Confirmed!\nStrike: {strike_price} | IV: {iv_change:.2f}% | OI: {oi_change:.2f}% | Delta: {delta:.2f}")

        # HIGH GAMMA ALERT
        if gamma > GAMMA_THRESHOLD:
            alerts.append(f"⚡ HIGH GAMMA: Big Move Incoming!\nStrike: {strike_price} | {opt_type}_Gamma: {gamma:.4f}")

        # HIGH TIME DECAY ALERT
        if theta < THETA_DECAY_THRESHOLD:
            alerts.append(f"⏳ HIGH TIME DECAY: Risk for Long Options!\nStrike: {strike_price} | {opt_type}_Theta: {theta:.2f}")

        # IV CRASH ALERT
        if iv_change < -IV_CRASH_THRESHOLD:
            alerts.append(f"🔥 IV CRASH ALERT: Sudden drop in IV!\nStrike: {strike_price} | {opt_type}_IV: {iv_change:.2f}%")

        # OI SURGE ALERT
        if oi_change > OI_SPIKE_THRESHOLD:
            alerts.append(f"🚀 OI SURGE ALERT: Institutional buying/selling!\nStrike: {strike_price} | {opt_type}_OI: {oi_change:.2f}%")

        # IV Rising but Price Stable Alert
        if iv_change > IV_SPIKE_THRESHOLD and price_change < PRICE_STABILITY_THRESHOLD:
            alerts.append(f"📈 IV RISING BUT PRICE STABLE: Expect Big Move Soon!\nStrike: {strike_price} | {opt_type}_IV: {iv_change:.2f}% | Price Change: {price_change:.2f}%")

        # SHORT SQUEEZE ALERT
        if iv_change > IV_SPIKE_THRESHOLD and oi_change > OI_SPIKE_THRESHOLD:
            alerts.append(f"🛑 SHORT SQUEEZE RISK: IV & OI surging together!\nStrike: {strike_price} | {opt_type}_IV: {iv_change:.2f}% | {opt_type}_OI: {oi_change:.2f}%")

        # CALLS DOMINATING
        if opt_type == "CE" and iv > df[df["Type"] == "PE"]["IV"].max():
            alerts.append(f"🟢 CALLS DOMINATING: Bullish sentiment detected!\nStrike: {strike_price} | CE_IV: {iv:.2f} (Change: {iv_change:.2f})")

        # PUTS DOMINATING
        if opt_type == "PE" and iv > df[df["Type"] == "CE"]["IV"].max():
            alerts.append(f"🔴 PUTS DOMINATING: Bearish sentiment detected!\nStrike: {strike_price} | PE_IV: {iv:.2f} (Change: {iv_change:.2f})")
        # STRADDLE TRIGGER ALERT
        if iv_change > IV_SPIKE_THRESHOLD and df[df["StrikePrice"] == strike_price]["IV"].max() > IV_SPIKE_THRESHOLD:
            alerts.append(f"💥 STRADDLE TRIGGER ALERT: Both CE & PE IV rising sharply!\nStrike: {strike_price}")

        # Directional Bias
        if iv_change > 5 and oi_change > 10 and delta > 0.75 and price_change > 0.5:
            alerts.append(f"📈 DIRECTIONAL BIAS: Bullish Setup!\nStrike: {strike_price}")
        elif iv_change > 5 and oi_change > 10 and delta < -0.75 and price_change < -0.5:
            alerts.append(f"📉 DIRECTIONAL BIAS: Bearish Setup!\nStrike: {strike_price}")

        # Reversal Setup Detection
        if iv_change < -IV_CRASH_THRESHOLD and oi_change > OI_SPIKE_THRESHOLD and abs(gamma) < 0.015:
            alerts.append(f"🌀 REVERSAL SETUP: IV drop with OI rise and low gamma!\nStrike: {strike_price}")

        # Save latest values
        previous_data[key] = {"IV": iv, "OI": oi, "LTP": ltp, "Vega": vega, "Theta": theta, "Delta": delta, "Gamma": gamma}


    previous_data["underlying_price"] = underlying_price  # Save underlying price for next cycle
    
    # Save session state after updating previous_data
    save_session_state()

    # After your existing analysis, add these enhancements:
    
    # Get basic df first (your existing logic)
    net_oi_df = df.pivot(index='StrikePrice', columns='Type', values='OI').fillna(0)
    net_oi_df['NetOI'] = net_oi_df['PE'] - net_oi_df['CE']

    underlying_price = option_chain["data"]["last_price"]
    
    enhanced_alerts = []
    
    # 1. Market Microstructure Analysis
    micro_alerts, micro_signals = analyze_market_microstructure(df, underlying_price)
    enhanced_alerts.extend(micro_alerts)
    
    # 2. Market Timing Intelligence
    timing_alerts, timing_signals = get_market_timing_signals()
    enhanced_alerts.extend(timing_alerts)
    
    # 3. Advanced PCR Analysis
    pcr_alerts, pcr_signals = advanced_pcr_analysis(df, underlying_price)
    enhanced_alerts.extend(pcr_alerts)
    
    # 4. Breakout Prediction
    breakout_alerts, breakout_signals = predict_breakout_probability(df, underlying_price, micro_signals)
    enhanced_alerts.extend(breakout_alerts)
    
    # 5. Enhanced Momentum Analysis
    rolling_data = st.session_state.rolling_data
    for _, row in df.iterrows():
        momentum_indicators = calculate_momentum_indicators(
            rolling_data, row['StrikePrice'], row['Type']
        )
        
        # Generate momentum-based alerts
        if 'iv_momentum' in momentum_indicators and momentum_indicators['iv_momentum'] > 10:
            enhanced_alerts.append(f"📈 IV MOMENTUM SURGE: {row['StrikePrice']} {row['Type']}\n"
                                  f"IV Rate of Change: {momentum_indicators['iv_momentum']:.2f}%")
        
        if 'oi_acceleration' in momentum_indicators and momentum_indicators['oi_acceleration'] > 100000:
            enhanced_alerts.append(f"🚀 OI ACCELERATION: {row['StrikePrice']} {row['Type']}\n"
                                  f"OI building up rapidly")
    
    # Combine all signals
    enhanced_signals = {
        'microstructure': micro_signals,
        'timing': timing_signals,
        'pcr_analysis': pcr_signals,
        'breakout_prediction': breakout_signals
    }
    
    # Add enhanced alerts to session state
    if enhanced_alerts:
        MAX_ALERTS = 20
        combined_alerts = enhanced_alerts + st.session_state.alerts
        st.session_state.alerts = combined_alerts[:MAX_ALERTS]
        save_session_state()  # Save after updating alerts

        send_unique_telegram_alerts(enhanced_alerts)
    
    # 🔍 Summary of Flows by Strike (Latest Snapshot)
    with st.expander("📋 Summary of Flows by Strike (Latest Snapshot)", expanded=False):
        if 'Flow_Type' in log_df.columns:
            summary_df = log_df.groupby(['StrikePrice', 'Type']).agg({
                'Flow_Type': 'last',
                'SentimentScore': 'last'
            }).unstack(fill_value='-')

            # Flatten MultiIndex columns
            summary_df.columns = ['CE_Flow', 'PE_Flow', 'CE_Score', 'PE_Score']
            summary_df = summary_df.reset_index().sort_values(by="StrikePrice")

            # Optional coloring function
            def highlight_flows(val):
                if "Buy" in str(val):
                    return 'background-color: lightgreen'
                elif "Write" in str(val) or "Short" in str(val):
                    return 'background-color: lightcoral'
                return ''

            styled = summary_df.style.applymap(highlight_flows, subset=['CE_Flow', 'PE_Flow'])
            st.dataframe(styled, use_container_width=True)
        else:
            st.info("🕵️ Flow_Type data not available yet – please wait for updates or rerun the app.")



        
    save_to_csv(df)
    
    # Final save to ensure all data is persisted
    save_session_state()
    
    return df, net_oi_df, enhanced_signals


# 🔹 Enhanced expiry day analysis function - add this after your analyze_data function
def expiry_day_analysis(df, underlying_price, expiry_date):
    """Advanced expiry day analysis for reversal prediction"""
    
    if not is_expiry_day(expiry_date):
        return [], {}
    
    alerts = []
    expiry_signals = {}
    
    # 1. PIN RISK ANALYSIS - Detect strikes with massive OI concentration
    strike_oi = df.groupby('StrikePrice')['OI'].sum().reset_index()
    max_oi_strike = strike_oi.loc[strike_oi['OI'].idxmax()]
    
    if max_oi_strike['OI'] > PIN_RISK_THRESHOLD:
        distance_to_pin = abs(underlying_price - max_oi_strike['StrikePrice'])
        pin_probability = max(0, 100 - (distance_to_pin / 10))  # Rough pin probability
        
        alerts.append(f"📌 PIN RISK ALERT: Massive OI at {max_oi_strike['StrikePrice']}\n"
                     f"OI: {max_oi_strike['OI']/1000000:.1f}M | Pin Probability: {pin_probability:.0f}%\n"
                     f"Distance: {distance_to_pin:.0f} points")
        
        expiry_signals['pin_strike'] = max_oi_strike['StrikePrice']
        expiry_signals['pin_probability'] = pin_probability
    
    # 2. GAMMA WALL DETECTION - Find strikes with extreme gamma concentration
    ce_gamma = df[df['Type'] == 'CE'].groupby('StrikePrice')['Gamma'].sum()
    pe_gamma = df[df['Type'] == 'PE'].groupby('StrikePrice')['Gamma'].sum()
    total_gamma = ce_gamma.add(pe_gamma, fill_value=0)
    
    gamma_wall_strike = total_gamma.idxmax()
    max_gamma = total_gamma.max()
    
    if max_gamma > GAMMA_SQUEEZE_THRESHOLD:
        wall_direction = "above" if underlying_price < gamma_wall_strike else "below"
        alerts.append(f"🧱 GAMMA WALL at {gamma_wall_strike}\n"
                     f"Total Gamma: {max_gamma:.4f} | Price is {wall_direction} wall\n"
                     f"Expect price to be pulled toward {gamma_wall_strike}")
        
        expiry_signals['gamma_wall'] = gamma_wall_strike
        expiry_signals['gamma_strength'] = max_gamma
    
    # 3. VOLATILITY CRUSH DETECTION
    avg_iv = df['IV'].mean()
    if 'previous_avg_iv' in st.session_state:
        iv_drop = st.session_state.previous_avg_iv - avg_iv
        if iv_drop > VOLATILITY_CRUSH_THRESHOLD:
            alerts.append(f"💥 VOLATILITY CRUSH: IV collapsed by {iv_drop:.1f}%\n"
                         f"Current Avg IV: {avg_iv:.1f}% | Previous: {st.session_state.previous_avg_iv:.1f}%\n"
                         f"Long options severely impacted!")
    
    st.session_state.previous_avg_iv = avg_iv
    save_session_state()  # Save after updating previous_avg_iv
    
    # 4. THETA BURN ACCELERATION
    avg_theta = df['Theta'].mean()
    if avg_theta < EXPIRY_DAY_THETA_ACCELERATION:
        alerts.append(f"⏰ EXTREME THETA BURN: {avg_theta:.2f}\n"
                     f"Time decay accelerating rapidly - Avoid long positions!")
    
    # 5. DELTA NEUTRAL LEVEL CALCULATION
    ce_delta_oi = (df[df['Type'] == 'CE']['Delta'] * df[df['Type'] == 'CE']['OI']).sum()
    pe_delta_oi = (df[df['Type'] == 'PE']['Delta'] * df[df['Type'] == 'PE']['OI']).sum()
    total_delta_oi = ce_delta_oi + pe_delta_oi
    
    if abs(total_delta_oi) > 1000000:  # Significant delta imbalance
        bias = "Bullish" if total_delta_oi > 0 else "Bearish"
        alerts.append(f"⚖️ DELTA IMBALANCE: {bias} bias detected\n"
                     f"Net Delta*OI: {total_delta_oi/1000000:.1f}M\n"
                     f"Market makers likely to hedge by {'buying' if total_delta_oi > 0 else 'selling'}")
    
    # 6. SUPPORT/RESISTANCE FLIP ZONES
    high_oi_strikes = strike_oi[strike_oi['OI'] > strike_oi['OI'].quantile(0.8)]['StrikePrice'].tolist()
    nearby_strikes = [s for s in high_oi_strikes if abs(s - underlying_price) <= 100]
    
    if nearby_strikes:
        for strike in nearby_strikes:
            position = "above" if underlying_price > strike else "below"
            alerts.append(f"🔄 KEY LEVEL: {strike} (Price is {position})\n"
                         f"High OI zone - Watch for support/resistance flip")
    
    # 7. EXPIRY REVERSAL PREDICTION MODEL
    reversal_score = 0
    reversal_factors = []
    
    # Factor 1: Distance from max pain
    max_pain = strike_oi.loc[strike_oi['OI'].idxmax(), 'StrikePrice']
    distance_from_max_pain = abs(underlying_price - max_pain)
    if distance_from_max_pain > 50:
        reversal_score += min(20, distance_from_max_pain / 5)
        reversal_factors.append(f"Distance from Max Pain ({max_pain}): +{min(20, distance_from_max_pain/5):.0f}")
    
    # Factor 2: OI concentration
    oi_concentration = max_oi_strike['OI'] / strike_oi['OI'].sum()
    if oi_concentration > 0.3:
        reversal_score += 25
        reversal_factors.append(f"High OI Concentration: +25")
    
    # Factor 3: Gamma exposure
    if max_gamma > GAMMA_SQUEEZE_THRESHOLD:
        reversal_score += 20
        reversal_factors.append(f"High Gamma Exposure: +20")
    
    # Factor 4: Time to expiry (last 2 hours get max score)
    current_time = datetime.now(IST).time()
    if current_time >= datetime.strptime("14:30", "%H:%M").time():  # After 2:30 PM
        reversal_score += 15
        reversal_factors.append("Late expiry session: +15")
    
    if reversal_score > 40:
        direction = "toward Max Pain" if distance_from_max_pain > 30 else "away from current level"
        alerts.append(f"🎯 EXPIRY REVERSAL SIGNAL (Score: {reversal_score:.0f}/100)\n"
                     f"High probability reversal {direction}\n"
                     f"Factors: {'; '.join(reversal_factors)}")
        
        expiry_signals['reversal_score'] = reversal_score
        expiry_signals['reversal_direction'] = direction
    
    return alerts, expiry_signals

def create_comprehensive_analysis(df, underlying_price):
    """Create comprehensive analysis similar to the artifact"""
    
    # Calculate key metrics
    ce_df = df[df['Type'] == 'CE'].copy()
    pe_df = df[df['Type'] == 'PE'].copy()
    
    total_ce_oi = ce_df['OI'].sum()
    total_pe_oi = pe_df['OI'].sum()
    put_call_ratio = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    
    # Find max pain (strike with highest total OI)
    strike_oi = df.groupby('StrikePrice')['OI'].sum().reset_index()
    max_pain_strike = strike_oi.loc[strike_oi['OI'].idxmax(), 'StrikePrice']
    max_pain_oi = strike_oi['OI'].max()
    
    # Estimate current spot based on delta (find ATM)
    atm_ce = ce_df[abs(ce_df['Delta'] - 0.5) < 0.05]  # Tighter range
    estimated_spot = atm_ce['StrikePrice'].iloc[0] if len(atm_ce) > 0 else underlying_price
    
    return {
        'total_ce_oi': total_ce_oi / 1000000,
        'total_pe_oi': total_pe_oi / 1000000, 
        'put_call_ratio': put_call_ratio,
        'max_pain_strike': max_pain_strike,
        'max_pain_oi': max_pain_oi / 1000000,
        'estimated_spot': estimated_spot
    }

# 🔹 Add this function to create expiry day dashboard
def render_expiry_dashboard(df, underlying_price, expiry_date, expiry_signals):
    """Render expiry analysis dashboard - always visible"""
    
    st.markdown("---")
    
    # Dynamic title based on expiry status
    if is_expiry_day(expiry_date):
        st.subheader("🏁 EXPIRY DAY SPECIAL ANALYSIS")
        st.markdown("⚠️ **TODAY IS EXPIRY DAY - ENHANCED FEATURES ACTIVE**")
    else:
        days_to_expiry = (datetime.strptime(expiry_date, "%Y-%m-%d").date() - datetime.now(IST).date()).days
        st.subheader(f"📅 EXPIRY ANALYSIS ({days_to_expiry} days to expiry)")
    
    # Time to expiry countdown (always show)
    # Time to expiry countdown (always show)
    expiry_time = datetime.strptime(f"{expiry_date} 15:30", "%Y-%m-%d %H:%M").replace(tzinfo=IST)
    current_time = datetime.now(IST)
    time_left = expiry_time - current_time

    
    if time_left.total_seconds() > 0:
        days_left = time_left.days
        hours_left = int((time_left.total_seconds() % 86400) // 3600)
        minutes_left = int((time_left.total_seconds() % 3600) // 60)
        st.markdown(f"⏰ **Time to Expiry: {days_left}d {hours_left}h {minutes_left}m**")
    else:
        st.markdown("🏁 **EXPIRED**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Pin Risk Meter (enhanced on expiry day)
        if is_expiry_day(expiry_date) and 'pin_probability' in expiry_signals:
            st.metric(
                "Pin Risk", 
                f"{expiry_signals['pin_probability']:.0f}%",
                help=f"Probability of pinning at {expiry_signals.get('pin_strike', 'N/A')}"
            )
        else:
            # Calculate basic pin risk for non-expiry days
            strike_oi = df.groupby('StrikePrice')['OI'].sum().reset_index()
            max_oi_strike = strike_oi.loc[strike_oi['OI'].idxmax()]
            distance_to_pin = abs(underlying_price - max_oi_strike['StrikePrice'])
            basic_pin_risk = max(0, 100 - (distance_to_pin / 20))  # Less sensitive for non-expiry
            st.metric("Pin Risk", f"{basic_pin_risk:.0f}%", help="Basic pin risk calculation")
    
    with col2:
        # Reversal Score (enhanced on expiry day)
        if is_expiry_day(expiry_date) and 'reversal_score' in expiry_signals:
            st.metric(
                "Reversal Score", 
                f"{expiry_signals['reversal_score']:.0f}/100",
                help="Higher score = Higher reversal probability"
            )
        else:
            # Basic reversal indicator for non-expiry days
            strike_oi = df.groupby('StrikePrice')['OI'].sum().reset_index()
            max_pain = strike_oi.loc[strike_oi['OI'].idxmax(), 'StrikePrice']
            distance_from_max_pain = abs(underlying_price - max_pain)
            basic_reversal = min(100, distance_from_max_pain / 2)
            st.metric("Reversal Potential", f"{basic_reversal:.0f}/100", help="Distance from Max Pain indicator")
    
    with col3:
        # Gamma Wall (always calculate)
        ce_gamma = df[df['Type'] == 'CE'].groupby('StrikePrice')['Gamma'].sum()
        pe_gamma = df[df['Type'] == 'PE'].groupby('StrikePrice')['Gamma'].sum()
        total_gamma = ce_gamma.add(pe_gamma, fill_value=0)
        
        if len(total_gamma) > 0:
            gamma_wall_strike = total_gamma.idxmax()
            max_gamma = total_gamma.max()
            st.metric(
                "Gamma Wall", 
                f"{gamma_wall_strike:.0f}",
                help=f"Strike with highest gamma: {max_gamma:.4f}"
            )
        else:
            st.metric("Gamma Wall", "None")
    
    # Always show key levels chart
    st.subheader("🎯 Key Trading Levels")
    
    strike_oi = df.groupby('StrikePrice')['OI'].sum().reset_index()
    max_pain = strike_oi.loc[strike_oi['OI'].idxmax(), 'StrikePrice']
    
    fig_zones = go.Figure()
    
    # Add OI bars
    fig_zones.add_trace(go.Bar(
        x=strike_oi['StrikePrice'],
        y=strike_oi['OI'] / 1000000,
        name='Total OI (M)',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # Mark current price
    fig_zones.add_vline(
        x=underlying_price, 
        line_dash="dash", 
        line_color="blue",
        annotation_text="Current Price"
    )
    
    # Mark max pain
    fig_zones.add_vline(
        x=max_pain, 
        line_dash="dot", 
        line_color="red",
        annotation_text="Max Pain"
    )
    
    # Mark gamma wall if exists
    if len(total_gamma) > 0:
        fig_zones.add_vline(
            x=gamma_wall_strike, 
            line_dash="dashdot", 
            line_color="orange",
            annotation_text="Gamma Wall"
        )
    
    title_suffix = " (EXPIRY DAY)" if is_expiry_day(expiry_date) else f" ({days_to_expiry} days to expiry)"
    fig_zones.update_layout(
        title=f"Key Levels{title_suffix}",
        xaxis_title="Strike Price",
        yaxis_title="OI (Millions)",
        height=400
    )
    
    st.plotly_chart(fig_zones, use_container_width=True)
    
    # Enhanced strategy section on expiry day
    if is_expiry_day(expiry_date):
        st.subheader("📋 EXPIRY DAY Strategy Hints")
        
        strategy_col1, strategy_col2 = st.columns(2)
        
        with strategy_col1:
            st.markdown("**🟢 Bullish Scenarios:**")
            st.write("• Price below Max Pain → Likely to move up")
            st.write("• Heavy Put OI above current price")
            st.write("• Positive delta imbalance")
            
        with strategy_col2:
            st.markdown("**🔴 Bearish Scenarios:**")
            st.write("• Price above Max Pain → Likely to move down")
            st.write("• Heavy Call OI above current price")
            st.write("• Negative delta imbalance")
        
        st.markdown("**⚠️ EXPIRY DAY RISKS:**")
        st.write("• Avoid long options in last 2 hours (extreme theta)")
        st.write("• Pin risk highest between 3:00-3:30 PM")
        st.write("• Volatility crush can be severe")
    else:
        st.subheader("📋 General Strategy Notes")
        
        distance_to_max_pain = underlying_price - max_pain
        if distance_to_max_pain > 50:
            st.write("🔼 **Price above Max Pain** - Potential downward pressure")
        elif distance_to_max_pain < -50:
            st.write("🔽 **Price below Max Pain** - Potential upward pressure")
        else:
            st.write("⚖️ **Price near Max Pain** - Relatively balanced")


# 🔹 ENHANCED DASHBOARD RENDERING
def render_enhanced_dashboard(df, underlying_price, enhanced_signals):
    """Render enhanced intraday dashboard"""
    
    st.markdown("---")
    st.subheader("🚀 INTRADAY TRADING INTELLIGENCE")
    
    # Top row - Key intraday metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        timing = enhanced_signals['timing']
        st.metric(
            "Market Phase",
            timing.get('phase', 'Unknown'),
            help=timing.get('strategy_hint', 'No hints available')
        )
    
    with col2:
        pcr = enhanced_signals['pcr_analysis']
        st.metric(
            "Sentiment Score",
            f"{pcr.get('sentiment_score', 0):+.0f}",
            help=f"PCR: {pcr.get('pcr_weighted', 0):.2f}"
        )
    
    with col3:
        breakout = enhanced_signals['breakout_prediction']
        st.metric(
            "Breakout Probability",
            f"{breakout.get('breakout_probability', 0):.0f}%",
            help="ML-based breakout prediction"
        )
    
    with col4:
        micro = enhanced_signals['microstructure']
        st.metric(
            "Smart Money Bias",
            micro.get('smart_money_bias', 'None'),
            help=f"Value: ₹{micro.get('smart_money_value', 0):.1f}M"
        )
    
    # Intraday Strategy Recommendations
    st.subheader("📋 Live Trading Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Current Setup:**")
        
        # Dynamic recommendations based on signals
        breakout_prob = enhanced_signals['breakout_prediction'].get('breakout_probability', 0)
        sentiment_score = enhanced_signals['pcr_analysis'].get('sentiment_score', 0)
        
        if breakout_prob > 70:
            st.success("🚀 HIGH VOLATILITY SETUP - Consider Long Straddle/Strangle")
        elif breakout_prob < 30 and abs(sentiment_score) < 25:
            st.info("😴 LOW VOLATILITY SETUP - Consider Short Straddle (with caution)")
        else:
            st.warning("⚖️ MIXED SIGNALS - Wait for clearer setup")
        
        # Market phase specific advice
        phase = enhanced_signals['timing'].get('current_phase', '')
        if phase == 'opening':
            st.write("• Opening hour: Use wider stops")
            st.write("• Avoid premium selling strategies")
        elif phase == 'power_hour':
            st.write("• Final moves expected")
            st.write("• Consider closing positions")
        else:
            st.write("• Mid-session: Range trading opportunities")
    
    with col2:
        st.markdown("**⚠️ Risk Management:**")
        
        # Dynamic risk warnings
        micro_signals = enhanced_signals['microstructure']
        if micro_signals.get('liquid_strikes_count', 0) < 3:
            st.error("🚨 LOW LIQUIDITY - Avoid large positions")
        
        vwap_dev = micro_signals.get('vwap_deviation', 0)
        if vwap_dev > 1:
            st.warning(f"📊 Price {vwap_dev:.1f}% away from VWAP")
        
        st.write("• Use appropriate position sizing")
        st.write("• Monitor time decay in final hour")
        st.write("• Keep stops based on volatility")
    
    # Advanced Charts Section
    st.subheader("📊 Advanced Intraday Charts")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Smart Money Flow", "📈 Momentum Heatmap", "⚡ Volatility Surface"])
    
    with tab1:
        # Smart Money Flow Chart
        df['Smart_Money_Score'] = df['OI'] * df['LTP']
        flow_data = df.groupby(['StrikePrice', 'Type'])['Smart_Money_Score'].sum().reset_index()
        
        fig_flow = px.bar(
            flow_data,
            x='StrikePrice',
            y='Smart_Money_Score',
            color='Type',
            title="Smart Money Flow by Strike",
            color_discrete_map={'CE': '#ff0000', 'PE': '#00ff00'}
        )
        st.plotly_chart(fig_flow, use_container_width=True)
    
    with tab2:
        # Momentum Heatmap
        pivot_momentum = df.pivot(index='StrikePrice', columns='Type', values='IV').fillna(0)
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_momentum.values,
            x=['CE', 'PE'],
            y=pivot_momentum.index,
            colorscale='RdYlGn',
            hoverongaps=False
        ))
        fig_heatmap.update_layout(title="IV Momentum Heatmap")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab3:
        # 3D Volatility Surface
        fig_3d = go.Figure(data=[go.Surface(
            z=pivot_momentum.values,
            x=['CE', 'PE'],
            y=pivot_momentum.index
        )])
        fig_3d.update_layout(title="3D Volatility Surface", scene=dict(
            xaxis_title="Option Type",
            yaxis_title="Strike Price", 
            zaxis_title="Implied Volatility"
        ))
        st.plotly_chart(fig_3d, use_container_width=True)

def render_analysis_dashboard(df, underlying_price):
    """Render the comprehensive analysis dashboard"""
    
    # Calculate metrics
    metrics = create_comprehensive_analysis(df, underlying_price)
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 IV Analysis", "🔢 Greeks", "📋 Data Table"])
    
    with tab1:
        # Key Metrics Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total CE OI", 
                f"{metrics['total_ce_oi']:.1f}M",
                help="Total Call Options Open Interest"
            )
        
        with col2:
            st.metric(
                "Total PE OI", 
                f"{metrics['total_pe_oi']:.1f}M",
                help="Total Put Options Open Interest"
            )
        
        with col3:
            st.metric(
                "Put/Call Ratio", 
                f"{metrics['put_call_ratio']:.2f}",
                help="Ratio of Put OI to Call OI"
            )
        
        with col4:
            st.metric(
                "Max Pain", 
                f"{metrics['max_pain_strike']:.0f}",
                help="Strike with highest total OI"
            )
        
        # Open Interest Distribution Chart
        st.subheader("📊 Open Interest Distribution")
        
        # Prepare data for OI chart
        pivot_df = df.pivot(index='StrikePrice', columns='Type', values='OI').fillna(0)
        
        fig_oi = go.Figure()
        fig_oi.add_trace(go.Bar(
            name='Call OI',
            x=pivot_df.index,
            y=pivot_df['CE'],
            marker_color='#EF4444'
        ))
        fig_oi.add_trace(go.Bar(
            name='Put OI', 
            x=pivot_df.index,
            y=pivot_df['PE'],
            marker_color='#3B82F6'
        ))
        
        fig_oi.update_layout(
            title="Open Interest by Strike Price",
            xaxis_title="Strike Price",
            yaxis_title="Open Interest",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig_oi, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Implied Volatility Analysis")
        
        # IV Skew Chart
        fig_iv = go.Figure()
        
        ce_df = df[df['Type'] == 'CE']
        pe_df = df[df['Type'] == 'PE']
        
        fig_iv.add_trace(go.Scatter(
            x=ce_df['StrikePrice'],
            y=ce_df['IV'],
            mode='lines+markers',
            name='Call IV',
            line=dict(color='#3B82F6', width=2)
        ))
        
        fig_iv.add_trace(go.Scatter(
            x=pe_df['StrikePrice'],
            y=pe_df['IV'],
            mode='lines+markers',
            name='Put IV',
            line=dict(color='#EF4444', width=2)
        ))
        
        fig_iv.update_layout(
            title="Implied Volatility Skew",
            xaxis_title="Strike Price",
            yaxis_title="Implied Volatility (%)",
            height=400
        )
        st.plotly_chart(fig_iv, use_container_width=True)
        
        # IV vs OI Scatter
        st.subheader("🎯 IV vs Open Interest Scatter")
        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=ce_df['OI'],
            y=ce_df['IV'],
            mode='markers',
            name='Calls',
            marker=dict(color='#3B82F6', size=8),
            text=ce_df['StrikePrice'],
            hovertemplate='Strike: %{text}<br>OI: %{x}<br>IV: %{y}%'
        ))
        
        fig_scatter.add_trace(go.Scatter(
            x=pe_df['OI'],
            y=pe_df['IV'],
            mode='markers',
            name='Puts',
            marker=dict(color='#EF4444', size=8),
            text=pe_df['StrikePrice'],
            hovertemplate='Strike: %{text}<br>OI: %{x}<br>IV: %{y}%'
        ))
        
        fig_scatter.update_layout(
            title="IV vs Open Interest",
            xaxis_title="Open Interest",
            yaxis_title="Implied Volatility (%)",
            height=400
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        st.subheader("🔢 Greeks Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Delta Profile
            fig_delta = go.Figure()
            fig_delta.add_trace(go.Scatter(
                x=ce_df['StrikePrice'],
                y=ce_df['Delta'],
                mode='lines+markers',
                name='Call Delta',
                line=dict(color='#3B82F6', width=2)
            ))
            fig_delta.add_trace(go.Scatter(
                x=pe_df['StrikePrice'],
                y=pe_df['Delta'],
                mode='lines+markers',
                name='Put Delta',
                line=dict(color='#EF4444', width=2)
            ))
            fig_delta.update_layout(
                title="Delta Profile",
                xaxis_title="Strike Price",
                yaxis_title="Delta",
                height=300
            )
            st.plotly_chart(fig_delta, use_container_width=True)
        
        with col2:
            # Gamma Profile
            fig_gamma = go.Figure()
            fig_gamma.add_trace(go.Scatter(
                x=ce_df['StrikePrice'],
                y=ce_df['Gamma'],
                mode='lines+markers',
                name='Call Gamma',
                line=dict(color='#10B981', width=2)
            ))
            fig_gamma.add_trace(go.Scatter(
                x=pe_df['StrikePrice'],
                y=pe_df['Gamma'],
                mode='lines+markers',
                name='Put Gamma',
                line=dict(color='#F59E0B', width=2)
            ))
            fig_gamma.update_layout(
                title="Gamma Profile",
                xaxis_title="Strike Price", 
                yaxis_title="Gamma",
                height=300
            )
            st.plotly_chart(fig_gamma, use_container_width=True)
        
        # Theta Analysis
        st.subheader("⏰ Theta (Time Decay) Analysis")
        fig_theta = go.Figure()
        fig_theta.add_trace(go.Bar(
            name='Call Theta',
            x=ce_df['StrikePrice'],
            y=ce_df['Theta'],
            marker_color='#3B82F6'
        ))
        fig_theta.add_trace(go.Bar(
            name='Put Theta',
            x=pe_df['StrikePrice'], 
            y=pe_df['Theta'],
            marker_color='#EF4444'
        ))
        fig_theta.update_layout(
            title="Time Decay by Strike",
            xaxis_title="Strike Price",
            yaxis_title="Theta",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig_theta, use_container_width=True)
    
    with tab4:
        st.subheader("📋 Detailed Options Data")
        
        # Format the dataframe for better display
        display_df = df.copy()
        display_df['OI_M'] = (display_df['OI'] / 1000000).round(2)
        display_df['IV_%'] = display_df['IV'].round(2)
        display_df['LTP'] = display_df['LTP'].round(2)
        display_df['Delta'] = display_df['Delta'].round(4)
        display_df['Gamma'] = display_df['Gamma'].round(4)
        display_df['Theta'] = display_df['Theta'].round(2)
        
        # Highlight max pain row
        def highlight_max_pain(row):
            if row['StrikePrice'] == metrics['max_pain_strike']:
                return ['background-color: yellow'] * len(row)
            return [''] * len(row)
        
        styled_df = display_df.style.apply(highlight_max_pain, axis=1)
        st.dataframe(styled_df, use_container_width=True)
    
    # Key Insights Section
    st.subheader("🔍 Key Market Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Market Sentiment**")
        sentiment = "Bearish" if metrics['put_call_ratio'] > 1 else "Bullish"
        st.write(f"• Put/Call Ratio: {metrics['put_call_ratio']:.2f} - {sentiment} sentiment")
        st.write(f"• Total PE OI: {metrics['total_pe_oi']:.1f}M vs CE OI: {metrics['total_ce_oi']:.1f}M")
        st.write(f"• Max Pain at {metrics['max_pain_strike']:.0f} with {metrics['max_pain_oi']:.1f}M total OI")
    
    with col2:
        st.markdown("**Technical Notes**") 
        st.write(f"• Current Nifty Level: ~{underlying_price:.0f}")
        st.write(f"• ATM Strike appears to be around {metrics['estimated_spot']:.0f}")
        st.write("• High OI strikes suggest key support/resistance levels")

def send_unique_telegram_alerts(alerts):
    if "sent_alerts" not in st.session_state:
        st.session_state.sent_alerts = set()  # Use a set for efficiency

    new_alerts = [alert for alert in alerts if alert not in st.session_state.sent_alerts]

    if new_alerts:
        message = "\n".join(new_alerts)
        send_telegram_alert(message)  # Send only new alerts
        st.session_state.sent_alerts.update(new_alerts)  # Store sent alerts persistently
        save_session_state()  # Save after updating sent_alerts

# Function to save data to CSV
def save_to_csv(df):
    if not os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, index=False)
    else:
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)

# 🔹 REAL-TIME MARKET MICROSTRUCTURE ANALYSIS
def analyze_market_microstructure(df, underlying_price):
    """Advanced microstructure analysis for institutional flow detection"""
    
    microstructure_signals = {}
    alerts = []
    
    # 1. OI-WEIGHTED PRICE ANALYSIS
    df['Trade_Value'] = df['LTP'] * df['OI']
    total_value = df['Trade_Value'].sum()
    
    if total_value > 0:
        vwap = (df['Trade_Value'] * df['StrikePrice']).sum() / total_value
        vwap_deviation = abs(underlying_price - vwap) / underlying_price * 100
        
        if vwap_deviation > VWAP_DEVIATION_THRESHOLD:
            direction = "above" if underlying_price > vwap else "below"
            alerts.append(f"📊 VWAP DEVIATION: Price {direction} VWAP by {vwap_deviation:.2f}%\n"
                         f"Current: {underlying_price:.0f} | VWAP: {vwap:.0f}")
        
        microstructure_signals['vwap'] = vwap
        microstructure_signals['vwap_deviation'] = vwap_deviation
    
    # 2. SMART MONEY DETECTION (Large Block Trades)
    df['Smart_Money_Score'] = df['OI'] * df['LTP']
    smart_money_trades = df[df['Smart_Money_Score'] > SMART_MONEY_THRESHOLD]
    
    if len(smart_money_trades) > 0:
        dominant_type = smart_money_trades['Type'].mode()[0] if len(smart_money_trades['Type'].mode()) > 0 else 'Mixed'
        total_smart_value = smart_money_trades['Smart_Money_Score'].sum() / 1000000
        
        alerts.append(f"🐋 SMART MONEY DETECTED: {dominant_type} dominance\n"
                     f"Total Value: ₹{total_smart_value:.1f}M across {len(smart_money_trades)} strikes")
        
        microstructure_signals['smart_money_bias'] = dominant_type
        microstructure_signals['smart_money_value'] = total_smart_value
    
    # 3. LIQUIDITY HEAT MAP
    liquid_strikes = df[df['OI'] > LIQUIDITY_THRESHOLD]['StrikePrice'].unique()
    atm_distance = abs(liquid_strikes - underlying_price)
    liquid_atm_strikes = liquid_strikes[atm_distance <= 200]  # Within 200 points
    
    if len(liquid_atm_strikes) < 3:
        alerts.append("⚠️ LOW LIQUIDITY WARNING: Limited liquid strikes near ATM")
    
    microstructure_signals['liquid_strikes_count'] = len(liquid_atm_strikes)
    
    # 4. ORDER FLOW IMBALANCE DETECTION
    ce_flow = df[df['Type'] == 'CE']['Smart_Money_Score'].sum()
    pe_flow = df[df['Type'] == 'PE']['Smart_Money_Score'].sum()
    flow_ratio = pe_flow / ce_flow if ce_flow > 0 else float('inf')
    
    if flow_ratio > 2:
        alerts.append("🔴 BEARISH FLOW DOMINANCE: Heavy institutional Put buying")
    elif flow_ratio < 0.5:
        alerts.append("🟢 BULLISH FLOW DOMINANCE: Heavy institutional Call buying")
    
    microstructure_signals['flow_ratio'] = flow_ratio
    
    return alerts, microstructure_signals

# 🔹 ADVANCED VOLATILITY REGIME DETECTION
def detect_volatility_regime(df_history):
    """Multi-timeframe volatility regime analysis"""
    
    if len(df_history) < 20:
        return [], {}
    
    alerts = []
    vol_signals = {}
    
    # Calculate rolling volatility for different periods
    for period in VOLATILITY_REGIME_PERIODS:
        if len(df_history) >= period:
            recent_ivs = [data['avg_iv'] for data in df_history[-period:]]
            vol_std = np.std(recent_ivs)
            vol_mean = np.mean(recent_ivs)
            current_iv = recent_ivs[-1]
            
            # Z-score for current IV
            z_score = (current_iv - vol_mean) / vol_std if vol_std > 0 else 0
            
            if abs(z_score) > 2:  # 2 standard deviations
                regime = "HIGH" if z_score > 0 else "LOW"
                alerts.append(f"📊 {period}-PERIOD VOL REGIME: {regime} volatility detected\n"
                             f"Current IV: {current_iv:.1f}% | Z-Score: {z_score:.2f}")
                
                vol_signals[f'regime_{period}'] = regime
                vol_signals[f'z_score_{period}'] = z_score
    
    return alerts, vol_signals

# 🔹 MARKET TIMING INTELLIGENCE
def get_market_timing_signals():
    """Advanced market timing based on intraday patterns"""
    
    current_time = datetime.now(IST).time()
    current_hour = current_time.hour
    current_minute = current_time.minute
    
    timing_signals = {}
    alerts = []
    
    # Determine current market phase
    if 9 <= current_hour < 10 or (current_hour == 10 and current_minute <= 30):
        phase = "opening"
        timing_signals['phase'] = "Opening Volatility"
        timing_signals['strategy_hint'] = "High volatility expected, avoid premium selling"
        
    elif 11 <= current_hour < 14:
        phase = "mid_session"
        timing_signals['phase'] = "Mid-Session Calm"
        timing_signals['strategy_hint'] = "Lower volatility, consider range trading"
        
    elif current_hour >= 14:
        phase = "power_hour"
        timing_signals['phase'] = "Power Hour"
        timing_signals['strategy_hint'] = "Final moves, watch for directional breakouts"
        
        # Special power hour alerts
        if current_hour >= 15 and current_minute >= 15:
            alerts.append("⚡ FINAL 15 MINUTES: Avoid new positions, close existing trades")
            
    else:
        phase = "pre_market"
        timing_signals['phase'] = "Pre-Market"
    
    # Time-based strategy adjustments
    if phase == "opening":
        alerts.append("🌅 OPENING HOUR: High volatility expected - Use wider stops")
    elif phase == "power_hour":
        alerts.append("⚡ POWER HOUR: Final directional moves likely")
    
    timing_signals['current_phase'] = phase
    return alerts, timing_signals

# 🔹 ENHANCED PCR WITH SENTIMENT ANALYSIS
def advanced_pcr_analysis(df, underlying_price):
    """Advanced Put-Call Ratio with sentiment scoring"""
    
    ce_df = df[df['Type'] == 'CE']
    pe_df = df[df['Type'] == 'PE']
    
    # Multiple PCR calculations
    pcr_oi = pe_df['OI'].sum() / ce_df['OI'].sum() if ce_df['OI'].sum() > 0 else 0
    pcr_OI = pe_df['LTP'].sum() / ce_df['LTP'].sum() if ce_df['LTP'].sum() > 0 else 0
    
    # Weighted PCR (OI * LTP)
    ce_weighted = (ce_df['OI'] * ce_df['LTP']).sum()
    pe_weighted = (pe_df['OI'] * pe_df['LTP']).sum()
    pcr_weighted = pe_weighted / ce_weighted if ce_weighted > 0 else 0
    
    # Sentiment Score (-100 to +100)
    sentiment_score = 0
    
    if pcr_weighted > 1.5:
        sentiment_score = -75  # Very bearish
    elif pcr_weighted > 1.2:
        sentiment_score = -50  # Bearish
    elif pcr_weighted > 0.8:
        sentiment_score = 0    # Neutral
    elif pcr_weighted > 0.6:
        sentiment_score = 50   # Bullish
    else:
        sentiment_score = 75   # Very bullish
    
    alerts = []
    
    # PCR-based alerts
    if pcr_weighted > 1.5:
        alerts.append(f"🔴 EXTREME BEARISH SENTIMENT: Weighted PCR {pcr_weighted:.2f}\n"
                     f"Potential contrarian bullish setup")
    elif pcr_weighted < 0.5:
        alerts.append(f"🟢 EXTREME BULLISH SENTIMENT: Weighted PCR {pcr_weighted:.2f}\n"
                     f"Potential contrarian bearish setup")
    
    pcr_signals = {
        'pcr_oi': pcr_oi,
        'pcr_weighted': pcr_weighted,
        'sentiment_score': sentiment_score,
        'sentiment_text': 'Very Bearish' if sentiment_score < -50 else 
                         'Bearish' if sentiment_score < -25 else
                         'Neutral' if abs(sentiment_score) <= 25 else
                         'Bullish' if sentiment_score < 50 else 'Very Bullish'
    }
    
    return alerts, pcr_signals

# 🔹 BREAKOUT PREDICTION MODEL
def predict_breakout_probability(df, underlying_price, microstructure_signals):
    """ML-based breakout prediction model"""
    
    try:
        # Feature engineering
        features = []
        
        # ATM IV features
        atm_strikes = df[abs(df['StrikePrice'] - underlying_price) <= 50]
        if len(atm_strikes) > 0:
            atm_iv_mean = atm_strikes['IV'].mean()
            atm_iv_std = atm_strikes['IV'].std()
            features.extend([atm_iv_mean, atm_iv_std])
        else:
            features.extend([0, 0])
        
        # OI concentration
        total_oi = df['OI'].sum()
        atm_oi_concentration = atm_strikes['OI'].sum() / total_oi if total_oi > 0 else 0
        features.append(atm_oi_concentration)
        
        # Gamma exposure
        total_gamma = df['Gamma'].sum()
        features.append(total_gamma)
        
        # Microstructure features
        features.extend([
            microstructure_signals.get('vwap_deviation', 0),
            microstructure_signals.get('flow_ratio', 1),
            microstructure_signals.get('smart_money_value', 0)
        ])
        
        # Simple heuristic model (replace with trained ML model in production)
        breakout_score = 0
        
        # High IV + High OI concentration = Higher breakout probability
        if len(features) >= 3:
            if features[0] > 20 and features[2] > 0.3:  # High IV + High OI concentration
                breakout_score += 30
            
            if features[3] > 0.05:  # High gamma
                breakout_score += 25
            
            if features[4] > 1:  # VWAP deviation
                breakout_score += 20
            
            if features[5] > 2 or features[5] < 0.5:  # Extreme flow ratio
                breakout_score += 25
        
        breakout_probability = min(100, breakout_score)
        
        alerts = []
        if breakout_probability > 70:
            alerts.append(f"🚀 HIGH BREAKOUT PROBABILITY: {breakout_probability}%\n"
                         f"Consider straddle/strangle strategies")
        
        return alerts, {'breakout_probability': breakout_probability, 'features': features}
        
    except Exception as e:
        return [], {'breakout_probability': 0, 'error': str(e)}

# 🔹 ENHANCED MOMENTUM INDICATORS
def calculate_momentum_indicators(rolling_data, strike_price, opt_type):
    """Calculate advanced momentum indicators"""
    
    key = f"{strike_price}_{opt_type}"
    if key not in rolling_data or len(rolling_data[key]) < 5:
        return {}
    
    data_points = list(rolling_data[key])
    
    # Extract time series
    iv_series = [point['IV'] for point in data_points]
    oi_series = [point['OI'] for point in data_points]
    ltp_series = [point['LTP'] for point in data_points]
    
    indicators = {}
    
    try:
        # IV Momentum (rate of change)
        if len(iv_series) >= 2:
            iv_momentum = (iv_series[-1] - iv_series[-2]) / iv_series[-2] * 100 if iv_series[-2] != 0 else 0
            indicators['iv_momentum'] = iv_momentum
        
        # OI Acceleration
        if len(oi_series) >= 3:
            oi_change_1 = oi_series[-1] - oi_series[-2]
            oi_change_2 = oi_series[-2] - oi_series[-3]
            oi_acceleration = oi_change_1 - oi_change_2
            indicators['oi_acceleration'] = oi_acceleration
        
        # Price Momentum using RSI-like calculation
        if len(ltp_series) >= 5:
            price_changes = [ltp_series[i] - ltp_series[i-1] for i in range(1, len(ltp_series))]
            gains = [change for change in price_changes if change > 0]
            losses = [-change for change in price_changes if change < 0]
            
            avg_gain = sum(gains) / len(gains) if gains else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                indicators['price_rsi'] = rsi
        
    except Exception as e:
        indicators['error'] = str(e)
    
    return indicators

# 🔹 MODIFY YOUR MAIN FUNCTION TO USE ENHANCED FEATURES
# Replace your main_with_expiry_features() with this enhanced version:

def ultimate_main_function():
    """Ultimate enhanced main function with all features"""
    
    expiry_dates = get_expiry_dates()
    if not expiry_dates:
        st.error("No expiry dates found.")
        return

    nearest_expiry = expiry_dates[0]
    
    # Enhanced sidebar
    st.sidebar.markdown("# 🚀 ENHANCED DASHBOARD")
    st.sidebar.write(f"**Expiry:** {nearest_expiry}")
    
    if is_expiry_day(nearest_expiry):
        st.sidebar.markdown("🏁 **EXPIRY DAY ACTIVE**")
    
    # Advanced controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Dashboard Settings**")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 30, 300, 60)
    show_advanced = st.sidebar.checkbox("Show Advanced Features", value=True)
    
    manual_refresh = st.sidebar.button("🔄 Refresh Now")
    
    if manual_refresh or auto_refresh:
        with st.spinner("Fetching enhanced market data..."):
            option_chain = fetch_option_chain(nearest_expiry)
            
            if option_chain:
                # Use enhanced analysis
                df, net_oi_df, enhanced_signals = enhanced_analyze_data(option_chain)




                underlying_price = option_chain["data"]["last_price"]
                
                # Expiry analysis (your existing code)
                expiry_alerts, expiry_signals = expiry_day_analysis(df, underlying_price, nearest_expiry)
                
                if expiry_alerts:
                    st.session_state.alerts = expiry_alerts + st.session_state.alerts[:5]
                    save_session_state()  # Save after updating alerts
                    send_unique_telegram_alerts(expiry_alerts)

                # 🔹 ENHANCED HEADER
                st.title("🚀 ULTIMATE NIFTY OPTIONS DASHBOARD")
                
                
                # Real-time status bar
                status_col1, status_col2, status_col3, status_col4 = st.columns(4)
                
                with status_col1:
                    st.metric("Nifty", f"₹{underlying_price:.2f}")
                
                with status_col2:
                    timing_phase = enhanced_signals['timing'].get('phase', 'Unknown')
                    st.metric("Phase", timing_phase)
                
                with status_col3:
                    sentiment = enhanced_signals['pcr_analysis'].get('sentiment_text', 'Neutral')
                    st.metric("Sentiment", sentiment)
                
                with status_col4:
                    breakout_prob = enhanced_signals['breakout_prediction'].get('breakout_probability', 0)
                    st.metric("Breakout Risk", f"{breakout_prob}%")

                # 🔹 ENHANCED ALERTS
                if st.session_state.alerts:
                    with st.expander("🔔 LIVE TRADING ALERTS", expanded=True):
                        for alert in st.session_state.alerts[:8]:
                            if any(word in alert for word in ["BREAKOUT", "SMART MONEY", "MOMENTUM"]):
                                st.success(alert)
                            elif "WARNING" in alert or "RISK" in alert:
                                st.error(alert)
                            else:
                                st.warning(alert)

                # 🔹 ENHANCED DASHBOARDS
                if show_advanced:
                    render_enhanced_dashboard(df, underlying_price, enhanced_signals)
                
                # Your existing dashboards
                render_expiry_dashboard(df, underlying_price, nearest_expiry, expiry_signals)
                render_analysis_dashboard(df, underlying_price)

                # Enhanced Net OI with additional insights
                st.subheader("⚖️ Enhanced Net OI & Flow Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Your existing Net OI chart
                    net_oi_df['Color'] = net_oi_df['NetOI'].apply(lambda x: 'green' if x >= 0 else 'red')
                    fig_netoi = px.bar(
                        net_oi_df,
                        x=net_oi_df.index,
                        y="NetOI",
                        title="Net OI Imbalance (PE - CE)",
                        color='Color',
                        color_discrete_map={'green': '#90ee90', 'red': '#ffcccb'}
                    )
                    st.plotly_chart(fig_netoi, use_container_width=True)
                
                with col2:
                    # New: Smart Money Distribution
                    df['Smart_Money'] = df['OI'] * df['LTP'] / 1000000  # in millions
                    smart_money_dist = df.groupby('Type')['Smart_Money'].sum()

                    fig_smart = px.pie(
                        values=smart_money_dist.values,
                        names=smart_money_dist.index,
                        title="Smart Money Distribution (₹M)",
                        color=smart_money_dist.index,
                        color_discrete_map={'CE': '#ffcccb', 'PE': '#90ee90'}
                    )

                    st.plotly_chart(fig_smart, use_container_width=True)


            else:
                st.error("Failed to fetch option chain data.")

        # Auto refresh logic
        if auto_refresh and not manual_refresh:
            # Periodic save before refresh
            save_session_state()
            time.sleep(refresh_interval)
            st.rerun()

# 🔹 REPLACE YOUR MAIN EXECUTION
if __name__ == "__main__":
    ultimate_main_function()











































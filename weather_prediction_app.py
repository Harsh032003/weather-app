import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pygam import LinearGAM, s, f
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import math

# --- Helper functions for header, footer, and login --- #
def add_header():
    st.markdown("""
    <style>
    .header {
        padding: 10px;
        background-color: #8baaac;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    <div class="header">
        Weather Forecasting Portal
    </div>
    """, unsafe_allow_html=True)

def add_footer():
    st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #8baaac;
        color: #555;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        © 2023 Weather Forecasting Portal. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

def login():
    st.title("Login")
    st.write("Please enter your credentials to access the Weather Forecasting Portal.")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # Simple hardcoded credentials (customize as needed)
        if username == "admin" and password == "password":
            st.success("Logged in successfully!")
            st.session_state.logged_in = True
        else:
            st.error("Invalid username or password.")

# --- Check Login Status --- #
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()  # Stop execution until user logs in

# --- Main App --- #
add_header()

# st.title("Weather Temperature Prediction with pyGAM")
st.title("Weather Temperature Prediction")

st.markdown("""
This app trains a Generalized Additive Model (GAM) on weather data and uses the model to predict temperature based on your input.  
**Dataset Requirements:**  
- **Formatted Date** (datetime)  
- **Temperature (C)**  
- **Humidity** (recorded as a fraction between 0 and 1)  
- **Wind Speed (km/h)**  
- **Pressure (millibars)**  
- **Apparent Temperature (C)**  
- **Wind Bearing (degrees)**  
- **Visibility (km)**  
- **Loud Cover**  
- **Precip Type** (categorical)  
""")

# 1. Load Data and Train Model (Cached as a Resource)
@st.cache_resource
def load_and_train_model():
    # Load dataset (ensure your CSV file is in the same folder)
    df = pd.read_csv("weatherHistory.csv")
    
    # Convert date column to datetime with utc=True to avoid warnings
    df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], utc=True)
    
    # One-hot encode the categorical column "Precip Type" if it exists
    if "Precip Type" in df.columns:
        df = pd.get_dummies(df, columns=["Precip Type"], drop_first=True)
    
    # Drop columns that are purely textual and not used for prediction
    drop_columns = ["Summary", "Daily Summary"]
    df = df.drop(columns=drop_columns, errors="ignore")
    
    # Extract additional time-based features from Formatted Date
    df["Month"] = df["Formatted Date"].dt.month
    df["Hour"] = df["Formatted Date"].dt.hour
    
    # Define required columns (adjust these as needed)
    required_cols = ["Humidity", "Wind Speed (km/h)", "Pressure (millibars)", 
                     "Temperature (C)", "Apparent Temperature (C)", "Wind Bearing (degrees)",
                     "Visibility (km)", "Loud Cover", "Month", "Hour"]
    df = df.dropna(subset=required_cols)
    
    # Display descriptive statistics of the columns we will use
    st.write("### Descriptive Statistics")
    st.write(df[required_cols].describe())
    
    # Define features and target.
    feature_cols = ["Humidity", "Wind Speed (km/h)", "Pressure (millibars)",
                    "Apparent Temperature (C)", "Wind Bearing (degrees)",
                    "Visibility (km)", "Loud Cover", "Month", "Hour"]
    
    # Include any Precip Type dummy columns if they exist
    precip_dummy_cols = [col for col in df.columns if col.startswith("Precip Type_")]
    feature_cols.extend(precip_dummy_cols)
    
    target = "Temperature (C)"
    X = df[feature_cols].values
    y = df[target].values
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- Updated model training using TermList with unpacking ---
    from pygam.terms import TermList
    n_features = X_train_scaled.shape[1]
    terms = TermList(*[s(i) for i in range(n_features)])
    gam = LinearGAM(terms=terms).fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = gam.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"**Mean Squared Error:** {mse:.4f}")
    
    # Plot Actual vs. Predicted Temperature for the test set
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax, alpha=0.6)
    ax.set_xlabel("Actual Temperature (C)")
    ax.set_ylabel("Predicted Temperature (C)")
    ax.set_title("Actual vs Predicted Temperature")
    st.pyplot(fig)
    
    return gam, scaler, feature_cols

gam, scaler, feature_cols = load_and_train_model()
n_features = len(feature_cols)  # Define n_features for use in plotting

# 2. User Input for Prediction
st.header("Predict Temperature")
st.markdown("""
Enter weather details below. For Humidity, input in percentage (it will be converted to a fraction, e.g., 90 becomes 0.90).  
For Precip Type, select one of the available options.
""")

# User input for numeric features:
humidity_input = st.number_input("Humidity (%)", value=70.0, min_value=0.0, max_value=100.0, step=1.0)
wind_speed_input = st.number_input("Wind Speed (km/h)", value=10.0, min_value=0.0, max_value=100.0, step=0.5)
pressure_input = st.number_input("Pressure (millibars)", value=1010.0, min_value=900.0, max_value=1100.0, step=1.0)
apparent_temp_input = st.number_input("Apparent Temperature (C)", value=12.0)
wind_bearing_input = st.number_input("Wind Bearing (degrees)", value=180.0, min_value=0.0, max_value=360.0, step=1.0)
visibility_input = st.number_input("Visibility (km)", value=10.0)
loud_cover_input = st.number_input("Loud Cover", value=0.5, min_value=0.0, max_value=1.0, step=0.01)
month_input = st.number_input("Month (1-12)", value=4, min_value=1, max_value=12, step=1)
hour_input = st.number_input("Hour (0-23)", value=12, min_value=0, max_value=23, step=1)

# For Precip Type, assume possible options (e.g., "none", "rain", "snow")—adjust as needed.
precip_options = ["none", "rain", "snow"]
precip_input = st.selectbox("Precip Type", precip_options)

# One-hot encode the Precip Type input based on training dummy columns.
precip_dummy = {}
for col in feature_cols:
    if col.startswith("Precip Type_"):
        precip_dummy[col] = 1.0 if col == f"Precip Type_{precip_input}" else 0.0

# Prepare user input as a dictionary matching feature_cols.
user_data = {
    "Humidity": humidity_input / 100.0,  # convert percentage to fraction
    "Wind Speed (km/h)": wind_speed_input,
    "Pressure (millibars)": pressure_input,
    "Apparent Temperature (C)": apparent_temp_input,
    "Wind Bearing (degrees)": wind_bearing_input,
    "Visibility (km)": visibility_input,
    "Loud Cover": loud_cover_input,
    "Month": month_input,
    "Hour": hour_input,
}

# Include Precip Type dummies if available in feature_cols
for col in feature_cols:
    if col.startswith("Precip Type_"):
        user_data[col] = precip_dummy.get(col, 0.0)

# Create user input array ensuring the order of columns matches feature_cols
user_input = np.array([[user_data[col] for col in feature_cols]])
user_input_scaled = scaler.transform(user_input)

if st.button("Predict Temperature"):
    # Predict temperature using the trained GAM model
    predicted_temp = gam.predict(user_input_scaled)[0]
    st.success(f"🌡️ Predicted Temperature: {predicted_temp:.2f} °C")
    # st.write(f"Predicted Temperature : {predicted_temp:.2f} °C")
    
    # 3. Plot Partial Dependence for Each Feature with User Input Highlighted
    st.subheader("Feature Effects with Your Input")
    
    # Set up grid layout with 3 columns per row
    n_cols = 3
    n_rows = math.ceil(n_features / n_cols)
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # If there is only one row, ensure axes is a list
    if n_rows == 1:
        axes = [axes]
    
    # Flatten axes for easier iteration if it's a 2D array
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    
    for i in range(n_features):
        XX = gam.generate_X_grid(term=i)
        pdp = gam.partial_dependence(term=i, X=XX)
        axes[i].plot(XX[:, i], pdp, color='blue')
        axes[i].axvline(x=user_input_scaled[0, i], color='red', linestyle='--', label="Your Input")
        axes[i].set_xlabel(f"{feature_cols[i]} (scaled)")
        axes[i].set_ylabel("Effect on Temperature")
        axes[i].set_title(f"Effect of {feature_cols[i]}")
        axes[i].legend()
    
    # Hide any unused subplots if total plots < n_rows * n_cols
    for j in range(n_features, n_rows * n_cols):
        fig2.delaxes(axes[j])
    
    st.pyplot(fig2)

add_footer()

import streamlit as st
import numpy as np
import random
import time
import pandas as pd
import plotly.graph_objs as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from fpdf import FPDF
import base64
import os

# Set the page configuration
st.set_page_config(page_title="LSTM Anomaly Detector", layout="wide")

# Page Heading
st.title("Efficient Real-Time Anomaly Detection")

# Sidebar controls for simulation settings
st.sidebar.header("Settings")
seasonal_amplitude = st.sidebar.slider("Seasonal Amplitude", 5, 20, 10)
noise_level = st.sidebar.slider("Noise Level", 0, 5, 2)
anomaly_value = st.sidebar.slider("Anomaly Value", 10, 100, 50)
window_size = st.sidebar.slider("Window Size", 50, 500, 100)
initial_epochs = st.sidebar.slider("Initial Training Epochs", 5, 50, 10)

# Initialize session state variables
if 'data_points' not in st.session_state:
    st.session_state.data_points = []
    st.session_state.anomalies = []
    st.session_state.t = 0
    st.session_state.running = False
    st.session_state.model = None

# Function to simulate data stream with seasonal and noise patterns
def simulate_data(t, seasonal_amplitude, noise_level, anomaly_value):
    """
    Simulates a single data point in a time series.
    Adds seasonal pattern, noise, and potential anomalies.
    
    Parameters:
    t : int
        Current time step in the simulation
    seasonal_amplitude : float
        Amplitude of the seasonal sinusoidal pattern
    noise_level : float
        Maximum range of random noise to add to the data
    anomaly_value : float
        Value added when an anomaly is generated (randomly chosen)
    
    Returns:
    float
        Simulated data point
    """
    seasonal_pattern = seasonal_amplitude * np.sin(t / 100)
    noise = random.uniform(-noise_level, noise_level)
    anomaly = random.choice([0]*199 + [anomaly_value])  # Rare anomalies
    long_term_trend = 0.005 * t  # Adding a small upward trend
    data_point = seasonal_pattern + noise + anomaly + long_term_trend
    return data_point

# Function to create and compile LSTM Autoencoder model
def create_autoencoder(window_size):
    """
    Creates and compiles an LSTM Autoencoder for anomaly detection.
    
    Parameters:
    window_size : int
        Number of time steps in the sliding window
    
    Returns:
    model : keras.Model
        Compiled LSTM autoencoder model
    """
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(window_size, 1), return_sequences=False))
    model.add(RepeatVector(window_size))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to train the autoencoder
def train_autoencoder(data, window_size, epochs):
    """
    Trains the LSTM autoencoder on the provided data.
    
    Parameters:
    data : list
        List of time series data to train the model
    window_size : int
        Number of time steps in the sliding window
    epochs : int
        Number of epochs for training
    
    Returns:
    model : keras.Model
        Trained LSTM autoencoder model
    """
    data = np.array(data).reshape(-1, window_size, 1)
    model = create_autoencoder(window_size)
    model.fit(data, data, epochs=epochs, batch_size=32, verbose=0)
    return model

# Function for anomaly detection using EMA for dynamic thresholding
def detect_anomalies_lstm(model, data_window, threshold_ema):
    """
    Detects anomalies by comparing reconstruction error (MSE) to a dynamic threshold.
    The threshold is updated using an Exponential Moving Average (EMA).
    
    Parameters:
    model : keras.Model
        Trained LSTM autoencoder model
    data_window : list
        Sliding window of time series data
    threshold_ema : float
        Previous value of the dynamic EMA threshold
    
    Returns:
    is_anomaly : bool
        True if the current data point is an anomaly
    mse : float
        Mean Squared Error between the input and reconstructed data
    threshold_ema : float
        Updated EMA threshold
    """
    data_window = np.array(data_window).reshape(1, window_size, 1)
    reconstructed = model.predict(data_window)
    mse = np.mean(np.power(data_window - reconstructed, 2))
    threshold_ema = 0.9 * threshold_ema + 0.1 * mse  # Updating EMA threshold
    return mse > threshold_ema, mse, threshold_ema

# Function to generate PDF report
def export_to_pdf(data_points, anomalies):
    """
    Generates a PDF report summarizing the detected anomalies in the data stream.
    
    Parameters:
    data_points : list
        List of data points in the time series
    anomalies : list
        List of detected anomalies (NaN if not an anomaly)
    
    Returns:
    pdf_file_path : str
        Path to the generated PDF file
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Real-Time Anomaly Detection Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Total Data Points: {len(data_points)}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Total Anomalies Detected: {len([x for x in anomalies if not np.isnan(x)])}", ln=True, align='L')
    pdf.ln(10)
    pdf.cell(200, 10, txt="Data Points and Anomalies", ln=True, align='L')
    pdf.ln(5)
    
    # Loop through data points and anomalies, summarizing in the PDF
    for i, (data, anomaly) in enumerate(zip(data_points, anomalies)):
        pdf.cell(200, 10, txt=f"Point {i+1}: Value = {data}, Anomaly = {'Yes' if not np.isnan(anomaly) else 'No'}", ln=True)
    
    # Check if the file path exists or create a unique name
    pdf_file_path = "anomaly_detection_report.pdf"
    pdf.output(pdf_file_path)
    
    if os.path.exists(pdf_file_path):
        return pdf_file_path
    else:
        raise ValueError("Failed to generate the PDF report.")

# Function to generate download link for PDF
def generate_pdf_download_link(pdf_path):
    """
    Generates a downloadable link for the generated PDF file.
    
    Parameters:
    pdf_path : str
        Path to the generated PDF file
    
    Returns:
    href : str
        HTML link to download the PDF
    """
    try:
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
        href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="anomaly_detection_report.pdf">Download PDF Report</a>'
        return href
    except FileNotFoundError:
        st.error("PDF file not found. Please try exporting again.")
        return None

# Sidebar buttons
start_button = st.sidebar.button("Start Simulation", disabled=st.session_state.running)
stop_button = st.sidebar.button("Stop Simulation", disabled=not st.session_state.running)
export_button = st.sidebar.button("Export Details as PDF", disabled=st.session_state.running or not st.session_state.data_points)

# Real-time simulation and anomaly detection
if start_button:
    st.session_state.running = True
    st.session_state.data_points = []
    st.session_state.anomalies = []
    st.session_state.t = 0
    st.session_state.model = None

if stop_button:
    st.session_state.running = False

# Clear Data and retrain model with new initial data
if st.session_state.running:
    if st.session_state.model is None:
        initial_data = [simulate_data(t, seasonal_amplitude, noise_level, anomaly_value) for t in range(window_size)]
        st.session_state.model = train_autoencoder(initial_data, window_size, initial_epochs)
    
    # Real-time Plotting with Plotly
    data_stream = []
    anomaly_points = []
    threshold_ema = 0.1  # Initial threshold EMA
    while st.session_state.running:
        data_point = simulate_data(st.session_state.t, seasonal_amplitude, noise_level, anomaly_value)
        st.session_state.data_points.append(data_point)

        if len(st.session_state.data_points) >= window_size:
            data_window = st.session_state.data_points[-window_size:]
            is_anomaly, mse, threshold_ema = detect_anomalies_lstm(st.session_state.model, data_window, threshold_ema)

            if is_anomaly:
                st.session_state.anomalies.append(data_point)
            else:
                st.session_state.anomalies.append(np.nan)
            
            data_stream = st.session_state.data_points[-200:]
            anomaly_points = st.session_state.anomalies[-200:]

            # Real-time Plotting with Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(data_stream))), y=data_stream, mode='lines', name='Data Stream', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=list(range(len(anomaly_points))), y=anomaly_points, mode='markers', name='Anomalies', marker=dict(color='red')))
            st.plotly_chart(fig)

        st.session_state.t += 1
        time.sleep(0.1)

# Export PDF after simulation stops
if export_button:
    try:
        pdf_path = export_to_pdf(st.session_state.data_points, st.session_state.anomalies)
        download_link = generate_pdf_download_link(pdf_path)
        if download_link:
            st.sidebar.markdown(download_link, unsafe_allow_html=True)
    except ValueError as e:
        st.sidebar.error(f"Error generating PDF: {e}")

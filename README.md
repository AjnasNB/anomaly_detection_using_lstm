
# Anomaly Detection Application

## Application Overview
This project demonstrates real-time anomaly detection using an LSTM Autoencoder. The application is built using Streamlit and TensorFlow, with real-time visualizations powered by Plotly.

## Features
- **Real-time Anomaly Detection**: Continuously monitors a data stream to detect anomalies using LSTM Autoencoder.
- **Dynamic Thresholding**: Uses Exponential Moving Average (EMA) for adaptive anomaly thresholding.
- **Real-time Visualization**: Displays the data stream and detected anomalies in real-time.
- **PDF Reporting**: Generates a PDF report summarizing the anomalies detected.

## Video Demo
You can watch a demo of how the application works here: [Demo Video](./anomaly_detection_working.py%20-%20cuddle%20-%20Visual%20Studio%20Code%202024-09-14%2017-17-59.mp4)

## How to Run the Application

### Step 1: Install the Required Dependencies
Make sure you have Python 3.x installed. To install the required libraries, use the following command:

```bash
pip install -r requirements.txt
```

This will install TensorFlow, Streamlit, Plotly, and other necessary dependencies.

### Step 2: Run the Streamlit Application
To run the Streamlit application, execute the following command in your terminal or command prompt:

```bash
streamlit run anomaly_detection.py
```

Make sure to replace `anomaly_detection.py` with the actual filename if it's different.

### Step 3: Interact via the Browser
Once the application starts, Streamlit will open a new tab in your web browser where you can interact with the anomaly detection system. If it does not open automatically, you can manually open your browser and navigate to the following address:

```bash
http://localhost:8501
```

## Conclusion
This application provides an efficient way to monitor real-time data streams and detect anomalies, with user-friendly visualizations and reporting capabilities.

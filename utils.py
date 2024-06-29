import pandas as pd
import numpy as np
from scipy.stats import gamma, poisson
import matplotlib.pyplot as plt
import io

def interpret_recommendations(issue_text, prediction, recommended_suppliers, external_factors, probabilities, threshold):
    """
    Interpret the recommendations and provide transparency.

    Args:
        issue_text (str): Description of the issue.
        prediction (list): Prediction result from the model.
        recommended_suppliers (list): List of recommended suppliers.
        external_factors (str): External factors affecting the decision.
        probabilities (list): List of probabilities for each class.
        threshold (float): Probability threshold for detecting issues.

    Returns:
        dict: Interpretation of the recommendations.
    """
    issue_detected = "Yes" if probabilities[1] >= threshold else "No"
    interpretation = {
        "issue_text": issue_text,
        "prediction": issue_detected,
        "recommended_suppliers": recommended_suppliers,
        "external_factors": external_factors,
        "delay_probability": f"{probabilities[1]:.2f}",
        "threshold": threshold
    }
    return interpretation

def detect_issues_automatically(order_id, supplier_name, external_factors, sensor_data_df):
    """
    Automatically detect issues based on order ID, supplier name, and external factors.

    Args:
        order_id (str): Order ID.
        supplier_name (str): Current supplier name.
        external_factors (str): External factors affecting the decision.
        sensor_data_df (DataFrame): DataFrame containing sensor data.

    Returns:
        str: Detected issue description.
    """
    detected_issues = []

    if "Weather" in external_factors:
        weather_issues = sensor_data_df[sensor_data_df['sensor_id'] == 1]
        if not weather_issues.empty:
            detected_issues.append("Weather delays causing delivery issues")

    if "Geopolitical" in external_factors:
        geopolitical_issues = sensor_data_df[sensor_data_df['sensor_id'] == 2]
        if not geopolitical_issues.empty:
            detected_issues.append("Geopolitical tensions affecting supply chain")

    if "Market" in external_factors:
        market_issues = sensor_data_df[sensor_data_df['sensor_id'] == 3]
        if not market_issues.empty:
            detected_issues.append("Market trends causing supply chain fluctuations")

    issue_description = " ".join(detected_issues) if detected_issues else "No significant issues detected"
    return issue_description

def predict_delay_probability(delivery_times):
    """
    Predict delay probability using gamma-Poisson distribution.

    Args:
        delivery_times (pd.Series): Series of delivery times.

    Returns:
        tuple: Delay probability, shape, loc, and scale parameters.
    """
    shape, loc, scale = gamma.fit(delivery_times, floc=0)
    mean_delivery_time = np.mean(delivery_times)
    delay_probability = poisson.cdf(mean_delivery_time + 2, mean_delivery_time)
    return delay_probability, shape, loc, scale

def generate_probability_graph(delivery_times):
    """
    Generate probability density function graph.

    Args:
        delivery_times (pd.Series): Series of delivery times.

    Returns:
        BytesIO: Image buffer containing the graph.
    """
    delay_probability, shape, loc, scale = predict_delay_probability(delivery_times)
    img = io.BytesIO()
    x = np.linspace(0, max(delivery_times) + 10, 100)
    y = gamma.pdf(x, shape, loc=loc, scale=scale)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', lw=2)
    plt.title('Probability Density Function')
    plt.xlabel('Delivery Time')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.savefig(img, format='png')
    img.seek(0)
    return img

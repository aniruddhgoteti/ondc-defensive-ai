from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification
from scipy.stats import gamma, poisson

app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load supplier data
suppliers_df = pd.read_csv('data/suppliers.csv')
delivery_data_df = pd.read_csv('data/delivery_data.csv')

# Function to prepare input for BERT
def prepare_input(text):
    """
    Tokenizes the input text for the BERT model.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        dict: Tokenized input suitable for the BERT model.
    """
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=128)
    return inputs

# Function to predict issues using the fine-tuned BERT model
def detect_issues(text):
    """
    Uses the fine-tuned BERT model to predict issues from the input text.

    Args:
        text (str): The input text describing a supply chain event.

    Returns:
        list: Predicted class for the input text (0 for no issue, 1 for issue).
    """
    inputs = prepare_input(text)
    outputs = model(inputs)
    predictions = tf.nn.softmax(outputs.logits, axis=-1)
    predicted_class = np.argmax(predictions, axis=1).numpy().tolist()
    return predicted_class

# Function to recommend alternative suppliers
def recommend_suppliers(current_supplier):
    """
    Recommends alternative suppliers, excluding the current one.

    Args:
        current_supplier (str): The name of the current supplier.

    Returns:
        list: A list of alternative suppliers.
    """
    alternatives = suppliers_df[suppliers_df['supplier'] != current_supplier].sample(3)
    return alternatives.to_dict('records')

# Function to predict delay probability using gamma-Poisson distribution
def predict_delay_probability(delivery_times):
    """
    Predicts the delay probability using a gamma-Poisson distribution.

    Args:
        delivery_times (pd.Series): Series of delivery times for a supplier.

    Returns:
        float: Probability of a delay.
    """
    shape, loc, scale = gamma.fit(delivery_times, floc=0)
    mean_delivery_time = np.mean(delivery_times)
    delay_probability = poisson.cdf(mean_delivery_time + 2, mean_delivery_time)
    return delay_probability

@app.route('/')
def index():
    """
    Renders the index page for the application.

    Returns:
        Response: Rendered HTML of the index page.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the form submission.

    Returns:
        Response: Rendered HTML with prediction results.
    """
    text = request.form['text']
    current_supplier = request.form['supplier']
    external_factors = request.form['external_factors']
    prediction = detect_issues(text)
    issue_detected = "Yes" if prediction[0] == 1 else "No"

    if issue_detected == "Yes":
        recommended_suppliers = recommend_suppliers(current_supplier)
        delivery_times = delivery_data_df[delivery_data_df['supplier'] == current_supplier]['delivery_time']
        delay_probability = predict_delay_probability(delivery_times)
    else:
        recommended_suppliers = []
        delay_probability = 0

    return render_template('result.html', text=text, issue_detected=issue_detected, 
                           recommended_suppliers=recommended_suppliers, delay_probability=delay_probability,
                           external_factors=external_factors)

@app.route('/buyer', methods=['GET', 'POST'])
def buyer():
    """
    Handles buyer-specific interactions, such as checking order status.

    Returns:
        Response: Rendered HTML for buyer interactions.
    """
    if request.method == 'POST':
        order_id = request.form['order_id']
        # Mock implementation: In a real scenario, this would query a database or API
        order_status = "In Transit"  # Mock status
        return render_template('buyer_result.html', order_id=order_id, order_status=order_status)
    return render_template('buyer.html')

@app.route('/seller', methods=['GET', 'POST'])
def seller():
    """
    Handles seller-specific interactions, such as managing inventory and orders.

    Returns:
        Response: Rendered HTML for seller interactions.
    """
    if request.method == 'POST':
        action = request.form['action']
        if action == 'update_inventory':
            # Mock implementation: In a real scenario, this would update the inventory in a database
            inventory_status = "Inventory Updated Successfully"
            return render_template('seller_result.html', inventory_status=inventory_status)
    return render_template('seller.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

import tensorflow as tf
import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification

def load_data():
    suppliers_df = pd.read_csv('data/suppliers.csv')
    delivery_data_df = pd.read_csv('data/delivery_data.csv')
    sensor_data_df = pd.read_csv('data/sensor_data.csv')
    return suppliers_df, delivery_data_df, sensor_data_df

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

def prepare_input(text):
    return tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=128)

def detect_issues(text):
    try:
        inputs = prepare_input(text)
        outputs = model(inputs)
        predictions = tf.nn.softmax(outputs.logits, axis=-1)
        probabilities = predictions.numpy().tolist()[0]
        return probabilities
    except Exception as e:
        print(f"Error in detect_issues: {e}")
        raise

def recommend_suppliers(current_supplier, suppliers_df):
    try:
        alternatives = suppliers_df[suppliers_df['supplier'] != current_supplier].sort_values(by='reliability_score', ascending=False).head(3)
        print(f"Recommended Suppliers: {alternatives}")
        return alternatives.to_dict('records')
    except Exception as e:
        print(f"Error in recommend_suppliers: {e}")
        raise

import tensorflow as tf
import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification
import io
import matplotlib.pyplot as plt
from scipy.stats import gamma, poisson

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
        return np.argmax(predictions, axis=1)
    except Exception as e:
        print(f"Error in detect_issues: {e}")
        raise

def recommend_suppliers(current_supplier, suppliers_df):
    try:
        alternatives = suppliers_df[suppliers_df['supplier'] != current_supplier].sample(3)
        return alternatives.to_dict('records')
    except Exception as e:
        print(f"Error in recommend_suppliers: {e}")
        raise

def predict_delay_probability(delivery_times):
    try:
        shape, loc, scale = gamma.fit(delivery_times, floc=0)
        mean_delivery_time = np.mean(delivery_times)
        delay_probability = poisson.cdf(mean_delivery_time + 2, mean_delivery_time)
        return delay_probability, shape, loc, scale
    except Exception as e:
        print(f"Error in predict_delay_probability: {e}")
        raise

def generate_probability_graph(delivery_times):
    try:
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
    except Exception as e:
        print(f"Error in generate_probability_graph: {e}")
        raise

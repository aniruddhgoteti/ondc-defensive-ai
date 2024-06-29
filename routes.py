from flask import render_template, request, send_file
from models import detect_issues, recommend_suppliers, generate_probability_graph, load_data
from utils import interpret_recommendations

def setup_routes(app):
    suppliers_df, delivery_data_df, sensor_data_df = load_data()

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/buyer', methods=['GET', 'POST'])
    def buyer():
        if request.method == 'POST':
            order_id = request.form['order_id']
            issue_text = request.form['issue_text']
            supplier_name = request.form['supplier_name']
            external_factors = request.form['external_factors']
            
            prediction = detect_issues(issue_text)
            issue_detected = "Yes" if prediction[0] == 1 else "No"
            
            if issue_detected == "Yes":
                recommended_suppliers = recommend_suppliers(supplier_name, suppliers_df)
                delivery_times = delivery_data_df[delivery_data_df['supplier'] == supplier_name]['delivery_time']
                img = generate_probability_graph(delivery_times)
                interpretation = interpret_recommendations(issue_text, prediction, recommended_suppliers, external_factors)
                return render_template('buyer_result.html', order_id=order_id, issue_detected=issue_detected, 
                                       recommended_suppliers=recommended_suppliers, delay_probability=interpretation['delay_probability'],
                                       external_factors=external_factors, graph_url="/buyer_plot.png", interpretation=interpretation)
            else:
                return render_template('buyer_result.html', order_id=order_id, issue_detected=issue_detected, graph_url=None, interpretation=None)

        return render_template('buyer.html')

    @app.route('/buyer_plot.png')
    def buyer_plot_png():
        img = generate_probability_graph(delivery_data_df['delivery_time'])
        return send_file(img, mimetype='image/png')

    @app.route('/seller', methods=['GET', 'POST'])
    def seller():
        if request.method == 'POST':
            product_id = request.form['product_id']
            quantity = int(request.form['quantity'])
            action = request.form['action']
            
            if action == 'add':
                inventory_status = f"Added {quantity} units of Product ID {product_id}."
            else:
                inventory_status = f"Removed {quantity} units of Product ID {product_id}."

            return render_template('seller_result.html', inventory_status=inventory_status)
        
        return render_template('seller.html')

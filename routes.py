from flask import render_template, request, send_file
from models import detect_issues, recommend_suppliers, load_data
from utils import interpret_recommendations, detect_issues_automatically, generate_probability_graph, detect_external_factors

def setup_routes(app):
    suppliers_df, delivery_data_df, sensor_data_df = load_data()

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/buyer', methods=['GET', 'POST'])
    def buyer():
        if request.method == 'POST':
            order_id = request.form['order_id']
            supplier_name = request.form['supplier_name']
            threshold = float(request.form['threshold'])
            
            # Automatically detect external factors
            external_factors = detect_external_factors(sensor_data_df)
            print(f"Detected External Factors: {external_factors}")
            
            issue_text = detect_issues_automatically(order_id, supplier_name, external_factors, sensor_data_df)
            print(f"Issue Text: {issue_text}")
            
            probabilities = detect_issues(issue_text)
            print(f"Probabilities: {probabilities}")
            
            recommended_suppliers = recommend_suppliers(supplier_name, suppliers_df)
            print(f"Recommended Suppliers: {recommended_suppliers}")
            
            delivery_times = delivery_data_df[delivery_data_df['supplier'] == supplier_name]['delivery_time']
            img = generate_probability_graph(delivery_times)
            interpretation = interpret_recommendations(issue_text, None, recommended_suppliers, external_factors, probabilities, threshold)
            print(f"Interpretation: {interpretation}")
            
            return render_template('buyer_result.html', order_id=order_id, interpretation=interpretation, 
                                   graph_url="/buyer_plot.png")
            
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

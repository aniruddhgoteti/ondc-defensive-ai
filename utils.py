def interpret_recommendations(issue_text, prediction, recommended_suppliers, external_factors):
    """
    Interpret the recommendations and provide transparency.
    
    Args:
        issue_text (str): Description of the issue.
        prediction (list): Prediction result from the model.
        recommended_suppliers (list): List of recommended suppliers.
        external_factors (str): External factors affecting the decision.
    
    Returns:
        dict: Interpretation of the recommendations.
    """
    interpretation = {
        "issue_text": issue_text,
        "prediction": "Issue Detected" if prediction[0] == 1 else "No Issue Detected",
        "recommended_suppliers": recommended_suppliers,
        "external_factors": external_factors,
        "delay_probability": "High" if prediction[0] == 1 else "Low"
    }
    return interpretation

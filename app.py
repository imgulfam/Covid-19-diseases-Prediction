import os
from flask import Flask, request, jsonify, render_template
from joblib import load
import matplotlib.pyplot as plt
import plotly.graph_objects as go

app = Flask(__name__)

# Directory for storing models and generated graphs
MODEL_DIR = "models/"
GRAPH_DIR = "static/graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

# Mapping diseases to their model files
disease_models = {
    "covid-19": "COVID-19_xgboost_model.joblib",
    "mumps": "Mumps_xgboost_model.joblib",
    "measles": "Measles_xgboost_model.joblib",
    "polio": "Polio_xgboost_model.joblib",
    "rubella": "Rubella_xgboost_model.joblib",
    "pertussis": "Pertussis_xgboost_model.joblib",
    "hepatitis": "Hepatitis_xgboost_model.joblib",
    "smallpox": "Smallpox_xgboost_model.joblib"
}

# Serve index.html
@app.route('/')
def index():
    return render_template('index.html')

# Serve the specific HTML file for each disease
@app.route('/disease/<disease>')
def disease_page(disease):
    if disease in disease_models:
        return render_template(f'{disease}.html')
    else:
        return "Disease page not found", 404

# Helper function to generate map graph
def generate_map_graph(disease, country, year):
    # Create a map graph with dummy data for illustration
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, f"Map for {disease} in {country}, {year}", 
            horizontalalignment='center', verticalalignment='center', fontsize=12)
    
    map_file_path = os.path.join(GRAPH_DIR, f"map_{disease}_{country}_{year}.png")
    fig.savefig(map_file_path)
    plt.close(fig)
    return f"static/graphs/{os.path.basename(map_file_path)}"

# Helper function to generate 3D graph
def generate_3d_graph(disease, year, cases):
    fig = go.Figure(data=[go.Scatter3d(
        x=[year],
        y=[cases],
        z=[0],
        mode='markers',
        marker=dict(size=10, color='blue'),
        text=[f"{disease} Cases in {year}: {cases}"]
    )])
    fig.update_layout(title=f"3D Plot: {disease} - Year {year}, Cases {cases}",
                      scene=dict(
                          xaxis_title='Year',
                          yaxis_title='Predicted Cases',
                          zaxis_title='Dummy Z-axis'
                      ))
    
    plot_file_path = os.path.join(GRAPH_DIR, f"3d_plot_{disease}_{year}.html")
    fig.write_html(plot_file_path)
    return f"static/graphs/{os.path.basename(plot_file_path)}"

# Prediction route for each disease
@app.route('/predict/<disease>', methods=['POST'])
def predict_disease(disease):
    if disease not in disease_models:
        return jsonify({"error": "Invalid disease"}), 400

    # Load the appropriate model
    model_path = os.path.join(MODEL_DIR, disease_models[disease])
    model = load(model_path)

    # Extract user inputs
    data = request.form
    year = int(data.get('year', 0))  # Provide a default value
    country = data.get('country', '')

    # Validate input values
    if year <= 0 or not country:
        return jsonify({"error": "Invalid input values"}), 400

    # Prepare the input for prediction (modify according to your model's requirements)
    try:
        predicted_cases = model.predict([[year]])[0]  # Adjust the input format as needed
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Generate graphs
    map_graph_path = generate_map_graph(disease, country, year)
    graph_3d_path = generate_3d_graph(disease, year, predicted_cases)

    # Return the generated graphs' paths
    return jsonify({
        "mapGraphUrl": "/" + map_graph_path,
        "threeDGraphUrl": "/" + graph_3d_path
    })

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('data.csv')

# Find-S Algorithm
def find_s_algorithm(data):
    specific_hypothesis = None
    for _, row in data.iterrows():
        if row['Purchase'] == 'Yes':
            if specific_hypothesis is None:
                specific_hypothesis = row[:-1].values
            else:
                for i in range(len(specific_hypothesis)):
                    if specific_hypothesis[i] != row[i]:
                        specific_hypothesis[i] = '?'
    return specific_hypothesis

# Function to recommend product names based on hypothesis and fallback logic
def recommend_products(user_data, hypothesis):
    attributes = df.columns[:-1]
    recommendations = df.copy()

    for i, attribute in enumerate(hypothesis):
        if attribute != '?' and attribute != 'Φ':
            recommendations = recommendations[recommendations[attributes[i]] == attribute]

    # Exclude products already bought by the user
    bought_products = user_data['ProductID'].values
    recommendations = recommendations[~recommendations['ProductID'].isin(bought_products)]

    # If no recommendations are found, fall back to similar category suggestions
    if recommendations.empty:
        user_categories = user_data['Category'].unique()
        recommendations = df[(df['Category'].isin(user_categories)) & (~df['ProductID'].isin(bought_products))]

    # Return only the product names
    return recommendations['ProductName'].drop_duplicates().tolist()

# Home route to render the form
@app.route('/')
def index():
    return render_template('index.html')

# Route to get recommendations based on Find-S
@app.route('/recommend', methods=['POST'])
def recommend_product():
    user_id = request.form.get('UserID', type=int)
    user_data = df[df['UserID'] == user_id]

    if user_data.empty:
        return render_template('index.html', error="No purchase history found for this user")

    # Apply Find-S to get a specific hypothesis
    specific_hypothesis_find_s = find_s_algorithm(user_data)

    # Generate recommendations based on the specific hypothesis or fallback
    recommendations = recommend_products(user_data, specific_hypothesis_find_s)

    return render_template('index.html', 
                           user_id=user_id, 
                           recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)

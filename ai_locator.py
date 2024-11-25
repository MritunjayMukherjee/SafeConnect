from flask import Flask, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)

# Ensure the static directory exists for storing images
os.makedirs('static', exist_ok=True)

@app.route('/')
def index():
    # Load and preprocess the data
    path = '/Users/mritunjaymukherjee/Downloads/population.csv'
    df = pd.read_csv(path)
    df["refugees"] = df["refugees"] + df["asylum_seekers"]

    _ = [df["ooc"][i] if df["coo_name"][i] != df["coa_name"][i]
         else 0 for i in range(len(df))]
    df["refugees"] = df["refugees"] + _
    _ = [df["ooc"][i] if df["coo_name"][i] == df["coa_name"][i]
         else 0 for i in range(len(df))]
    df["idps"] = df["idps"] + _

    df = df.drop(["coo", "coa", "asylum_seekers", "ooc", "oip", "hst"], axis=1)
    df = df.replace({
        "Central African Rep.": "CAR",
        "Dem. Rep. of the Congo": "DR Congo",
        "Iran (Islamic Rep. of)": "Iran",
        "Russian Federation": "Russia",
        "Serbia and Kosovo: S/RES/1244 (1999)": "Serbia & Kosovo",
        "Syrian Arab Rep.": "Syria",
        "Venezuela (Bolivarian Republic of)": "Venezuela"
    })

    data = df[['year', 'coa_name', 'refugees']]
    data = data.groupby(['year', 'coa_name'], as_index=False).sum()

    country_name = 'Sudan'  # Default to Sudan
    country_data = data[data['coa_name'] == country_name]

    if country_data.empty:
        return render_template('index.html', image_url=None, message="No data found for Sudan")

    X = country_data[['year']].values
    y = country_data['refugees'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    future_years = np.array(range(2023, 2033)).reshape(-1, 1)
    future_predictions = model.predict(future_years)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual Data')
    plt.plot(future_years, future_predictions, color='red', label='Predictions (2023-2032)')
    plt.xlabel('Year')
    plt.ylabel('Number of Refugees')
    plt.title(f'Refugee Prediction for {country_name} (2010-2032)')
    plt.legend()
    plt.grid()

    # Save plot to static directory
    image_path = os.path.join('static', 'refugee_prediction.png')
    plt.savefig(image_path)
    plt.close()

    # Render the image in the template
    return render_template('index.html', image_url=image_path, message=None)

if __name__ == '__main__':
    app.run(debug=True)

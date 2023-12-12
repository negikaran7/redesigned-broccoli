from dash import Dash, html, Output, Input, State, callback
import dash_mantine_components as dmc
import pandas as pd
import joblib

app = Dash(__name__)

input_data = dmc.Stack(
    children=[
        dmc.TextInput(
            id="input-1",
            label="Height",
            type="number",
            style={"width": 200},
            placeholder="Enter height",
            value=None,
        ),
        dmc.TextInput(
            id="input-2",
            label="Weight",
            type="number",
            style={"width": 200},
            placeholder="Enter weight",
            value=None,
        ),
        html.Div(id="prediction-output")  # Display prediction here
    ],
)

header_data = dmc.Header(
    height=60,
    children=[dmc.Text("Dog vs Cat Classification")],
    style={
        "backgroundColor": "#8a4244",
        "fontFamily": "Arial, sans-serif",
        "textAlign": "center",
        "fontSize": "36px",
        "fontWeight": "bold",
        "textTransform": "uppercase",
        "color": "#333",
    },
)

app.layout = html.Div(
    [
        dmc.MantineProvider(
            theme={"colorScheme": "dark"},
            children=[
                header_data,
                dmc.Paper(
                    [input_data],
                    p="lg",
                ),
            ],
        )
    ],
    style={
        "width": "50%",
        "textAlign": "center",
        "margin": "10px auto",
    },
)

@app.callback(
    Output("prediction-output", "children"),
    [Input("input-1", "value"), Input("input-2", "value")],
)
def get_predictions(data1, data2):
    if data1 is not None and data2 is not None:
        # Unpickle classifier
        clf = joblib.load("clf.pkl")
        # Get values from input fields
        height = data1
        weight = data2
        # Put inputs into a dataframe
        X = pd.DataFrame([[height, weight]], columns=["Height", "Weight"])
        # Get prediction
        prediction = clf.predict(X)[0]
        return html.Div(f"Prediction: {prediction}")

if __name__ == "__main__":
    app.run_server(debug=False, host="127.0.0.1", port=8181)

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import datetime
from plotly.subplots import make_subplots
import io
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import plotly.io as pio
import pickle
import sqlite3
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the dashboard
app.layout = html.Div(
    [
        html.H1("Anomaly Detection Dashboard"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Upload(
                            id="upload-data-2020",
                            children=html.Div(
                                ["Drag and Drop or ", html.A("Select a File")]
                            ),
                            style={
                                "width": "100%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "5px",
                                "textAlign": "center",
                                "margin": "10px",
                            },
                        ),
                        html.Div(id="output-data-upload-2020"),
                    ],
                    width=6,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            "Email:",
                            style={"display": "inline-block", "marginRight": "10px"},
                        ),
                        dcc.Input(
                            id="email-input",
                            type="email",
                            placeholder="Enter your email",
                            style={"display": "inline-block", "width": "auto"},
                        ),
                    ],
                    width=6,
                ),
            ]
        ),
        html.Button(
            "Run Models", id="run-models", n_clicks=0, style={"margin": "10px"}
        ),
        html.Button(
            "Save Dashboard", id="save-dashboard", n_clicks=0, style={"margin": "10px"}
        ),
        dcc.Download(id="download-dashboard-html"),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="hist-smoke"), width=4),
                dbc.Col(dcc.Graph(id="hist-temp"), width=4),
                dbc.Col(dcc.Graph(id="scatter-plot"), width=4),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="pca-plot"), width=6),
                dbc.Col(dcc.Graph(id="label-proportions"), width=6),
            ]
        ),
        html.Div(id="anomaly-data-points"),
    ]
)


# Function to parse uploaded files
def parse_contents(contents, filename):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            return pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif "xls" in filename:
            return pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return None


# Callback to display file names after upload
@app.callback(
    Output("output-data-upload-2020", "children"), Input("upload-data-2020", "filename")
)
def update_output_2020(filename):
    if filename is not None:
        return html.Div(["Uploaded File: ", filename])


# Function to send email
def send_email(receiver_email, data_point_index):
    sender_email = "imar.asuman@gmail.com"
    password = "Student@12345"

    subject = "Anomaly Detected"
    body = (
        f"An anomaly was detected in data point {data_point_index}. Please investigate."
    )

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.office365.com", 587) as server:
            server.starttls()
            server.login(sender_email, password)
            text = msg.as_string()
            server.sendmail(sender_email, receiver_email, text)
            print(f"Email sent for anomaly in data point {data_point_index}")
    except smtplib.SMTPAuthenticationError as e:
        print(f"Failed to send email: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Callback to handle file uploads, run models, and predict anomalies
@app.callback(
    Output("hist-smoke", "figure"),
    Output("hist-temp", "figure"),
    Output("scatter-plot", "figure"),
    Output("pca-plot", "figure"),
    Output("label-proportions", "figure"),
    Output("anomaly-data-points", "children"),
    Input("run-models", "n_clicks"),
    State("upload-data-2020", "contents"),
    State("upload-data-2020", "filename"),
    State("email-input", "value"),
)
def update_output(n_clicks, contents_2020, filename_2020, email):
    if n_clicks == 0:
        return {}, {}, {}, {}, {}, ""

    df_2020 = parse_contents(contents_2020, filename_2020)

    if df_2020 is None:
        return {}, {}, {}, {}, {}, ""

    new_data = df_2020
    new_data["Timestamp"] = pd.to_datetime(new_data["Timestamp"])

    new_data["Smoke_Abnormal"] = (
        (new_data["Smoke_Level"] > 1) & (new_data["Smoke_Level"] < 10)
    ).astype(int)
    new_data["Smoke_Fire_Alarm"] = (new_data["Smoke_Level"] >= 20).astype(int)
    new_data["Temp_Abnormal"] = (
        (new_data["Temperature_Level"] < 20) | (new_data["Temperature_Level"] > 25)
    ).astype(int)
    new_data["Temp_Fire_Alarm"] = (
        (new_data["Temperature_Level"] >= 57) & (new_data["Temperature_Level"] <= 77)
    ).astype(int)

    fig_smoke = px.histogram(
        new_data, x="Smoke_Level", nbins=30, title="Distribution of Smoke Levels"
    )
    fig_temp = px.histogram(
        new_data,
        x="Temperature_Level",
        nbins=30,
        title="Distribution of Temperature Levels",
        color_discrete_sequence=["orange"],
    )
    fig_scatter = px.scatter(
        new_data,
        x="Smoke_Level",
        y="Temperature_Level",
        color="Door_Status",
        title="Smoke Level vs Temperature Level",
    )

    features = [
        "Smoke_Level",
        "Temperature_Level",
        "Smoke_Abnormal",
        "Smoke_Fire_Alarm",
        "Temp_Abnormal",
        "Temp_Fire_Alarm",
    ]
    X = new_data[features]

    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    iso_forest.fit(X)
    new_data["Anomaly_Iso"] = iso_forest.predict(X)
    new_data["Anomaly_Iso"] = new_data["Anomaly_Iso"].map({1: 0, -1: 1})

    kmeans = KMeans(n_clusters=5, random_state=42)
    new_data["Cluster"] = kmeans.fit_predict(X)
    anomalous_clusters = [1, 2, 4]
    new_data["Anomaly_KMeans"] = new_data["Cluster"].apply(
        lambda x: 1 if x in anomalous_clusters else 0
    )

    dbscan = DBSCAN(eps=0.5, min_samples=10)
    new_data["Anomaly_DBSCAN"] = dbscan.fit_predict(X)
    new_data["Anomaly_DBSCAN"] = (new_data["Anomaly_DBSCAN"] == -1).astype(int)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig_pca = make_subplots(
        rows=1, cols=3, subplot_titles=["K-Means", "Isolation Forest", "DBSCAN"]
    )
    fig_pca.add_trace(
        go.Scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            mode="markers",
            marker=dict(color=new_data["Anomaly_KMeans"], colorscale="Viridis"),
            name="K-Means",
        ),
        row=1,
        col=1,
    )
    fig_pca.add_trace(
        go.Scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            mode="markers",
            marker=dict(color=new_data["Anomaly_Iso"], colorscale="Viridis"),
            name="Isolation Forest",
        ),
        row=1,
        col=2,
    )
    fig_pca.add_trace(
        go.Scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            mode="markers",
            marker=dict(color=new_data["Anomaly_DBSCAN"], colorscale="Viridis"),
            name="DBSCAN",
        ),
        row=1,
        col=3,
    )
    fig_pca.update_layout(title="PCA of Anomaly Detection Results")

    proportions = {
        "K-Means": new_data["Anomaly_KMeans"].value_counts(normalize=True),
        "Isolation Forest": new_data["Anomaly_Iso"].value_counts(normalize=True),
        "DBSCAN": new_data["Anomaly_DBSCAN"].value_counts(normalize=True),
    }
    proportions_df = (
        pd.DataFrame(proportions)
        .reset_index()
        .melt(
            id_vars="index",
            value_vars=proportions.keys(),
            var_name="Model",
            value_name="Proportion",
        )
    )
    fig_proportions = px.bar(
        proportions_df,
        x="index",
        y="Proportion",
        color="Model",
        barmode="group",
        title="Proportion of Anomalies by Model",
    )

    # Train Random Forest Classifier
    y = new_data["Anomaly_KMeans"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Save the model
    with open("anomaly_detector.pkl", "wb") as f:
        pickle.dump(clf, f)

    # Create 5 new data points for prediction
    new_data_points = pd.read_csv("iot.csv")

    # Add engineered features to new data points
    new_data_points["Smoke_Abnormal"] = (
        (new_data_points["Smoke_Level"] > 1) & (new_data_points["Smoke_Level"] < 10)
    ).astype(int)
    new_data_points["Smoke_Fire_Alarm"] = (new_data_points["Smoke_Level"] >= 20).astype(
        int
    )
    new_data_points["Temp_Abnormal"] = (
        (new_data_points["Temperature_Level"] < 20)
        | (new_data_points["Temperature_Level"] > 25)
    ).astype(int)
    new_data_points["Temp_Fire_Alarm"] = (
        (new_data_points["Temperature_Level"] >= 57)
        & (new_data_points["Temperature_Level"] <= 77)
    ).astype(int)

    # Load the model
    with open("anomaly_detector.pkl", "rb") as f:
        clf_loaded = pickle.load(f)

    # Predict labels for new data points
    predictions = clf_loaded.predict(new_data_points)

    # Create a database connection
    conn = sqlite3.connect("anomaly_data.db")
    cursor = conn.cursor()

    # Create a table if it doesn't exist
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS anomaly_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Smoke_Level REAL,
            Temperature_Level REAL,
            Smoke_Abnormal INTEGER,
            Smoke_Fire_Alarm INTEGER,
            Temp_Abnormal INTEGER,
            Temp_Fire_Alarm INTEGER,
            Anomaly_Label INTEGER
        )
    """
    )

    anomaly_data_points = []
    # Insert new data points with predictions into the database
    for i, row in new_data_points.iterrows():
        label = predictions[i]
        cursor.execute(
            """
            INSERT INTO anomaly_records (Smoke_Level, Temperature_Level, Smoke_Abnormal, Smoke_Fire_Alarm, Temp_Abnormal, Temp_Fire_Alarm, Anomaly_Label)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                row["Smoke_Level"],
                row["Temperature_Level"],
                row["Smoke_Abnormal"],
                row["Smoke_Fire_Alarm"],
                row["Temp_Abnormal"],
                row["Temp_Fire_Alarm"],
                label,
            ),
        )

        # Send email if label is 1
        if label == 1 and email:
            send_email(email, i)
            anomaly_data_points.append(row)

    # Commit and close the connection
    conn.commit()
    conn.close()

    if anomaly_data_points:
        anomaly_df = pd.DataFrame(anomaly_data_points)
        anomaly_data_points_html = html.Div(
            [
                html.H4("Anomaly Data Points:"),
                html.Table(
                    [
                        html.Thead(
                            html.Tr([html.Th(col) for col in anomaly_df.columns])
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(anomaly_df.iloc[i][col])
                                        for col in anomaly_df.columns
                                    ]
                                )
                                for i in range(len(anomaly_df))
                            ]
                        ),
                    ]
                ),
            ]
        )
    else:
        anomaly_data_points_html = html.Div(
            [html.H4("Anomaly Data Points:"), html.P("No anomalies detected.")]
        )
    return (
        fig_smoke,
        fig_temp,
        fig_scatter,
        fig_pca,
        fig_proportions,
        anomaly_data_points_html,
    )


# Callback to handle saving the dashboard
@app.callback(
    Output("download-dashboard-html", "data"),
    Input("save-dashboard", "n_clicks"),
    State("hist-smoke", "figure"),
    State("hist-temp", "figure"),
    State("scatter-plot", "figure"),
    State("pca-plot", "figure"),
    State("label-proportions", "figure"),
    prevent_initial_call=True,
)
def save_dashboard(
    n_clicks, fig_smoke, fig_temp, fig_scatter, fig_pca, fig_proportions
):
    if n_clicks > 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dashboard_{timestamp}.html"

        fig_smoke_html = pio.to_html(fig_smoke, full_html=False)
        fig_temp_html = pio.to_html(fig_temp, full_html=False)
        fig_scatter_html = pio.to_html(fig_scatter, full_html=False)
        fig_pca_html = pio.to_html(fig_pca, full_html=False)
        fig_proportions_html = pio.to_html(fig_proportions, full_html=False)

        html_content = f"""
        <html>
        <head>
            <title>Dashboard</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
        </head>
        <body class="container">
            <h1 class="my-4">Anomaly Detection Dashboard</h1>
            <div class="row">
                <div class="col-md-4">{fig_smoke_html}</div>
                <div class="col-md-4">{fig_temp_html}</div>
                <div class="col-md-4">{fig_scatter_html}</div>
            </div>
            <div class="row my-4">
                <div class="col-md-6">{fig_pca_html}</div>
                <div class="col-md-6">{fig_proportions_html}</div>
            </div>
        </body>
        </html>
        """

        return dict(content=html_content, filename=filename)

    return dict(content="", filename="")


if __name__ == "__main__":
    app.run_server(debug=False)

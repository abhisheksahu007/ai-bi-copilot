import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="InsightPilot AI", layout="wide")

st.title("InsightPilot AI")
st.caption("AI Business Intelligence Copilot")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="latin1")

    st.subheader("Raw Data")
    st.write(df.head())

    # Cleaning
    df = df.drop_duplicates()
    df = df.fillna(method="ffill")

    st.subheader("Data Summary")
    summary = df.describe()
    st.write(summary)

    # Visualization
    st.subheader("Trend Visualization")
    num_cols = df.select_dtypes(include="number").columns

    for col in num_cols[:2]:
        fig = plt.figure()
        plt.plot(df[col])
        plt.title(col)
        st.pyplot(fig)
        st.write("Numeric columns:", num_cols)
        st.write("Shape:", df.shape)



    # Anomaly Detection
    st.subheader("Anomaly Detection")
    model = IsolationForest(contamination=0.05)
    df["anomaly"] = model.fit_predict(df[num_cols])
    anomalies = df[df["anomaly"] == -1]
    st.write(anomalies)

    # Generate Insights (basic)
    st.subheader("AI Insights")
    insights = f"""
    Dataset contains {len(df)} records.

    Key Observations:
    - Average values show operational trends.
    - {len(anomalies)} anomalies detected.
    - Monitor unusual spikes for risk mitigation.
    """

    st.write(insights)

    # Generate PDF
    if st.button("Generate Executive Report"):
        doc = SimpleDocTemplate("report.pdf")
        styles = getSampleStyleSheet()
        content = [
            Paragraph("Executive Report", styles['Heading1']),
            Paragraph(insights, styles['BodyText'])
        ]
        doc.build(content)

        with open("report.pdf", "rb") as f:
            st.download_button("Download Report", f, file_name="Report.pdf")
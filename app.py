import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from prophet import Prophet
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from gtts import gTTS
import os
from openai import AzureOpenAI

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="InsightPilot AI",
    page_icon="📊",
    layout="wide"
)

# ------------------ SIDEBAR ------------------
st.sidebar.title("InsightPilot AI")
st.sidebar.caption("Decision Intelligence Engine")
st.sidebar.info("Upload your data to generate insights instantly.")

# ------------------ TITLE ------------------
st.title("📊 InsightPilot AI")
st.caption("AI Business Intelligence Copilot")

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    # ---------- LOAD DATA (handles encoding issues) ----------
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except:
        df = pd.read_csv(uploaded_file, encoding="latin1")

    df.columns = df.columns.str.strip()

    st.success("File uploaded successfully!")

    # ---------- DATA CLEANING ----------
    df = df.drop_duplicates()
    df = df.fillna(method="ffill").fillna(0)

    st.subheader("Preview Data")
    st.dataframe(df.head(), use_container_width=True)

    # ---------- METRICS ----------
    num_cols = df.select_dtypes(include="number").columns

    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric("Numeric Columns", len(num_cols))

    st.divider()

    # ---------- VISUALIZATION ----------
    st.subheader("Trend Visualization")

    for col in num_cols[:2]:
        fig = plt.figure()
        plt.plot(df[col])
        plt.title(col)
        st.pyplot(fig)

    st.divider()

    # ---------- ANOMALY DETECTION ----------
    st.subheader("Anomaly Detection")

    anomalies = pd.DataFrame()

    if len(num_cols) > 0:
        clean_df = df[num_cols].dropna()

        if not clean_df.empty:
            model = IsolationForest(contamination=0.05, random_state=42)
            df.loc[clean_df.index, "anomaly"] = model.fit_predict(clean_df)

            anomalies = df[df["anomaly"] == -1]
            st.write(f"Anomalies detected: {len(anomalies)}")
            st.dataframe(anomalies)

        else:
            st.info("Numeric data empty after cleaning.")
    else:
        st.warning("No numeric columns available.")

    st.divider()

    # ---------- AI INSIGHTS (AZURE OPENAI) ----------
    st.subheader("AI Generated Insights")

    try:
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-01",
            azure_endpoint=os.getenv("AZURE_ENDPOINT")
        )

        prompt = f"""
        Analyze this dataset summary and provide:
        - key insights
        - risks
        - business recommendations

        Summary:
        {df.describe().to_string()}
        """

        response = client.chat.completions.create(
            model=os.getenv("AZURE_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt}]
        )

        insights = response.choices[0].message.content

    except Exception as e:
        insights = "AI insights unavailable. Check API configuration."
        st.warning(str(e))

    st.write(insights)

    st.divider()

    # ---------- NATURAL LANGUAGE Q&A ----------
    st.subheader("Ask Questions About Your Data")

    question = st.text_input("Ask in plain English")

    if question:
        try:
            q_prompt = f"""
            Answer the question based on this dataset summary:

            {df.describe().to_string()}

            Question: {question}
            """

            answer = client.chat.completions.create(
                model=os.getenv("AZURE_DEPLOYMENT"),
                messages=[{"role": "user", "content": q_prompt}]
            )

            st.write(answer.choices[0].message.content)

        except:
            st.warning("Unable to process question.")

    st.divider()

    # ---------- FORECASTING ----------
    st.subheader("Forecast Future Trend")

    try:
        date_col = df.columns[0]
        value_col = num_cols[0]

        forecast_df = df[[date_col, value_col]].dropna()
        forecast_df.columns = ["ds", "y"]
        forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

        model = Prophet()
        model.fit(forecast_df)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        fig2 = plt.figure()
        plt.plot(forecast["ds"], forecast["yhat"])
        plt.title("30-Day Forecast")
        st.pyplot(fig2)

    except:
        st.info("Forecast requires a date column.")

    st.divider()

    # ---------- VOICE INSIGHTS ----------
    st.subheader("Voice Summary")

    if st.button("Play Voice Insights"):
        tts = gTTS(insights)
        tts.save("voice.mp3")
        audio_file = open("voice.mp3", "rb")
        st.audio(audio_file.read(), format="audio/mp3")

    st.divider()

    # ---------- PDF REPORT ----------
    if st.button("Generate Executive Report"):
        doc = SimpleDocTemplate("report.pdf")
        styles = getSampleStyleSheet()

        content = [
            Paragraph("Executive Report", styles['Heading1']),
            Paragraph(insights, styles['BodyText'])
        ]

        doc.build(content)

        with open("report.pdf", "rb") as f:
            st.download_button("Download Report", f, file_name="Executive_Report.pdf")
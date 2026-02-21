import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from prophet import Prophet
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from openai import AzureOpenAI
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="InsightPilot AI", page_icon="📊", layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.title("InsightPilot AI")
st.sidebar.caption("Decision Intelligence Engine")
st.sidebar.info("Upload data to generate insights instantly.")

# ---------------- TITLE ----------------
st.title("📊 InsightPilot AI")
st.caption("AI Business Intelligence Copilot")

# ---------------- AZURE OPENAI CLIENT ----------------
client = None
if os.getenv("AZURE_OPENAI_KEY"):
    try:
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-01",
            azure_endpoint=os.getenv("AZURE_ENDPOINT")
        )
    except:
        client = None

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    # ---------- LOAD DATA SAFELY ----------
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except:
        df = pd.read_csv(uploaded_file, encoding="latin1")

    df.columns = df.columns.str.strip()

    # remove commas & currency symbols
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(",", "")
        df[col] = df[col].str.replace("₹", "").str.replace("$", "")

    # ---------- CLEAN ----------
    df = df.drop_duplicates()
    df = df.fillna(method="ffill").fillna(0)

    # convert numeric columns safely
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    st.success("File uploaded successfully!")
    st.subheader("Preview Data")
    st.dataframe(df.head(), use_container_width=True)

    # ---------- METRICS ----------
    num_cols = df.select_dtypes(include=["number"]).columns

    st.subheader("Key Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Numeric Columns", len(num_cols))

    st.divider()

    # ---------- VISUALIZATION ----------
    st.subheader("Trend Visualization")

    plotted = False
    for col in num_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) > 5:
            fig = plt.figure()
            plt.plot(series)
            plt.title(col)
            st.pyplot(fig)
            plotted = True
            break

    if not plotted:
        st.info("No numeric data available for visualization.")

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

    # ---------- AI INSIGHTS ----------
    st.subheader("AI Generated Insights")

    insights = None

    if client:
        try:
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
                messages=[
                    {"role": "system", "content": "You are a business intelligence analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            insights = response.choices[0].message.content

        except:
            insights = None

    if not insights:
        insights = f"""
        Dataset contains {len(df)} rows.
        Numeric columns detected: {len(num_cols)}.
        {len(anomalies)} anomalies detected.
        Monitor unusual spikes and trends.
        """

    st.write(insights)

    st.divider()

    # ---------- NATURAL LANGUAGE Q&A ----------
    st.subheader("Ask Questions About Your Data")

    question = st.text_input("Ask in plain English")

    if question:
        if client:
            try:
                q_prompt = f"""
                Dataset summary:
                {df.describe().to_string()}

                Question: {question}
                """

                answer = client.chat.completions.create(
                    model=os.getenv("AZURE_DEPLOYMENT"),
                    messages=[{"role": "user", "content": q_prompt}]
                )

                st.write(answer.choices[0].message.content)
            except:
                st.info("AI Q&A unavailable.")
        else:
            st.info("Configure Azure OpenAI to enable AI answers.")

    st.divider()

    # ---------- FORECAST ----------
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
        st.info("Forecast requires a valid date column.")

    st.divider()

    # ---------- VOICE INSIGHTS ----------
    st.subheader("Voice Summary")

    if st.button("Play Voice Insights"):
        try:
            from gtts import gTTS
            tts = gTTS(insights)
            tts.save("voice.mp3")
            audio_file = open("voice.mp3", "rb")
            st.audio(audio_file.read(), format="audio/mp3")
        except:
            st.info("Voice feature unavailable in this environment.")

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
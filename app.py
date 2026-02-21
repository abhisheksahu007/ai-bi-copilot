import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from prophet import Prophet
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from openai import AzureOpenAI
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="InsightPilot AI", page_icon="📊", layout="wide")

# ---------------- DARK MODE TOGGLE ----------------
dark_mode = st.sidebar.toggle("🌗 Dark Mode", value=False)

if dark_mode:
    st.markdown("""
        <style>
        .stApp { background-color: #0E1117; color: white; }
        </style>
    """, unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("📊 InsightPilot AI")
st.caption("AI Business Intelligence Copilot")

st.sidebar.title("InsightPilot AI")
st.sidebar.caption("Decision Intelligence Engine")

# ---------------- AZURE CLIENT ----------------
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

    # ---------- LOAD & CLEAN DATA ----------
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except:
        df = pd.read_csv(uploaded_file, encoding="latin1")

    df.columns = df.columns.str.strip()
    df = df.replace('[₹$,]', '', regex=True)
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    df = df.drop_duplicates()
    df = df.fillna(method="ffill").fillna(0)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    num_cols = df.select_dtypes(include="number").columns

    st.success("Data uploaded successfully")

    # ---------------- KPI METRICS WITH TRENDS ----------------
    st.subheader("Key Metrics")

    col1, col2, col3 = st.columns(3)

    if len(num_cols) > 0:
        metric_col = num_cols[0]
        latest = df[metric_col].iloc[-1]
        previous = df[metric_col].iloc[-2] if len(df) > 1 else latest
        delta = latest - previous
    else:
        latest = previous = delta = 0

    col1.metric("Rows", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric("Latest Value", round(latest, 2), delta=round(delta, 2))

    st.divider()

    # ---------------- AUTO CHART SUGGESTIONS ----------------
    st.subheader("Smart Chart Suggestions")

    if len(num_cols) >= 2:
        suggested_x = num_cols[0]
        suggested_y = num_cols[1]

        st.caption(f"Suggested: Relationship between {suggested_x} and {suggested_y}")

        fig = px.scatter(df, x=suggested_x, y=suggested_y, trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

    elif len(num_cols) == 1:
        fig = px.line(df, y=num_cols[0], title="Trend")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Not enough numeric data for suggestions.")

    st.divider()

    # ---------------- MULTI-CHART DASHBOARD ----------------
    st.subheader("Interactive Dashboard")

    if len(num_cols) > 0:
        selected_cols = st.multiselect(
            "Select metrics to visualize",
            num_cols,
            default=list(num_cols[:2])
        )

        if selected_cols:
            chart_type = st.selectbox(
                "Chart Type",
                ["Line", "Bar", "Area"]
            )

            if chart_type == "Line":
                fig = px.line(df, y=selected_cols)
            elif chart_type == "Bar":
                fig = px.bar(df, y=selected_cols)
            else:
                fig = px.area(df, y=selected_cols)

            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------- ANOMALY DETECTION ----------------
    st.subheader("Anomaly Detection")

    if len(num_cols) > 0:
        clean_df = df[num_cols].dropna()

        if not clean_df.empty:
            model = IsolationForest(contamination=0.05, random_state=42)
            df.loc[clean_df.index, "anomaly"] = model.fit_predict(clean_df)

            anomalies = df[df["anomaly"] == -1]
            st.write(f"Detected anomalies: {len(anomalies)}")

            if not anomalies.empty:
                st.dataframe(anomalies, use_container_width=True)

    st.divider()

    # ---------------- AI INSIGHTS ----------------
    st.subheader("AI Generated Insights")

    insights = None

    if client:
        try:
            prompt = f"""
            Provide key insights, risks, and recommendations based on:
            {df.describe().to_string()}
            """

            response = client.chat.completions.create(
                model=os.getenv("AZURE_DEPLOYMENT"),
                messages=[
                    {"role": "system", "content": "You are a business analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            insights = response.choices[0].message.content
        except:
            insights = None

    if not insights:
        insights = "AI insights unavailable. Configure Azure OpenAI."

    st.write(insights)

    st.divider()

    # ---------------- NATURAL LANGUAGE Q&A ----------------
    st.subheader("Ask Questions")

    question = st.text_input("Ask about your data")

    if question and client:
        q_prompt = f"{df.describe().to_string()}\nQuestion: {question}"
        answer = client.chat.completions.create(
            model=os.getenv("AZURE_DEPLOYMENT"),
            messages=[{"role": "user", "content": q_prompt}]
        )
        st.write(answer.choices[0].message.content)

    st.divider()

    # ---------------- FORECAST ----------------
    st.subheader("Forecast")

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

        fig = px.line(forecast, x="ds", y="yhat", title="30-Day Forecast")
        st.plotly_chart(fig, use_container_width=True)

    except:
        st.info("Forecast requires a valid date column.")

    st.divider()

    # ---------------- VOICE SUMMARY ----------------
    if st.button("🔊 Play Voice Insights"):
        try:
            from gtts import gTTS
            tts = gTTS(insights)
            tts.save("voice.mp3")
            audio = open("voice.mp3", "rb")
            st.audio(audio.read(), format="audio/mp3")
        except:
            st.info("Voice feature unavailable.")

    # ---------------- PDF REPORT ----------------
    if st.button("📄 Generate Executive Report"):
        doc = SimpleDocTemplate("report.pdf")
        styles = getSampleStyleSheet()

        content = [
            Paragraph("Executive Report", styles['Heading1']),
            Paragraph(insights, styles['BodyText'])
        ]

        doc.build(content)

        with open("report.pdf", "rb") as f:
            st.download_button("Download Report", f, file_name="Executive_Report.pdf")
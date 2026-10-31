import streamlit as st
import pandas as pd
import joblib
import random

st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="wide")

model = joblib.load("model.pkl")
ref = joblib.load("reference.pkl")
metrics = joblib.load("metrics.pkl")
mean, std = ref["mean"], ref["std"]
features = list(mean.index)
importance = pd.Series(model.feature_importances_, index=features)


def risk_level(p):
    if p >= 0.7:
        return "High"
    if p >= 0.3:
        return "Medium"
    return "Low"


st.markdown("<h1 style='text-align:center'>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray'>Upload transactions and let the Random Forest model flag fraud</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Sensitivity")
    threshold = st.slider("Fraud threshold (%)", 1, 99, 50) / 100
    st.caption("Lower = catches more fraud but gives more false alarms. Higher = fewer false alarms but may miss fraud.")
    st.header("Risk levels")
    st.markdown("- 🟢 Low: below 30%\n- 🟠 Medium: 30% to 70%\n- 🔴 High: above 70%")
    st.header("About")
    st.markdown("- Model: Random Forest\n- Data: European cardholder transactions\n- Features: Time, Amount, V1–V28")

file = st.file_uploader("Browse and upload a CSV file", type=["csv"])
if file is None:
    st.info("Upload a file to begin.")
    st.stop()

data = pd.read_csv(file)
missing = [f for f in features if f not in data.columns]
if missing:
    st.error(f"Your file is missing required columns, for example: {missing[:5]}")
    st.stop()

X = data[features]
probs = model.predict_proba(X)[:, 1]
preds = (probs >= threshold).astype(int)

result = data.copy()
result["Prediction"] = ["Fraud" if p == 1 else "Genuine" for p in preds]
result["Fraud probability %"] = (probs * 100).round(2)
result["Risk"] = [risk_level(p) for p in probs]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total transactions", len(result))
c2.metric("Genuine", int((preds == 0).sum()))
c3.metric("Fraud detected", int((preds == 1).sum()))
c4.metric("High risk", int((result["Risk"] == "High").sum()))

tab1, tab2, tab3 = st.tabs(["📋 All predictions", "🔍 Single transaction", "📊 Model performance"])

with tab1:
    def highlight(r):
        c = "background-color:#ffd6d6;color:black" if r["Prediction"] == "Fraud" else "background-color:#d6f5d6;color:black"
        return [c] * len(r)

    st.dataframe(result.head(1000).style.apply(highlight,axis=1),use_container_width=True)
    st.bar_chart(result["Risk"].value_counts())

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download all results", result.to_csv(index=False).encode(),
                       "all_predictions.csv", "text/csv")
    flagged = result[result["Prediction"] == "Fraud"]
    d2.download_button("⬇️ Download flagged transactions only", flagged.to_csv(index=False).encode(),
                       "flagged_transactions.csv", "text/csv")

with tab2:
    if st.button("🎲 Pick random transaction"):
        st.session_state["idx"] = random.randrange(len(data))
    i = st.session_state.get("idx")
    if i is not None and i < len(data):
        row = X.iloc[[i]]
        is_fraud = preds[i] == 1
        risk = risk_level(probs[i])
        left, right = st.columns(2)

        with left:
            st.subheader(f"Transaction #{i + 1}")
            m1, m2 = st.columns(2)
            m1.metric("Amount", f"{row['Amount'].iloc[0]:.2f}")
            m2.metric("Time", f"{row['Time'].iloc[0]:.0f}")
            st.write("Fraud probability")
            st.progress(float(probs[i]))
            st.caption(f"{probs[i] * 100:.2f}%")
            if risk == "High":
                st.error("🔴 HIGH RISK")
            elif risk == "Medium":
                st.warning("🟠 MEDIUM RISK")
            else:
                st.success("🟢 LOW RISK")
            st.write(f"**Prediction (threshold {threshold * 100:.0f}%):**", "FRAUD" if is_fraud else "GENUINE")
            if "Class" in data.columns:
                st.write("Actual label:", "Fraud" if data["Class"].iloc[i] == 1 else "Genuine")

        with right:
            z = (row.iloc[0] - mean) / std
            score = (z.abs() * importance).sort_values(ascending=False).head(5)
            if is_fraud:
                st.subheader("Why it looks like fraud")
                st.write("These important features are far from normal genuine transactions:")
            else:
                st.subheader("Why it looks genuine")
                st.write("Even its most unusual important features are close to normal genuine transactions:")
            for f in score.index[:3]:
                st.markdown(f"- **{f}** = {row[f].iloc[0]:.2f} (genuine average {mean[f]:.2f}, {abs(z[f]):.1f} standard deviations away)")
            st.caption("Top 5 features by influence")
            st.bar_chart(score)

        with st.expander("See full transaction data"):
            st.dataframe(row)

with tab3:
    cm = metrics["cm"]
    st.subheader("Model performance on unseen test data")
    st.caption(f"Measured on {sum(map(sum, cm)):,} test transactions that the model never saw during training.")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    p2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    p3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
    p4.metric("F1-score", f"{metrics['f1'] * 100:.2f}%")
    st.markdown("- **Precision:** of the transactions flagged as fraud, how many were really fraud\n"
                "- **Recall:** of all real fraud, how much was caught\n"
                "- **F1-score:** balance of precision and recall")
    st.write("**Confusion matrix**")
    st.dataframe(pd.DataFrame(cm, index=["Actual Genuine", "Actual Fraud"],
                              columns=["Predicted Genuine", "Predicted Fraud"]))
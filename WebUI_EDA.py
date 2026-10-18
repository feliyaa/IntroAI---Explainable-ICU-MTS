import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

# MODEL TO BE CHANGE
model = tf.keras.models.load_model("./baseline_rnn.keras")
preprocessor = joblib.load("./preprocessor")

WINDOW = 58


st.title("Sepsis diagnostic")


def apply_preprocess(patient_df, preprocessor):
    expected = list(preprocessor.feature_names_in_)
    for col in expected:
        if col not in patient_df.columns:
            patient_df[col] = np.nan
    patient_df = patient_df[expected]
    out = preprocessor.transform(patient_df)
    try:
        cols = [
            c.split("__", 1)[1] if "__" in c else c
            for c in preprocessor.get_feature_names_out()
        ]
    except Exception:
        cols = expected
    return pd.DataFrame(out, columns=cols)

def add_padding(patient_df, window):
    # Assumes that the patient has fewer rows than the window.
    for i in patient_df["SepsisLabel"]:
        print(i)
    patient_df = patient_df.drop(columns=["Patient_ID", "SepsisLabel"])
    patient_df = patient_df.sort_values("Hour")

    num_feature = patient_df.shape[1]
    patient_np = patient_df.to_numpy()
    num_padding_rows = int(window) - patient_df.shape[0]
    all_zeros = np.zeros(num_feature)
    new_patient = []
    count = 0
    for i in range(num_padding_rows):
        new_patient.append(all_zeros)
        count += 1
    for hour in patient_np:
        new_patient.append(hour)
        count += 1
    new_patient_np = np.asarray(new_patient)
    print(count)
    print(len(new_patient))
    return new_patient_np


def infer(X_infer):
    y = model.predict(X_infer)
    label = None
    prob = y[0][0]
    if prob >= 0.5:
        label = "Sepsis"

    if prob < 0.5:
        label = "No Sepsis"

    return label, prob

patient_csv = st.file_uploader("Upload patient CSV", type=["csv"])

if patient_csv is not None:
    patient_df = pd.read_csv(patient_csv)
    if "Patient_ID" in patient_df.columns and patient_df["Patient_ID"].nunique() > 1:
        st.warning("Multiple Patient_IDs found; using the first one.")
        patient_df = patient_df[patient_df["Patient_ID"] == patient_df["Patient_ID"].iloc[0]]
    process_df = apply_preprocess(patient_df=patient_df, preprocessor=preprocessor)
    label, prob = infer(add_padding(process_df, WINDOW)[None, ...])
    if label == "Sepsis":
        st.markdown(f":red[CAUTION: THIS PATIENT IS LIKELY TO HAVE SEPSIS, CONFIDENT: ({prob:.4f})]")
    else:
        st.markdown(f":green[This patient is not likely to have sepsis, keep eye on the patient, CONFIDENT: ({prob:.4f})]")

    st.subheader("Patient time-series")
    candidates = [c for c in ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"] if c in patient_df.columns]
    selected = [c for c in candidates if st.checkbox(c, value=True)]
    if selected:
        st.line_chart(patient_df.sort_values("Hour").set_index("Hour")[selected])

## Added EDA - Raden Jovan

if patient_csv is not None:
    patient_csv.seek(0)
    eda_df = pd.read_csv(patient_csv)

    if "Patient_ID" in eda_df.columns and not eda_df.empty:
        first_id = eda_df["Patient_ID"].iloc[0]
        eda_df = eda_df[eda_df["Patient_ID"] == first_id].copy()

    st.divider()
    st.header("Patient Data Exploration")
    st.caption(
        "Original measurements before filling missing values, "
        "scaling or padding.")

    vital_columns = [
        col for col in
        ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
        if col in eda_df.columns]

    if eda_df.empty or not vital_columns:
        st.info("No usable patient records or vital-sign columns found.")
    else:
        vital_data = eda_df[vital_columns].apply(
            pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

        # Overview
        left, middle, right = st.columns(3)

        left.metric("Recorded rows", len(eda_df))
        middle.metric("Vital-sign columns", len(vital_columns))
        right.metric(
            "Missing vital-sign values",
            f"{vital_data.isna().mean().mean() * 100:.1f}%")

        # Missing-data chart
        st.subheader("Which measurements are missing?")

        missing = (
            vital_data.isna().mean().mul(100)
            .sort_values(ascending=False)
            .rename("Missing (%)"))

        st.bar_chart(missing)

        st.caption("Higher bars mean fewer available readings. ""Missing and invalid values are excluded from the "
            "statistics below—not counted as zero.")

        # Summary statistics
        st.subheader("Vital-sign summary")

        summary = vital_data.describe().T[
            ["count", "mean", "50%", "std", "min", "max"]].rename(columns={
            "count": "Available readings",
            "mean": "Average",
            "50%": "Median",
            "std": "Standard deviation",
            "min": "Minimum",
            "max": "Maximum"})

        st.dataframe(summary.round(2))

        st.caption("Average and median describe typical values. "
            "Standard deviation describes how much readings vary. "
            "Measurements retain their original units.")

        # Distribution chart
        st.subheader("How are the values distributed?")

        selected_measurement = st.selectbox(
            "Choose a vital sign",vital_columns,
            key="eda_vital_distribution")

        values = vital_data[selected_measurement].dropna()

        if values.empty:
            st.info("No available readings for this vital sign.")
        else:
            counts, edges = np.histogram(values, bins=15)

            distribution = pd.DataFrame({
                "Value range": [
                    f"{edges[i]:.2f}–{edges[i + 1]:.2f}"
                    for i in range(len(counts))],
                "Readings": counts})

            st.dataframe(distribution)
            st.bar_chart(
                pd.DataFrame({
                    "Value (bin midpoint)": (
                        edges[:-1] + edges[1:]
                    ) / 2,
                    "Readings": counts}).set_index("Value (bin midpoint)")
            )

            st.caption("Taller bars show value ranges recorded more often. "
                "This describes the selected patient, not all patients "
                "in the training dataset.")
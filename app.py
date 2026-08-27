import streamlit as st
import pandas as pd
import numpy as np

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# Header Section
# ============================================================

st.markdown(
    """
    # 📊 Vendor Invoice Intelligence Portal

    ### AI-Driven Freight Cost Prediction & Invoice Risk Flagging

    This internal analytics portal leverages machine learning to:

    - **Forecast freight costs accurately**
    - **Detect risky or abnormal vendor invoices**
    - **Reduce financial leakage and manual workload**
    """
)

st.divider()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("🤖 Model Selection")

selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.markdown(
    """
    ### 💼 Business Impact

    - Improved cost forecasting
    - Reduced invoice fraud & anomalies
    - Faster finance operations
    - Reduced manual workload
    """
)


# ============================================================
# FREIGHT COST PREDICTION
# ============================================================

if selected_model == "Freight Cost Prediction":

    st.subheader("🚚 Freight Cost Prediction")

    st.markdown(
        """
        **Objective:**

        Predict the expected freight cost for a vendor invoice
        using **Invoice Dollars**.
        """
    )

    with st.form("freight_form"):

        col1, col2 = st.columns(2)

        with col1:

            quantity = st.number_input(
                "Quantity",
                min_value=1,
                value=1200
            )

        with col2:

            dollars = st.number_input(
                "Invoice Dollars",
                min_value=1.0,
                value=18500.0
            )

        submit_freight = st.form_submit_button(
            "🔮 Predict Freight Cost"
        )

    if submit_freight:

        # Input for freight model
        input_data = {
            "Dollars": [dollars]
        }

        prediction = predict_freight_cost(input_data)

        predicted_freight = prediction["Predicted_Freight"].iloc[0]

        st.success(
            "Prediction completed successfully."
        )

        st.metric(
            label="💰 Estimated Freight Cost",
            value=f"${predicted_freight:,.2f}"
        )


# ============================================================
# INVOICE FLAG PREDICTION
# ============================================================

else:

    st.subheader(
        "🚨 Invoice Manual Approval Prediction"
    )

    st.markdown(
        """
        **Objective:**

        Predict whether a vendor invoice should be
        **flagged for manual approval** based on abnormal
        invoice and purchasing patterns.
        """
    )

    with st.form("invoice_flag_form"):

        col1, col2, col3 = st.columns(3)

        # ----------------------------------------------------
        # Column 1
        # ----------------------------------------------------

        with col1:

            invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value=1,
                value=50
            )

            freight = st.number_input(
                "Freight Cost",
                min_value=0.0,
                value=1.73
            )

        # ----------------------------------------------------
        # Column 2
        # ----------------------------------------------------

        with col2:

            invoice_dollars = st.number_input(
                "Invoice Dollars",
                min_value=1.0,
                value=352.95
            )

            total_item_quantity = st.number_input(
                "Total Item Quantity",
                min_value=1,
                value=162
            )

        # ----------------------------------------------------
        # Column 3
        # ----------------------------------------------------

        with col3:

            total_item_dollars = st.number_input(
                "Total Item Dollars",
                min_value=1.0,
                value=2476.0
            )

        submit_flag = st.form_submit_button(
            "🔍 Evaluate Invoice Risk"
        )

    if submit_flag:

        # Input features must match training features
        input_data = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars]
        }

        flag_prediction = predict_invoice_flag(
            input_data
        )

        predicted_flag = flag_prediction[
            "Predicted_Flag"
        ].iloc[0]

        is_flagged = bool(predicted_flag)

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        if is_flagged:

            st.error(
                "🚨 Invoice requires **MANUAL APPROVAL**"
            )

            st.warning(
                "The machine learning model has identified "
                "this invoice as potentially risky."
            )

        else:

            st.success(
                "✅ Invoice is **SAFE for AUTO-APPROVAL**"
            )

            st.info(
                "No significant risk was detected by the model."
            )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Invoice Intelligence Machine Learning Project | "
    "Python + SQL + Scikit-Learn + Streamlit"
)
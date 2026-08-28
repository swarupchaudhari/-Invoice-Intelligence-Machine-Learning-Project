# 📊 Vendor Invoice Intelligence System

## 🚚 Freight Cost Prediction & Invoice Risk Flagging

An end-to-end Machine Learning and Analytics project designed to help finance and procurement teams **predict vendor freight costs** and **identify potentially risky invoices requiring manual approval**.

The system combines **SQL, Python, Pandas, Scikit-Learn, SQLite, Machine Learning, and Streamlit** to create a complete data-to-deployment solution.

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Business Objectives](#-business-objectives)
- [Business Problem](#-business-problem)
- [Data Sources](#-data-sources)
- [Data Processing](#-data-processing)
- [Feature Engineering](#-feature-engineering)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Machine Learning Models](#-machine-learning-models)
- [Model Evaluation](#-model-evaluation)
- [Invoice Risk Flagging](#-invoice-risk-flagging)
- [Application](#-application)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Example Prediction](#-example-prediction)
- [Business Impact](#-business-impact)
- [Future Improvements](#-future-improvements)
- [Technologies Used](#-technologies-used)
- [Author](#-author)

---

# 📌 Project Overview

The **Vendor Invoice Intelligence System** is an end-to-end machine learning project that solves two important finance and procurement problems:

### 1️⃣ Freight Cost Prediction

Predict the expected freight cost of a vendor invoice using invoice-level information such as:

- Invoice Dollars
- Quantity

This can help finance teams with:

- Budget forecasting
- Cost estimation
- Vendor negotiations
- Identifying unusual freight charges

### 2️⃣ Invoice Risk Flagging

Classify vendor invoices as either:

```text
0 → Normal Invoice
1 → Risky / Requires Manual Approval
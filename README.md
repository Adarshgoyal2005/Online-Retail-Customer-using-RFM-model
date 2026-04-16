# 🛒 Online Retail Customer Analysis using RFM Model

A full-stack Machine Learning web application built with **Flask** that enables businesses to analyze customer behavior using the **RFM (Recency, Frequency, Monetary)** model. Upload your retail transaction data and get instant insights on customer segmentation, churn prediction, and sales forecasting.

---

## 🚀 Features

- 📊 **RFM Analysis** — Automatically computes Recency, Frequency, and Monetary scores for each customer
- 🎯 **Customer Segmentation** — K-Means clustering to classify customers into 4 segments:
  - 🔴 Low Value Customers
  - 🟡 Potential Customers
  - 🟢 Loyal Customers
  - ⭐ High Value Customers
- 📉 **Churn Prediction** — Identifies at-risk customers using:
  - Logistic Regression
  - Decision Tree Classifier
- 📈 **Sales Forecasting** — Predicts next month's revenue using historical trends
- 📋 **Interactive Dashboard** — Visual charts and KPI summaries
- 📁 **Multi-format Support** — Upload `.xlsx`, `.xls`, `.csv`, `.tsv`, or `.txt` files

---

## 🗂️ Project Structure

```
retail_ml_app/
│
├── app.py                  # Flask application & routes
├── requirements.txt        # Python dependencies
├── test_pipeline.py        # Pipeline unit tests
│
├── modules/
│   ├── __init__.py
│   ├── data_processor.py   # Data loading, cleaning & RFM computation
│   ├── segmentation.py     # K-Means customer segmentation
│   ├── churn_prediction.py # Logistic Regression & Decision Tree churn models
│   ├── sales_prediction.py # Sales forecasting model
│   └── visualizer.py       # Chart generation (Matplotlib)
│
├── templates/
│   ├── index.html          # Upload page
│   ├── results.html        # Results summary page
│   └── dashboard.html      # Visual charts dashboard
│
└── static/
    └── style.css           # Application styling
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Adarshgoyal2005/Online-Retail-Customer-using-RFM-model.git
cd Online-Retail-Customer-using-RFM-model
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python app.py
```

Then open your browser and go to: **http://localhost:5000**

---

## 📋 Input Data Format

Your dataset should contain the following columns (the app auto-detects them):

| Column | Description |
|--------|-------------|
| `InvoiceNo` / `Invoice` | Transaction ID |
| `CustomerID` / `Customer ID` | Unique customer identifier |
| `InvoiceDate` | Date of transaction |
| `Quantity` | Number of items purchased |
| `UnitPrice` / `Price` | Price per unit |

> ✅ The app supports the standard **UCI Online Retail II** dataset format out of the box.

---

## 🧠 ML Pipeline

```
File Upload
    │
    ▼
Data Cleaning (remove nulls, negatives, cancellations)
    │
    ▼
RFM Computation (Recency, Frequency, Monetary per customer)
    │
    ▼
K-Means Clustering (4 segments)
    │
    ├──► Churn Prediction (Logistic Regression + Decision Tree)
    │
    ├──► Sales Forecasting (next month revenue estimate)
    │
    └──► Dashboard (charts + KPI summary)
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `flask` | Web framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `scikit-learn` | ML models (KMeans, Logistic Regression, Decision Tree) |
| `matplotlib` | Chart generation |
| `openpyxl` | Excel file reading |

---

## 🖼️ Screenshots

> Upload your transaction file on the home page → View customer segments, churn %, and sales forecast on the results page → Explore visual charts on the dashboard.

---

## 👨‍💻 Author

**Adarsh Goyal**
- GitHub: [@Adarshgoyal2005](https://github.com/Adarshgoyal2005)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

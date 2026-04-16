import os
import uuid
import traceback
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename

from modules.data_processor import load_and_clean, compute_rfm
from modules.segmentation import run_kmeans, segment_summary
from modules.sales_prediction import predict_sales
from modules.churn_prediction import predict_churn
from modules.visualizer import generate_all_charts

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"xlsx", "xls", "csv", "tsv", "txt"}

app = Flask(__name__)
app.secret_key = "retail_ml_secret_2024"
app.jinja_env.globals.update(enumerate=enumerate)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

# In-memory store for processed data (keyed by UUID per session)
_data_store = {}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Route: Upload Page ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── Route: Process Upload ─────────────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Only .xlsx / .xls / .csv / .tsv / .txt files are supported.")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        # ── ML Pipeline ──────────────────────────────────────────────────────
        df, col_mapping = load_and_clean(filepath)
        rfm_df = compute_rfm(df)
        rfm_df = run_kmeans(rfm_df)

        seg_summary_data = segment_summary(rfm_df)
        sales_data = predict_sales(df)
        churn_data = predict_churn(rfm_df)
        charts = generate_all_charts(rfm_df, sales_data)

        # ── Summary KPIs ─────────────────────────────────────────────────────
        stats = {
            "total_customers":   len(rfm_df),
            "total_transactions": len(df),
            "total_revenue":     round(df["TotalAmount"].sum(), 2),
            "next_month_sales":  sales_data["next_month_sales"],
            "churn_pct":         churn_data["churn_pct"],
            "lr_accuracy":       churn_data["lr_accuracy"],
            "dt_accuracy":       churn_data["dt_accuracy"],
            "r2_score":          sales_data["r2_score"],
        }

        # Top 20 customers for table
        top_customers = (
            rfm_df.nlargest(20, "Monetary")[
                ["Customer ID", "Recency", "Frequency", "Monetary", "Segment"]
            ]
            .round(2)
            .to_dict(orient="records")
        )

        detected_cols = {role: col for role, col in col_mapping.items() if col}

        # ── Store data server-side; keep only a small key in session ─────────
        data_key = str(uuid.uuid4())
        _data_store[data_key] = {
            "stats":        stats,
            "seg_summary":  seg_summary_data,
            "top_customers": top_customers,
            "charts":       charts,
            "sales_data":   sales_data,
            "churn_data":   churn_data,
            "filename":     filename,
            "detected_cols": detected_cols,
        }
        session["data_key"] = data_key

        return redirect(url_for("results"))

    except Exception as e:
        traceback.print_exc()
        flash(f"Error during processing: {str(e)}")
        return redirect(url_for("index"))
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# ── Route: Results Summary Page ───────────────────────────────────────────────
@app.route("/results")
def results():
    data_key = session.get("data_key")
    if not data_key or data_key not in _data_store:
        flash("Session expired or no data found. Please upload a file first.")
        return redirect(url_for("index"))

    data = _data_store[data_key]
    return render_template(
        "results.html",
        stats=data["stats"],
        seg_summary=data["seg_summary"],
        top_customers=data["top_customers"],
        churn_data=data["churn_data"],
        filename=data["filename"],
        detected_cols=data["detected_cols"],
    )


# ── Route: Dashboard (Charts) Page ───────────────────────────────────────────
@app.route("/dashboard")
def dashboard():
    data_key = session.get("data_key")
    if not data_key or data_key not in _data_store:
        flash("Session expired or no data found. Please upload a file first.")
        return redirect(url_for("index"))

    data = _data_store[data_key]
    return render_template(
        "dashboard.html",
        stats=data["stats"],
        charts=data["charts"],
        sales_data=data["sales_data"],
        churn_data=data["churn_data"],
        filename=data["filename"],
        seg_summary=data["seg_summary"],
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)

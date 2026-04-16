import pandas as pd
from modules.data_processor import load_and_clean, compute_rfm
from modules.segmentation import run_kmeans, segment_summary
from modules.sales_prediction import predict_sales
from modules.churn_prediction import predict_churn
from modules.visualizer import generate_all_charts
import traceback

try:
    # 1. Create a dummy CSV file
    csv_data = """Customer ID,Invoice,InvoiceDate,Quantity,Price
C1,INV1,2023-01-01,2,10.0
C1,INV2,2023-02-01,1,10.0
C2,INV3,2023-01-15,5,20.0
C3,INV4,2023-03-01,1,100.0
C3,INV5,2023-03-10,2,50.0
C4,INV6,2023-03-12,1,10.0
C5,INV7,2023-03-15,2,15.0
"""
    with open("dummy.csv", "w") as f:
        f.write(csv_data)

    df, col_mapping = load_and_clean("dummy.csv")
    print("Cleaned DF:")
    print(df)
    
    rfm_df = compute_rfm(df)
    print("\nRFM DF:")
    print(rfm_df)
    
    rfm_df = run_kmeans(rfm_df)
    print("\nKMeans RFM DF:")
    print(rfm_df)
    
    seg_summary_data = segment_summary(rfm_df)
    print("\nSegment Summary:")
    print(seg_summary_data)
    
    sales_data = predict_sales(df)
    print("\nSales Data:")
    print(sales_data)
    
    churn_data = predict_churn(rfm_df)
    print("\nChurn Data:")
    print(churn_data)
    
    charts = generate_all_charts(rfm_df, sales_data)
    print("\nCharts generated successfully.")
    
except Exception as e:
    traceback.print_exc()

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import pyarrow.dataset as ds
import pyarrow.compute as pc
from huggingface_hub import HfFileSystem
import os
import time
import json
import traceback

app = FastAPI(title="Avezkibuckett-API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
HF_TOKEN = os.getenv("hf_token") 

# Initialize HfFileSystem
global_fs = HfFileSystem(token=HF_TOKEN)

# EXACT PATH using HfFileSystem bucket syntax
BUCKET_ID = "crozzx/avezkibuckett"
FILE_PATH = f"buckets/{BUCKET_ID}/idx_phone.1.parquet"

@app.get("/")
def root():
    return {"status": "online", "message": "Avezkibuckett API is Live."}

@app.get("/search")
def search(mobile: str = Query(..., description="10-digit mobile number")):
    start_time = time.time()
    
    try:
        if len(mobile) != 10 or not mobile.isdigit():
            return Response(
                content=json.dumps({"error": "Please provide a valid 10-digit mobile number."}, indent=4), 
                media_type="application/json"
            )

        # Load the single file directly
        dataset = ds.dataset(FILE_PATH, filesystem=global_fs, format="parquet")
        
        # Filter for the phone number
        filter_expr = pc.equal(pc.field("phoneNumber"), mobile)
        
        # Only read necessary columns to save RAM
        columns_to_read = ["name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber", "address"]
        
        # Iterate in batches to prevent OOM just in case
        scanner = dataset.scanner(
            filter=filter_expr,
            columns=columns_to_read,
            use_threads=False,
            batch_size=1024
        )
        
        raw_records = []
        for batch in scanner.to_batches():
            if batch.num_rows > 0:
                raw_records = batch.to_pylist()
                break # Stop reading once found
                
        elapsed = time.time() - start_time
        
        response_data = {
            "meta": {
                "credits": "Made by Mr. Anonymous",
                "search_query": mobile,
                "total_records": len(raw_records),
                "execution_time": f"{elapsed:.3f}s"
            },
            "results": []
        }

        if not raw_records:
            response_data["message"] = "No record found in the database."
            return Response(
                content=json.dumps(response_data, indent=4), 
                media_type="application/json"
            )
            
        for i, row in enumerate(raw_records, 1):
            clean_row = {}
            if row.get("name"): clean_row["name"] = row["name"]
            if row.get("fathersName"): clean_row["father_name"] = row["fathersName"]
            if row.get("phoneNumber"): clean_row["mobile"] = row["phoneNumber"]
            if row.get("aadharNumber"): clean_row["aadhar_number"] = row["aadharNumber"]
            if row.get("otherNumber"): clean_row["alternate_mobile"] = row["otherNumber"]
            if row.get("address"): clean_row["address"] = row["address"]
            
            clean_row["result_id"] = i
            response_data["results"].append(clean_row)
            
        return Response(
            content=json.dumps(response_data, indent=4, default=str), 
            media_type="application/json"
        )
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"CRITICAL ERROR:\n{error_trace}")
        return Response(
            content=json.dumps({"error": f"Search failed: {str(e)}"}, indent=4), 
            media_type="application/json"
        )

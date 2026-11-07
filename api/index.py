from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import duckdb
import os
import time
import json
import traceback

app = FastAPI(title="Avezkadata-API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
HF_TOKEN = os.getenv("hf_token") 

# Direct URL for the Hugging Face Dataset
# DuckDB's httpfs will use HTTP Range requests to read only the needed rows
BASE_URL = "https://huggingface.co/datasets/crozzx/Avezkadata/resolve/main"
FILE_LIST = [f"{BASE_URL}/idx_phone.{i}_part0{j}.parquet" for i in range(3) for j in range(1, 7)]

@app.get("/")
def root():
    return {"status": "online", "message": "Avezkadata API is Live."}

@app.get("/search")
def search(mobile: str = Query(..., description="10-digit mobile number")):
    start_time = time.time()
    
    try:
        if len(mobile) != 10 or not mobile.isdigit():
            return Response(
                content=json.dumps({"error": "Please provide a valid 10-digit mobile number."}, indent=4), 
                media_type="application/json"
            )

        con = duckdb.connect(database=':memory:')
        
        # Set writable home dir for extension install
        con.execute("SET home_directory='/tmp';")
        
        # Load httpfs for HTTP support
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")
        
        # Authenticate using HTTP Headers (for private repos)
        if HF_TOKEN:
            con.execute(f"""
                CREATE SECRET hf_secret (
                    TYPE HTTP,
                    EXTRA_HTTP_HEADERS MAP {{'Authorization': 'Bearer {HF_TOKEN}'}}
                );
            """)
        
        # The query uses DuckDB's read_parquet. 
        # DuckDB streams over HTTP and only reads what is needed.
        query = f"""
            SELECT name, fathersName AS father_name, phoneNumber AS mobile, 
                   aadharNumber AS aadhar_number, otherNumber AS alternate_mobile, 
                   address
            FROM read_parquet({FILE_LIST})
            WHERE phoneNumber = ?
            LIMIT 10
        """
        
        result = con.execute(query, [mobile]).fetchall()
        columns = [desc[0] for desc in con.description]
        
        con.close()
        
        elapsed = time.time() - start_time
        
        response_data = {
            "meta": {
                "credits": "Made by Mr. Anonymous",
                "search_query": mobile,
                "total_records": len(result),
                "execution_time": f"{elapsed:.3f}s"
            },
            "results": []
        }

        if not result:
            response_data["message"] = "No record found in the database."
            return Response(
                content=json.dumps(response_data, indent=4), 
                media_type="application/json"
            )
            
        for i, row in enumerate(result, 1):
            raw_row = dict(zip(columns, row))
            # Remove empty/null values to keep response clean
            clean_row = {k: v for k, v in raw_row.items() if v is not None and v != ""}
            
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

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import duckdb
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

# Direct Resolve URL for the specific file
FILE_URL = "https://huggingface.co/buckets/crozzx/avezkibuckett/resolve/main/idx_phone.1.parquet"

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

        con = duckdb.connect(database=':memory:')
        
        con.execute("SET home_directory='/tmp';")
        
        # INSTALL and LOAD httpfs to handle HTTPS URLs
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")
        
        # Authenticate the request to Hugging Face using HTTP headers
        if HF_TOKEN:
            con.execute(f"""
                CREATE SECRET hf_secret (
                    TYPE HTTP,
                    EXTRA_HTTP_HEADERS MAP {{'Authorization': 'Bearer {HF_TOKEN}'}}
                );
            """)
        
        # Query the exact URL
        query = f"""
            SELECT name, fathersName AS father_name, phoneNumber AS mobile, 
                   aadharNumber AS aadhar_number, otherNumber AS alternate_mobile, 
                   address
            FROM read_parquet('{FILE_URL}')
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

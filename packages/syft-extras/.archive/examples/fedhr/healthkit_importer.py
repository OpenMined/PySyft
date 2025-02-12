import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi.responses import HTMLResponse, JSONResponse

current_dir = Path(__file__).parent


async def get_page():
    with open(current_dir / "page.html") as f:
        return f.read()


async def handler(request, form_data=None):
    request_type = request.method
    headers = dict(request.headers)

    response = {}
    if request_type == "GET":
        response = await get_page()
        return HTMLResponse(response)
    elif request_type == "POST":
        try:
            response = await handle_post(request)
        except Exception as e:
            print("Error handling POST request:", e)
            response = {"error": f"Bad request: {e}", "cookies": request.cookies}
    else:
        response = {"error": f"unexpected method: {request.method}"}

    log_data = f"Timestamp: {datetime.now()}\n"
    log_data += f"Request Type: {request.method}\n"
    log_data += "Headers:\n"
    for header, value in headers.items():
        log_data += f"  {header}: {value}\n"

    log_data += "Body:\n"
    body = await request.body()
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    else:
        body = str(body)  # Convert to string if it's not bytes
    log_data += body + "\n"

    # Write log to a file
    with open(current_dir / "request_log.txt", "a") as log_file:
        log_file.write(log_data + "\n" + str(response) + "-" * 50 + "\n")
    return JSONResponse(response)


async def handle_post(request):
    csv_file_path = current_dir / "hr.csv"

    # Read the body and check if it's JSON
    try:
        body = await request.json()

        # Check if the JSON has the structure for metrics data
        if "data" in body and "metrics" in body["data"]:
            metrics_data = body["data"]["metrics"][0]["data"]

            # Filter data to only keep "date" and "Avg" (renamed to "avg")
            filtered_data = [
                {"date": entry["date"], "avg": entry["Avg"]}
                for entry in metrics_data
                if "date" in entry and "Avg" in entry
            ]

            # Convert to DataFrame
            new_df = pd.DataFrame(filtered_data)

            # Read existing CSV file if it exists
            if os.path.exists(csv_file_path):
                existing_df = pd.read_csv(csv_file_path)
                # Append new data and remove duplicates
                combined_df = pd.concat([existing_df, new_df]).drop_duplicates(
                    subset=["date"]
                )
            else:
                combined_df = new_df

            # Ensure the date column has a space before the timezone for consistent formatting
            combined_df["date"] = combined_df["date"].str.replace(
                r"(\d{2}:\d{2}:\d{2})(\+\d{4})", r"\1 \2", regex=True
            )

            # Convert the date column to datetime, allowing for mixed formats
            combined_df["date"] = pd.to_datetime(combined_df["date"], format="mixed")

            # Sort the combined DataFrame by the "date" column
            combined_df = combined_df.sort_values(by="date")

            # Save the combined DataFrame back to the CSV
            combined_df.to_csv(csv_file_path, index=False)

            # Add a message indicating that data was saved
            response = {
                "message": "Filtered, sorted, and appended data saved to metrics_data.csv"
            }

    except (json.JSONDecodeError, ValueError) as e:
        response = {"error": f"Error decoding or processing data: {e}"}

    return response

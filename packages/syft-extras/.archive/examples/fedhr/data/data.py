from fastapi.responses import JSONResponse
from pathlib import Path
import pandas as pd

current_dir = Path(__file__).parent


async def handler(request, form_data=None):
    csv_file_path = current_dir / ".." / "hr.csv"
    df = pd.read_csv(csv_file_path, header=None, names=["timestamp", "value"])

    # Convert the timestamp column to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Filter out any rows where the timestamp could not be parsed
    df = df.dropna(subset=["timestamp"])

    # Convert the value column to numeric, coercing errors to NaN
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Drop rows where the value could not be converted to a number or is missing
    df = df.dropna(subset=["value"])

    # Check if the timestamp column is already timezone-aware
    if df["timestamp"].dt.tz is None:
        # Make it timezone-aware if not
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        # Convert to UTC if it is already timezone-aware
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    # Set the timestamp as the index
    df.set_index("timestamp", inplace=True)

    # Resample the data to 1-minute intervals, averaging the values
    resampled_df = df.resample("1T").mean()

    # Filter out rows with NaN values after resampling
    resampled_df = resampled_df.dropna()

    # Get the current time as a timezone-aware datetime
    now_utc = pd.Timestamp.now(tz="UTC")

    # Keep only the last 48 hours of data
    recent = resampled_df.loc[now_utc - pd.Timedelta(hours=12) :]

    # Reset the index to have timestamp as a column
    recent.reset_index(inplace=True)

    # Convert the timestamp to a string in ISO format
    recent["timestamp"] = recent["timestamp"].astype(str)

    # Convert the DataFrame to a JSON format
    json_output = recent.to_dict(orient="records")

    # Return JSON response
    return JSONResponse(content=json_output)


# data = [
#     {"timestamp": "2024-11-04 12:18:00", "value": 91.0},
#     {"timestamp": "2024-11-04 12:22:00", "value": 88.0},
#     {"timestamp": "2024-11-04 12:26:55", "value": 66.0},
#     {"timestamp": "2024-11-04 12:28:55", "value": 98.5},
#     {"timestamp": "2024-11-04 12:29:55", "value": 100.98},
#     {"timestamp": "2024-11-04 12:30:55", "value": 107.5},
#     {"timestamp": "2024-11-04 12:31:55", "value": 104.0},
#     {"timestamp": "2024-11-04 12:33:55", "value": 93.0},
#     {"timestamp": "2024-11-04 12:35:00", "value": 90.0},
#     {"timestamp": "2024-11-04 12:41:00", "value": 93.0},
#     {"timestamp": "2024-11-04 12:48:31", "value": 90.0},
#     {"timestamp": "2024-11-04 12:52:00", "value": 73.0},
# ]

import json
import logging
import traceback
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from helper import FileInfo, list_files_with_metadata
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from syftbox.lib import Client

# Configure logging
logging.basicConfig(
    filename="server.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

current_dir = Path(__file__).parent

client_config = Client.load(
    "/Users/madhavajay/dev/syft/.clients/bigquery@openmined.org/config/config.json"
)
datasite = "bigquery@openmined.org"

app = FastAPI()

model_name = "llama3.1:latest"


def sandbox_path(file_path):
    abs_path = Path(file_path).resolve()

    if not abs_path.is_relative_to(client_config.datasite_path):
        logger.warning(
            f"Attempted access to {abs_path} outside sandbox {client_config.datasite_path}"
        )
        raise ValueError(
            f"The provided path: {abs_path} is outside the allowed sandbox directory: {client_config.datasite_path}."
        )

    return abs_path


@tool
def list_all_files_in_datasite() -> FileInfo:
    """
    Recursively lists all files in a directory and gathers metadata for each file.

    Returns:
        dict: A dictionary where each key is the file path and the value is another dictionary containing
              file metadata (hash, extension, size in MB, MIME type, and Magika group).
    """
    return list_files_with_metadata(client_config.datasite_path)


@tool
def get_csv_columns_and_types(csv_path) -> str:
    """
    Reads a CSV file using pandas and returns a dictionary containing
    the column names and their corresponding data types.

    Parameters:
    csv_path (str): The file path to the CSV file.

    Returns:
    dict: A dictionary where keys are column names and values are data types.
          Example: {'column1': dtype('int64'), 'column2': dtype('object')}
    """
    try:
        csv_path = sandbox_path(csv_path)
        logger.info(f"Getting CSV columns and types for {csv_path}")
        df = pd.read_csv(csv_path)
        column_info = {col: df[col].dtype for col in df.columns}
        return str(column_info)
    except Exception as e:
        logger.error(f"Error reading CSV columns: {e}")
        return {"status": "error", "message": str(e)}


@tool
def get_csv_head_as_dict(csv_path) -> str:
    """
    Reads a CSV file using pandas and returns the first `n` rows as a list of
    dictionaries, where each dictionary represents a row with column names as keys.

    Parameters:
    csv_path (str): The file path to the CSV file.

    Returns:
    list: A list of dictionaries, each representing a row from the CSV.
          Example: [{'column1': value1, 'column2': value2}, ...]
    """
    try:
        csv_path = sandbox_path(csv_path)
        logger.info(f"Getting CSV head as dict for {csv_path}")
        df = pd.read_csv(csv_path)
        head_data = df.head(5).to_dict(orient="records")
        return str(head_data)
    except Exception as e:
        logger.error(f"Error reading CSV head: {e}")
        return {"status": "error", "message": str(e)}


tool_mapping = {
    "list_all_files_in_datasite": list_all_files_in_datasite,
    "get_csv_columns_and_types": get_csv_columns_and_types,
    "get_csv_head_as_dict": get_csv_head_as_dict,
}
tools = list(tool_mapping.values())

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
).bind_tools(tools)


async def ask(content):
    try:
        prompt = HumanMessage(content=content)
        messages = [prompt]

        ai_message = llm.invoke(messages)
        logger.debug(f"AI message received: {ai_message}")
        for tool_call in ai_message.tool_calls:
            selected_tool = tool_mapping[tool_call["name"].lower()]
            tool_output = selected_tool.invoke(tool_call["args"])

            tool_output_str = json.dumps(tool_output)
            messages.append(ToolMessage(tool_output_str, tool_call_id=tool_call["id"]))

            logger.debug(f"Tool output: {tool_output_str}")

        for m in messages:
            logger.debug(f"Message type: {type(m)} - Content: {m}")

        result = llm.invoke(messages)
        logger.debug(f"Final AI response: {result}")
        return result
    except Exception as e:
        error = {"content": {"status": "error", "message": str(e)}}
        logger.debug(f"Ask error: {error}. {traceback.format_exc()}")
        return error


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def chat(request: Request):
    logger.info("Serving chat page")
    with open(current_dir / "page.html") as f:
        return HTMLResponse(f.read())


@app.post("/ask", include_in_schema=False)
async def query(request: Request):
    try:
        body = await request.json()
        logger.info(f"Received query: {body}")
        response = await ask(body["message"])
        if hasattr(response, "content"):
            content = response.content
        elif "content" in response:
            content = response["content"]

        print("Got response.content", content)
        print("Got response.content type", type(content))
        result = {"response": str(content)}
        logger.debug(f"Query response: {result}")
        print(">>> type of result", type(result))
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Error response: {e}")


main_app = FastAPI()
main_app.mount("/chat", app)


def main() -> None:
    debug = True
    uvicorn.run(
        "main:main_app" if debug else main_app,
        host="0.0.0.0",
        port=9082,
        log_level="debug" if debug else "info",
        reload=debug,
        reload_dirs="./",
    )


if __name__ == "__main__":
    main()

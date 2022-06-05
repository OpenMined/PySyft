
from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every
from datetime import datetime
import sentry_sdk

app = FastAPI()

pre_run_cells = {}
post_run_cells = {}
sentry_sdk.init(
    "https://bd20175e36374f1c9edef90c9b0ba94c@o488706.ingest.sentry.io/6465580",
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=1.0,
    send_default_pii=True
)    
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/pre_run_cell")
def pre_run_cell(ip, id):
    pre_run_cell[id] = {"ip": ip, "time": datetime.now()}
    return 

@app.post("/post_run_cell")
def post_run_cell(ip, id):
    post_run_cell[id] = {"ip": ip, "time": datetime.now()}
    return

@app.get("/running_vms")
def get_running_vms():
    running_cells = check_cells()
    return running_cells

@repeat_every(seconds=10)
def check_cells_and_notif():
    running_cells = check_cells()
    send_notif(running_cells)
    return 

def check_cells():
    running_cells = []
    for id in pre_run_cells:
        if id not in post_run_cells:
            running_cells.append(pre_run_cell[id])
    return running_cells

def send_notif(running_cells):
    for cell in running_cells:
        sentry_sdk.capture_exception(Exception(f"notebook blocked {cell["ip"]}"))
    return

@app.post("/clear_cells")
def clear_cells():
    global pre_run_cells, post_run_cells
    pre_run_cells = {}
    post_run_cells = {}
    return



# stdlib
from datetime import datetime

# third party
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_utils.tasks import repeat_every
from network_health import check_ip_port
from network_health import check_login_via_syft
from network_health import check_metadata_api
from network_health import check_network_status
from network_health import get_listed_public_networks
import sentry_sdk

app = FastAPI()
# origins = [
#     "http://localhost.tiangolo.com",
#     "https://localhost.tiangolo.com",
#     "http://localhost",
#     "http://localhost:8080",
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


pre_run_cells = {}
post_run_cells = {}
sentry_sdk.init(
    "https://bd20175e36374f1c9edef90c9b0ba94c@o488706.ingest.sentry.io/6465580",
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=1.0,
    send_default_pii=True,
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/pre_run_cell")
def pre_run_cell(ip, id, raw_cell):
    pre_run_cells[id] = {"ip": ip, "time": datetime.now(), "running_cell": raw_cell}
    return


@app.post("/post_run_cell")
def post_run_cell(ip, id):
    post_run_cells[id] = {"ip": ip, "time": datetime.now()}
    return


@app.get("/running_vms")
def get_running_vms():
    running_cells = check_cells()
    return running_cells


@app.on_event("startup")
@repeat_every(seconds=5 * 60)
def check_cells_and_notif():
    running_cells = check_cells()
    send_notif(running_cells)
    print("SENT sentry")
    return


def check_cells():
    running_cells = []
    for id in pre_run_cells:
        if id not in post_run_cells:
            running_cells.append(pre_run_cells[id])
    return running_cells


def send_notif(running_cells):
    for cell in running_cells:
        ip = cell["ip"]
        sentry_sdk.capture_exception(Exception(f"notebook blocked {ip}"))
    return


@app.post("/clear_cells")
def clear_cells():
    global pre_run_cells, post_run_cells
    pre_run_cells = {}
    post_run_cells = {}
    return


@app.get("/check_network")
def check_network():
    status_table_list = []
    network_list = get_listed_public_networks()

    for network in network_list:
        host_url = network["host_or_ip"]
        status = {}

        status["host_or_ip"] = host_url
        status["ssh_status"] = check_ip_port(host_url, port=80)
        status["ping_status"] = check_network_status(host_url)
        status["api_status"] = check_metadata_api(host_url)
        status["login_status"] = check_login_via_syft(host_url)

        status_table_list.append(status)
    return status_table_list

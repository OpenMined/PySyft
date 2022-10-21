# stdlib
import subprocess
import sys

# third party
import pandas as pd


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def auto_detect_domain_host_ip(silent: bool = False) -> str:
    ip_address = subprocess.check_output("echo $(curl -s ifconfig.co)", shell=True)
    domain_host_ip = ip_address.decode("utf-8").strip()
    if "google.colab" not in sys.modules:
        if not silent:
            print(f"Your DOMAIN_HOST_IP is: {domain_host_ip}")
    else:
        if not silent:
            print(
                "Google Colab detected, please manually set the `DOMAIN_HOST_IP` variable"
            )
        domain_host_ip = ""
    return domain_host_ip


def download_dataset_as_dataframe(dataset_url):
    def load_data_as_df(file_path):
        df = pd.read_csv(file_path)
        unique_trade_flows = df["Trade Flow"].unique()
        print()
        print(f"{color.BOLD}Data shape: {color.END}", df.shape, end="\n\n")
        print(f"{color.BOLD}Data Columns: {color.END}", list(df.columns), end="\n\n")
        print(
            f"{color.BOLD}Unique Trade Flows: {color.END}", unique_trade_flows, end="\n"
        )

        return df

    data = load_data_as_df(dataset_url)

    return data

## amke stats
# stat_folder = "./crypto/data"
# os.makedirs(stat_folder, exist_ok=True)
# stats_array = np.zeros(len(stats.keys())).astype(int)
# value = HE.encryptInt(stats_array)
# part_path = f"{stat_folder}/totals"
# value.save(part_path)

# # encode
# max_tv_id = 300_000 # random guess looking at their website
# tv_count_array = np.zeros(max_tv_id)

# tmdb_ids = np.array(list(value_counts.keys()))
# counts = np.array(value_counts.values)
# tv_count_array[tmdb_ids] += counts

# # decode
# non_zero_indices = np.nonzero(tv_count_array)[0].astype(int)
# non_zero_values = tv_count_array[non_zero_indices].astype(int)
# non_zero_dict = {int(k): int(v) for k, v in dict(zip(non_zero_indices, non_zero_values)).items()}
# print(non_zero_dict)
# return non_zero_dict
# HE = Pyfhel()
# HE.contextGen(scheme='bfv', n=2**15, t_bits=20)
# HE.keyGen()
# import os
# os.makedirs("crypto", exist_ok=True)
# HE.save_secret_key("crypto/pyfhel.secret")
# HE.save_public_key("crypto/pyfhel.pk")
# HE.save_context("crypto/pyfhel.context")

# brew install cmake zlib llvm libomp
# export CC=/opt/homebrew/opt/llvm/bin/clang
# uv pip install Pyfhel

# LDFLAGS="-L/opt/homebrew/opt/llvm/lib/c++ -L/opt/homebrew/opt/llvm/lib -lunwind"
# If you need to have llvm first in your PATH, run:
#   echo 'export PATH="/opt/homebrew/opt/llvm/bin:$PATH"' >> ~/.zshrc

# For compilers to find llvm you may need to set:
#   export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
#   export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"

# ln -s /Users/madhavajay/dev/syft/users/madhava/me@madhavajay.com/public/datasets/netflix_tmdb_imdb/NetflixViewingHistory_TMDB_IMDB.csv ./dfs_0.csv
# /// script
# dependencies = [
#    "pandas==2.2.3",
#    "syftbox==0.1.0",
#    "phe==1.5.0",
#    "pyfhel==3.4.2",
# ]
#
# [tool.uv.sources]
# syftbox = { path = "/Users/madhavajay/dev/syft", editable = true }
# ///
from typing_extensions import Optional

__name__ = "netflix_stats"
__author__ = "madhava@openmined.org"


def input_reader(private: bool = False, datasite: Optional[str] = None):
    import os

    import pandas as pd

    from syftbox.lib import extract_leftmost_email, sy_path

    df = None
    files = os.listdir("./inputs/dfs")
    for file in files:
        if file.startswith("dfs_"):
            full_path = f"./inputs/dfs/{file}"
            target_path = os.readlink(full_path)
            link_datasite = extract_leftmost_email(target_path)
            if link_datasite == datasite:
                df = pd.read_csv(sy_path(full_path, resolve_private=private))
                break
    inputs = {"datasite": datasite, "df": df}
    return inputs


def output_writer(result, private: bool = False):
    import json

    output_path = "./output/result/result.json"
    if not private:
        output_path = output_path.replace(".json", ".mock.json")
    with open(output_path, "w") as f:
        f.write(json.dumps(result))


# START YOUR CODE


def netflix_stats(datasite, df):
    import os

    crypto_folder = "./crypto"
    completed_sentinel = f"{crypto_folder}/{datasite}"
    if os.path.exists(completed_sentinel):
        print("✅ Already generated 🔐 Homomorphically Encrypted Stats")
        return

    import datetime
    import os

    import numpy as np
    import pandas as pd
    from Pyfhel import Pyfhel
    from Pyfhel.PyCtxt import PyCtxt

    HE = Pyfhel()
    HE.load_context(f"{crypto_folder}/pyfhel.context")
    HE.load_secret_key(f"{crypto_folder}/pyfhel.secret")
    HE.load_public_key(f"{crypto_folder}/pyfhel.pk")

    current_year = datetime.datetime.now().year
    df["netflix_date"] = pd.to_datetime(df["netflix_date"])
    year_df = df[df["netflix_date"].dt.year == current_year]
    year_tv_df = year_df[year_df["tmdb_media_type"] == "tv"]
    year_tv_df["day_of_week"] = year_tv_df["netflix_date"].dt.day_name()
    total_time = year_tv_df["imdb_runtime_minutes"].sum()
    total_views = len(year_tv_df)
    total_unique_show_views = year_tv_df["imdb_id"].nunique()
    # day_counts = year_tv_df["day_of_week"].value_counts()
    # favorite_day = list(day_counts.to_dict().keys())[0]
    # year_tv_df["day_of_week"] = year_tv_df["netflix_date"].dt.weekday
    # change to an int as a numpy array so we can add them

    value_counts = year_tv_df["tmdb_id"].value_counts().astype(int)
    # top_5_value_counts = {k: int(v) for k, v in value_counts.sort_values(ascending=False)[0:5].items()}

    stats = {
        "total_time": int(total_time),
        "total_views": int(total_views),
        "total_unique_show_views": int(total_unique_show_views),
        # "year_fav_day": str(favorite_day),
    }

    stat_folder = f"./{crypto_folder}/data"
    part_path = f"{stat_folder}/totals"
    slice_folder = f"{stat_folder}/view_counts"
    exists_files_folders = [stat_folder, part_path, slice_folder]
    os.makedirs(stat_folder, exist_ok=True)
    os.makedirs(slice_folder, exist_ok=True)

    for path in exists_files_folders:
        if not os.path.abspath(path):
            raise Exception(f"Requires {stat_folder} to finish syncing")

    imdb_id_files = os.listdir(slice_folder)
    if len(imdb_id_files) < 10:
        raise Exception(f"Requires {slice_folder} to finish syncing")

    # create totals
    stats_array = np.zeros(len(stats.keys())).astype(int)
    value = HE.encryptInt(stats_array)
    value.save(part_path)

    # write stats to encrypted array
    stats_array = np.zeros(len(stats)).astype(int)
    for i, value in enumerate(stats.values()):
        stats_array[i] = int(value)

    value = PyCtxt(pyfhel=HE)

    value.load(part_path)
    value += stats_array
    value.save(part_path)

    max_tv_id = 300_000  # just a guess
    slice_size = 30_000  # max size of the above HE context

    # create imdb_id slices
    counter = 0
    for i in range(0, max_tv_id + 1, slice_size):
        tv_count_array = np.zeros(slice_size).astype(int)
        tv_count_slice = HE.encryptInt(tv_count_array)
        part_path = f"{slice_folder}/tmdb_id_{counter:02}"
        tv_count_slice.save(part_path)
        counter += 1

    # write imdb_id value counts to chunked arrays
    for k, v in value_counts.items():
        imdb_id = int(k)
        index = imdb_id // slice_size
        sub_index = imdb_id % slice_size
        tv_count_slice = PyCtxt(pyfhel=HE)
        part_path = f"{slice_folder}/tmdb_id_{index:02}"
        empty_array = np.zeros(slice_size).astype(int)
        empty_array[sub_index] += int(v)
        tv_count_slice.load(part_path)
        tv_count_slice += empty_array
        tv_count_slice.save(part_path)

    with open(f"{crypto_folder}/{datasite}", "w") as f:
        print("✅ Writing 🔐 Homomorphically Encrypted Stats")
        f.write(str(datetime.datetime.now()))


# END YOUR CODE


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process some input.")
    parser.add_argument("--private", action="store_true", help="Run in private mode")
    parser.add_argument("--datasite", help="Datasite running operation")
    args = parser.parse_args()

    print(f"Running: {__name__} from {__author__}")
    inputs = input_reader(private=args.private, datasite=args.datasite)
    print("> Reading Inputs", inputs)

    output = netflix_stats(**inputs)

    print("> Writing Outputs", output)
    output_writer(output, private=args.private)
    print(f"> ✅ Running {__name__} Complete!")


main()

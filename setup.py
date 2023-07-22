import os
import requests
import csv


def fetch_data() -> bool:
    dirname = "data"
    def _download(url):
        filename = url.split('/')[-1]
        r = requests.get(url, stream=True)
        with open(os.path.join("data",filename), 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
            return filename

    url = "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
    os.makedirs("data", exist_ok=True)
    _download(url)
    gzip_file = os.path.join(dirname, "WISDM_ar_latest.tar.gz")
    fromdir = os.path.join(dirname, "WISDM_ar_v1.1")
    os.system(f"tar -zxvf {gzip_file} -C data")
    os.system(f"mv {fromdir}/* {dirname}")
    os.system(f"rm -rf {fromdir}")
    return True


def data_to_csv():
    def _read_csv(filename):
        with open(filename) as f:
            e = f.read()
        e = (
            e.replace(",\n", ",,")
            .replace("\n", "")
            .replace(",;", ",")
            .replace(";", ",")
            .split(",")
        )
        return e

    def _write_csv(filename, data):
        with open(filename, "w") as f:
            writer = csv.writer(f)
            for idx in range(len(data) // 6):
                writer.writerow(data[6 * idx : 6 * idx + 6])


    data = _read_csv("../data/WISDM_ar_v1.1_raw.txt")
    _write_csv("../data/WISDM_ar_v1.1.csv", data)


def main():
    if not os.path.exists("./data/WISDM_ar_v1.1.csv"):
        fetch_data()
        data_to_csv()
        print("data fetched & formatted")
    else:
        print("data already exists skip fetching & formatting")

import csv


def read_csv(filename):
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


def write_csv(filename, data):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        for idx in range(len(data) // 6):
            writer.writerow(data[6 * idx : 6 * idx + 6])


if __name__ == "__main__":
    data = read_csv("../data/WISDM_ar_v1.1_raw.txt")
    # print(data)
    write_csv("../data/WISDM_ar_v1.1.csv", data)

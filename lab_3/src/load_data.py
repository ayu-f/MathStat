import pandas as pd

def load_from_csv(file_name: str):
    data = pd.read_csv(file_name, encoding='1251', sep=';')

    return data


def load_octave(file_name: str):
    with open(file_name) as file:
        a, b = file.readline().split(' ')
        w = [float(line) for line in file]

    return float(a), float(b), w
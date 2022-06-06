from json import load
from optparse import Values

import numpy as np
from load_data import load_from_csv, load_octave
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt


def plot_data(data):
    for d in data:
        x = range(1, d.values[:, 0].size + 1)
        plt.plot(x, d.values[:, 0])
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.savefig("./report/images/data.png")
    plt.figure()


def plot_interval(data):
    x = range(1, data[:, 0].size + 1)
    plt.vlines(x, data[:, 0], data[:, 1], colors="c")


if __name__ == "__main__":
    data = []

    data.append(load_from_csv("data/Ch1_800nm_0.2.csv"))
    data.append(load_from_csv("data/Ch2_800nm_0.2.csv"))

    plot_data(data)

    eps = 1e-4
    size_range = range(2)
    for i in size_range:
        data[i].values[:, 1] = data[i].values[:, 0] + eps
        data[i].values[:, 0] -= eps
        plot_interval(data[i].values)
        plt.xlabel('n')
        plt.ylabel('mV')
        plt.savefig(f"./report/images/data_and_intervals{i + 1}.png")
        plt.figure()

    a, b, w = [], [], []
    for i in size_range:
        a_tmp, b_tmp, w_tmp = load_octave(f"data/Ch{i + 1}.txt")
        a.append(a_tmp)
        b.append(b_tmp)
        w.append(w_tmp)

    for i in size_range:
        data[i].values[:, 1] += (eps * pd.DataFrame(w[i])).values[:, 0]
        data[i].values[:, 0] -= (eps * pd.DataFrame(w[i])).values[:, 0]

    for i in size_range:
        plt.hist(w[i], histtype="stepfilled", color="m")
        plt.xlabel('w')
        plt.ylabel('N')
        plt.title(f'w Ch{i + 1}-800nm_0.2.csv')
        plt.savefig(f"./report/images/hist{i + 1}.png")
        plt.figure()

    for i in size_range:
        plot_interval(data[i].values)
        plt.plot([1, 200], [1 * b[i] + a[i], 200 * b[i] + a[i]])
        plt.title(f'Ch{i + 1}-800nm_0.2.csv')
        plt.xlabel('n')
        plt.ylabel('mV')
        plt.savefig(f"./report/images/di{i + 1}.png")
        plt.figure()

    data_fixed = []
    for i in size_range:
        x = pd.DataFrame(range(1, data[i].values[:, 0].size + 1))
        plt.vlines(x.values[:, 0], data[i].values[:, 0] - b[i] * x.values[:, 0],
                   data[i].values[:, 1] - b[i] * x.values[:, 0], colors="c")
        plt.plot([x.values[:, 0][0], x.values[:, 0][-1]], [a[i], a[i]], color='blue')
        plt.title(f'Ch{i + 1}-800nm_0.2.csv')
        plt.xlabel('n')
        plt.ylabel('mV')
        plt.savefig(f"./report/images/interval_new{i + 1}.png")
        plt.figure()
        data_fixed.append([data[i].values[:, 0] - b[i] * x.values[:, 0], data[i].values[:, 1] - b[i] * x.values[:, 0]])

    for i in size_range:
        x = pd.DataFrame(range(1, data[i].values[:, 0].size + 1))
        plt.hist((data_fixed[i][0] + data_fixed[i][1]) / 2, color="m")
        plt.xlabel('mV')
        plt.ylabel('N')
        plt.title(f'Ch{i + 1}-800nm_0.2.csv')
        plt.savefig(f"./report/images/hist_interval{i + 1}.png")
        plt.figure()

    R_interval = [0.001 * i + 1 for i in range(150)]
    Jaccars = []


    def countJakkar(R):
        data1_new = [[data_fixed[0][0][i] * R, data_fixed[0][1][i] * R] for i in range(200)]
        all_data = data1_new + [[data_fixed[1][0][i], data_fixed[1][1][i]] for i in range(200)]
        min_inc = list(all_data[0])
        max_inc = list(all_data[0])
        for interval in all_data:
            min_inc[0] = max(min_inc[0], interval[0])
            min_inc[1] = min(min_inc[1], interval[1])
            max_inc[0] = min(max_inc[0], interval[0])
            max_inc[1] = max(max_inc[1], interval[1])
        JK = (min_inc[1] - min_inc[0]) / (max_inc[1] - max_inc[0])
        return JK


    for R in R_interval:
        Jaccars.append(countJakkar(R))

    optimal_x = opt.fmin(lambda x: -countJakkar(x), 0)
    print(optimal_x[0])

    min1 = opt.root(countJakkar, 1)
    max1 = opt.root(countJakkar, 3)
    print(min1.x, max1.x)

    plt.plot(R_interval, Jaccars, zorder=1, color="cyan")
    plt.scatter(optimal_x[0], countJakkar(optimal_x[0]), color="r")
    plt.scatter(min1.x, countJakkar(min1.x), label="$min R$=" + str(min1.x[0])[0:7], color="m", zorder=2)
    plt.scatter(max1.x, countJakkar(max1.x), label="$max R$=" + str(max1.x[0])[0:7], color="m", zorder=2)
    plt.legend()
    plt.xlabel('$R_{21}$')
    plt.ylabel('Jaccard')
    plt.title('Jaccard vs R')
    plt.savefig("report/images/jakkar.png")
    plt.figure()

    data1_new = [[data_fixed[0][0][i] * optimal_x[0], data_fixed[0][1][i] * optimal_x[0]] for i in range(200)]
    all_data = data1_new + [[data_fixed[1][0][i], data_fixed[1][1][i]] for i in range(200)]
    plt.xlabel('mV')
    plt.ylabel('N')
    plt.hist([(x[0] + x[1]) / 2 for x in all_data], density=True, color="m")
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 1))
    plt.savefig("report/images/jakkar_combined_hist.png")
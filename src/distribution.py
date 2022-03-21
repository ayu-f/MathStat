import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
from scipy import stats as st
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF

class Distribution:
    ITER_COUNT = 1000

    def __init__(self, distr_name=None, size=0):
        self.a = None
        self.b = None
        self.distribution_name = distr_name
        self.size = size
        self.random_numbers = None
        self.density = None

    def generate_distr(self):
        if self.distribution_name == "Normal":
            self.random_numbers = st.norm.rvs(size=self.size)
        elif self.distribution_name == "Uniform":
            a = -math.sqrt(3)
            step = 2*math.sqrt(3)
            self.random_numbers = st.uniform.rvs(size=self.size, loc=a, scale=step)
        elif self.distribution_name == "Cauchy":
            self.random_numbers = st.cauchy.rvs(size=self.size)
        elif self.distribution_name == "Poisson":
            self.random_numbers = st.poisson.rvs(mu=10, size=self.size)
        elif self.distribution_name == "Laplace":
            self.random_numbers = st.laplace.rvs(size=self.size)

    """
    task 1
    """
    def get_density(self):
        if self.distribution_name == "Normal":
            self.density = st.norm()
        elif self.distribution_name == "Uniform":
            a = -math.sqrt(3)
            step = 2*math.sqrt(3)
            self.density = st.uniform(loc=a, scale=step)
        elif self.distribution_name == "Cauchy":
            self.density = st.cauchy()
        elif self.distribution_name == "Poisson":
            mu = 10
            self.density = st.poisson(mu)
        elif self.distribution_name == "Laplace":
            self.density = st.laplace(scale=1/math.sqrt(2), loc=0)

    def plot_dens(self):
        self.generate_distr()
        self.get_density()
        fig, ax = plt.subplots(1, 1)  # Create a figure and a set of subplots

        ax.hist(self.random_numbers, density=True, histtype="stepfilled")  # for histogram
        if self.distribution_name == "Poisson":
            x = numpy.arange(self.density.ppf(0.01), self.density.ppf(0.99))  # evenly spaced values [)
        else:
            x = numpy.linspace(self.density.ppf(0.01), self.density.ppf(0.99), num=100)  # evenly spaced values []

        if self.distribution_name == "Poisson":
            ax.plot(x, self.density.pmf(x), "g")
        else:
            ax.plot(x, self.density.pdf(x), "g")

        ax.set_xlabel(f"{len(self.random_numbers)}")
        ax.set_ylabel("density")
        ax.set_title(self.distribution_name)
        plt.grid()
        plt.show()

    """
    task 2
    """
    def calc_characteristics(self):
        mean = list()
        median = list()
        z_r = list()
        z_q = list()
        z_tr = list()

        for i in range(self.ITER_COUNT):
            self.generate_distr()
            arr = sorted(self.random_numbers)

            mean.append(numpy.mean(arr))
            median.append(numpy.median(arr))
            z_r.append((arr[0] + arr[-1]) / 2)
            z_q.append(
                (arr[math.ceil(self.size * 1/4)] + arr[math.ceil(self.size * 3/4)]) / 2
            )
            z_tr.append(self.truncated_mean(arr))

        E = list()
        D = list()
        for elem in [mean, median, z_r, z_q, z_tr]:
            E.append(round(numpy.mean(elem), 6))
            D.append(round(numpy.std(elem) ** 2, 6))

        return E, D

    def truncated_mean(self, arr) -> float:
        r = int(len(arr) / 4)
        sum = 0
        for i in range(r + 1, len(arr) - r + 1):
            sum += arr[i]
        return (1 / (len(arr) - 2 * r)) * sum

    """
    task 3
    """
    def moustache(self, distribution):
        q_1, q_3 = numpy.quantile(distribution, [0.25, 0.75])
        return q_1 - 3 / 2 * (q_3 - q_1), q_3 + 3 / 2 * (q_3 - q_1)

    def count_outlier(self, distribution):
        x1, x2 = self.moustache(distribution)
        filtered = [x for x in distribution if x > x2 or x < x1]
        return len(filtered)

    def outlier_share(self):
        count = 0
        for i in range(self.ITER_COUNT):
            self.generate_distr()
            arr = sorted(self.random_numbers)
            count += self.count_outlier(arr)
        res = count / (self.size * self.ITER_COUNT)
        self.generate_distr()
        print(f"{self.distribution_name} Size {self.size}: {res}")

    @staticmethod
    def draw_boxplot(data, distr_name):
        sns.set_theme(style="whitegrid")
        sns.boxplot(data=data, palette='pastel', orient='h')
        sns.despine(offset=10)
        plt.xlabel("x")
        plt.ylabel("n")
        plt.title(distr_name)
        # plt.savefig(img_path)
        plt.show()

    """
    task 4
    """
    def get_probability_distr(self, distr, size):
        self.distribution_name = distr
        self.size = size
        self.generate_distr()
        self.a, self.b = -4, 4
        x = numpy.linspace(self.a, self.b, 1000)
        arr = sorted(self.random_numbers)
        pdf, cdf = None, None
        if distr == "Normal":
            pdf = st.norm.pdf(x)
            cdf = st.norm.cdf(x)
        elif distr == "Uniform":
            begin = -math.sqrt(3)
            step = 2 * math.sqrt(3)
            pdf = st.uniform.pdf(x, loc=begin, scale=step)
            cdf = st.uniform.cdf(x, loc=begin, scale=step)
        elif distr == "Cauchy":
            pdf = st.cauchy.pdf(x)
            cdf = st.cauchy.cdf(x)
        elif distr == "Poisson":
            self.a, self.b = 6, 14
            x = numpy.linspace(self.a, self.b, -self.a + self.b + 1)
            pdf = st.poisson(10).pmf(x)
            cdf = st.poisson(10).cdf(x)
        elif distr == "Laplace":
            pdf = st.laplace.pdf(x, loc=0, scale=1 / math.sqrt(2))
            cdf = st.laplace.cdf(x, loc=0, scale=1 / math.sqrt(2))

        return arr, pdf, cdf, x

    def plot_ecdf(self, distr_list, sizes):
        sns.set_style("whitegrid")

        for distr in distr_list:
            figures, axs = plt.subplots(ncols=3, figsize=(14, 6))

            for i, size in enumerate(sizes):
                observations, pdf, cdf, x = self.get_probability_distr(distr, size)
                axs[i].plot(x, cdf, color="blue", label="cdf")
                ecdf = ECDF(observations)
                axs[i].plot(x, ecdf(x), color="red", label="ecdf")
                """sns.ecdfplot(
                    data=observations,
                    label="ecdf",
                    ax=axs[i],
                )"""
                axs[i].legend(loc="upper left")
                axs[i].set(xlabel="x", ylabel="F(x)")
                axs[i].set_title(f"n = {size}")
            figures.suptitle(f"{distr} distribution")
            plt.show()

    def plot_kde(self, distr_list, sizes, coefs):
        sns.set_style("whitegrid")

        for distr in distr_list:
            for size in sizes:
                figures, axs = plt.subplots(ncols=3, figsize=(14, 6))
                observations, pdf, cdf, x = self.get_probability_distr(distr, size)

                for i, coef in enumerate(coefs):
                    axs[i].plot(x, pdf, color="red", label="pdf")
                    sns.kdeplot(
                        data=observations,
                        bw_method="silverman",
                        bw_adjust=coef,
                        ax=axs[i],
                        alpha=.5,
                        label="kde",
                    )
                    axs[i].legend(loc="upper left")
                    axs[i].set(xlabel="x", ylabel="f(x)")
                    axs[i].set_xlim([self.a, self.b])
                    axs[i].set_title("h = " + str(coef))
                figures.suptitle(f"{distr} KDE n = {size}")
                print(f"{distr} KDE n = {size}")
                plt.show()
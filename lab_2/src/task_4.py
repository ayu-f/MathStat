import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt


class Task4:
    def mean(self, data):
        return np.mean(data)

    def dispersion(self, sample):
        return self.mean(list(map(lambda x: x * x, sample))) - (self.mean(sample)) ** 2

    def normal(self, size):
        return np.random.standard_normal(size=size)

    def run(self):
        alpha = 0.05

        m_all = {"norm": list(), "asymp": list()}
        s_all = {"norm": list(), "asymp": list()}
        x_all = list()
        for n in [20, 100]:
            x = self.normal(n)
            x_all.append(x)
            m = self.mean(x)
            s = np.sqrt(self.dispersion(x))
            m_n = [m - s * (stats.t.ppf(1 - alpha / 2, n - 1)) / np.sqrt(n - 1),
                  m + s * (stats.t.ppf(1 - alpha / 2, n - 1)) / np.sqrt(n - 1)]
            s_n = [s * np.sqrt(n) / np.sqrt(stats.chi2.ppf(1 - alpha / 2, n - 1)),
                  s * np.sqrt(n) / np.sqrt(stats.chi2.ppf(alpha / 2, n - 1))]

            m_all["norm"].append(m_n)
            s_all["norm"].append(s_n)

            m_as = [m - stats.norm.ppf(1 - alpha / 2) / np.sqrt(n), m + stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)]
            e = (sum(list(map(lambda el: (el - m) ** 4, x))) / n) / s ** 4 - 3
            s_as = [s / np.sqrt(1 + stats.norm.ppf(1 - alpha / 2) * np.sqrt((e + 2) / n)),
                    s / np.sqrt(1 - stats.norm.ppf(1 - alpha / 2) * np.sqrt((e + 2) / n))]

            m_all["asymp"].append(m_as)
            s_all["asymp"].append(s_as)

        for key in m_all.keys():
            print(f"m {key} : {m_all[key][0]},  {m_all[key][1]}")
            print(f"sigma {key}: {s_all[key][0]},  {s_all[key][1]}")







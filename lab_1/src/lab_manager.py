from pathlib import Path
from distribution import Distribution
from prettytable import PrettyTable

class LabManager:
    distributions = ["Normal", "Uniform", "Cauchy", "Poisson", "Laplace"]

    def run(self, num_lab: int):
        if num_lab == 1:
            self.lab1()
        elif num_lab == 2:
            self.lab2()
        elif num_lab == 3:
            self.lab3()
        elif num_lab == 4:
            self.lab4()

    def lab1(self):
        img_path = Path("images")
        for distr_name in self.distributions:
            for size in [10, 50, 1000]:
                distr = Distribution(distr_name, size)
                distr.plot_dens()

    def lab2(self):
        for distr_name in self.distributions:
            # print(f"****{distr_name}****")
            for size in [10, 100, 1000]:
                distr = Distribution(distr_name, size)
                E, D = distr.calc_characteristics()
                table = PrettyTable()
                table.field_names = [f"{distr_name} n = " + str(size), "Mean", "Median", "Zr", "Zq", "Ztr"]
                E.insert(0, 'E(z)')
                D.insert(0, 'D(z)')
                table.add_row(E)
                table.add_row(D)
                print(table)

    def lab3(self):
        img_path = Path("images/lab3")
        for distr_name in self.distributions:
            tips = []
            for size in [20, 100]:
                distr = Distribution(distr_name, size)
                distr.outlier_share()
                tips.append(distr.random_numbers)
            Distribution.draw_boxplot(tips, distr_name)

    def lab4(self):
        distr = Distribution()
        # distr.plot_ecdf(self.distributions, [20, 60, 100])
        distr.plot_kde(self.distributions, [20, 60, 100], [0.4, 0.8, 1.6])



import time
import simpy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle as pkl
import os, sys
# sys.path.append(os.path.abspath(r"C:\Users\Eshel\workspace\one.deep.moment"))
# sys.path.append(r'C:\Users\Eshel\workspace\butools2\Python')
# sys.path.append('../../one.deep.moment/')
# from butools.ph import *
import torch
import time
from utils import *

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln




class leadtime_no_negative:

    def __init__(self, dist_path, ind_demand_path, ind_lead_path, Lead_scale, S, s, max_S, num_samples=100000,
                 SIM_TIME=10000):

        self.SIM_TIME = SIM_TIME
        self.S = S
        self.s = s
        self.env = simpy.Environment()
        self.inventory = simpy.Container(self.env, init=self.S, capacity=self.S)
        self.last_event = 0
        self.inventory_times = defaultdict(float)
        self.fulfilled_demand = 0
        self.total_demand = 0
        self.demand_ind = 0
        self.lead_ind = 0
        self.monitor = {'last_time': 0, 'order_pending': False}

        # a_demand, T_demand = self.dist(ind_demand_path, dist_path)
        # a_lead, T_lead = self.dist(ind_lead_path, dist_path)
        # T_lead = T_lead * Lead_scale
        # print(compute_moments(a_lead, T_lead, T_lead.shape[0], 2))

        print('Start sampling')
        now = time.time()

        self.demands, self.demand_moms = self.generate_random_distribution(1) #.dist_from_file(ind_demand_path, dist_path, 1)
        self.lead_times, self.lead_moms = self.generate_random_distribution(Lead_scale)

        np.random.shuffle(self.demands)
        np.random.shuffle(self.lead_times)

        # self.lead_times = self.lead_times * Lead_scale

        print(self.lead_times.mean(), (self.lead_times ** 2).mean(), Lead_scale)

        end = time.time()
        tot_time = end - now

        print('end sampling, took: ', tot_time, ' seconds')

        self.num_cust_durations = {}
        for ind in range(0, max_S + 1):
            self.num_cust_durations[ind] = 0

    def generate_random_distribution(self, target_mean):
        """
        Randomly generate samples from a Lognormal, Weibull, or Gamma
        distribution having a specified mean and an SCV between 0.1 and 20.

        Parameters
        ----------
        target_mean : float
            Required mean of the generated distribution. Must be positive.

        Returns
        -------
        samples : np.ndarray
            Array containing 40,000,000 generated samples.

        moments : np.ndarray
            The first 10 analytical raw moments:
            [E[X], E[X^2], ..., E[X^10]].
        """

        if not np.isfinite(target_mean) or target_mean <= 0:
            raise ValueError("target_mean must be a positive finite number.")

        rng = np.random.default_rng()

        n_samples =  40_000_000
        moment_orders = np.arange(1, 11, dtype=np.float64)

        # Select the distribution with equal probability.
        distribution = rng.choice(["lognormal", "weibull", "gamma"])

        # Log-uniform sampling gives balanced coverage across the SCV range.
        min_scv = 0.1
        max_scv = 20.0

        target_scv = float(
            np.exp(
                rng.uniform(
                    np.log(min_scv),
                    np.log(max_scv)
                )
            )
        )

        if distribution == "lognormal":
            # For X ~ Lognormal(mu, sigma^2):
            #
            # E[X] = exp(mu + sigma^2 / 2)
            # SCV  = exp(sigma^2) - 1

            sigma_squared = np.log1p(target_scv)
            sigma = np.sqrt(sigma_squared)
            mu = np.log(target_mean) - 0.5 * sigma_squared

            samples = rng.lognormal(
                mean=mu,
                sigma=sigma,
                size=n_samples
            )

            # E[X^r] = exp(r*mu + r^2*sigma^2/2)
            log_moments = (
                    moment_orders * mu
                    + 0.5 * moment_orders ** 2 * sigma_squared
            )

            parameters = {
                "mu": mu,
                "sigma": sigma
            }

        elif distribution == "gamma":
            # For X ~ Gamma(shape, scale):
            #
            # E[X] = shape * scale
            # SCV  = 1 / shape

            shape = 1.0 / target_scv
            scale = target_mean / shape

            samples = rng.gamma(
                shape=shape,
                scale=scale,
                size=n_samples
            )

            # E[X^r] = scale^r * Gamma(shape+r) / Gamma(shape)
            log_moments = (
                    moment_orders * np.log(scale)
                    + gammaln(shape + moment_orders)
                    - gammaln(shape)
            )

            parameters = {
                "shape": shape,
                "scale": scale
            }

        else:
            # For X ~ Weibull(shape=k, scale=lambda):
            #
            # E[X] = lambda * Gamma(1 + 1/k)
            #
            # SCV = Gamma(1 + 2/k) /
            #       Gamma(1 + 1/k)^2 - 1

            target_log_ratio = np.log1p(target_scv)

            def weibull_scv_equation(shape):
                return (
                        gammaln(1.0 + 2.0 / shape)
                        - 2.0 * gammaln(1.0 + 1.0 / shape)
                        - target_log_ratio
                )

            # Weibull SCV decreases monotonically with its shape parameter.
            shape = brentq(
                weibull_scv_equation,
                0.05,
                1000.0
            )

            log_scale = (
                    np.log(target_mean)
                    - gammaln(1.0 + 1.0 / shape)
            )
            scale = np.exp(log_scale)

            samples = rng.weibull(
                a=shape,
                size=n_samples
            )
            samples *= scale

            # E[X^r] = scale^r * Gamma(1 + r/k)
            log_moments = (
                    moment_orders * log_scale
                    + gammaln(1.0 + moment_orders / shape)
            )

            parameters = {
                "shape": shape,
                "scale": scale
            }

        moments = np.exp(log_moments)

        # Useful for checking what was generated.
        print(f"Distribution: {distribution}")
        # print(f"Target mean: {target_mean:.6g}")
        print(f"Target SCV: {target_scv:.6g}")
        # print(f"Parameters: {parameters}")

        return samples, moments

    
    
    def dist_from_file(self, ind, dist_path, scale=1):

        path = os.path.join(dist_path, ind)
        files = os.listdir(path)
        ind_file = np.random.randint(0, len(files))

        dat = pkl.load(open(os.path.join(path, files[ind_file]), 'rb'))
        moms = dat[-2]
        if scale != 1:
            T = dat[1] / scale
            moms = compute_first_n_moments(dat[0], T, 10)
            # moms = np.array(torch.tensor(compute_moments(torch.tensor(dat[0]), torch.tensor(T), T.shape[0], 10))    )
        return (dat[-1], moms)

    def dist(self, ind, dist_path):

        files = os.listdir(dist_path)

        try:
            a, T, mm, scv, skew, kurt = pkl.load(open(os.path.join(dist_path, files[ind]), 'rb'))
        except:
            os.remove(os.path.join(dist_path, files[0]))
            print('Error loading file: ', files[0])

        if len(a.shape) == 2:
            a = a.reshape(-1)

        return (a, T)

    def run_simulation(self, ):

        self.env.process(self.demand_process())
        # self.env.process(self.monitor_inventory())

        self.env.run(until=self.SIM_TIME)

        # Stationary distribution
        total_time = sum(self.inventory_times.values())
        distribution = {k: v / total_time for k, v in self.inventory_times.items()}
        return distribution

    def demand_process(self, ):

        while True:
            yield self.env.timeout(self.demands[self.demand_ind % self.demands.shape[0]].item())
            self.demand_ind += 1
            self.total_demand += 1

            self.num_cust_durations[self.inventory.level] += self.env.now - self.last_event
            self.last_event = self.env.now

            if self.inventory.level > 0:
                yield self.inventory.get(1)
                self.fulfilled_demand += 1
            if self.inventory.level < self.s and not self.monitor['order_pending']:
                self.monitor['order_pending'] = True
                self.env.process(self.order_process())

            # if self.demand_ind % 100000 == 0:
            #     print('Current time: ', self.env.now, ' with inventory level: ', self.inventory.level)

    def order_process(self, ):

        yield self.env.timeout(self.lead_times[self.lead_ind % self.lead_times.shape[0]])
        self.lead_ind += 1
        self.num_cust_durations[self.inventory.level] += self.env.now - self.last_event
        self.last_event = self.env.now
        yield self.inventory.put(self.S - self.inventory.level)
        self.monitor['order_pending'] = False

    def monitor_inventory(self, ):
        while True:
            yield self.env.timeout(0.1)
            level = self.inventory.level
            now = self.env.now
            self.inventory_times[level] += now - self.monitor['last_time']
            self.monitor['last_time'] = now


def compute_moments(a, T, k, n):
    """ generate first n moments of FT (a, T)
    m_i = ((-1) ** i) i! a T^(-i) 1
    """
    T_in = torch.inverse(T)
    T_powers = torch.eye(k).double()
    signed_factorial = 1.
    one = torch.ones(k).double()

    moms = []
    for i in range(1, n + 1):
        signed_factorial *= -i
        T_powers = torch.matmul(T_powers, T_in)  # now T_powers is T^(-i)
        moms.append(signed_factorial * a @ T_powers @ one)

    return moms


from multiprocessing import Pool, cpu_count


def main():
    num_parallel_runs = 1  # or use cpu_count()

    with Pool(processes=num_parallel_runs) as pool:
        pool.map(run_single_simulation, range(num_parallel_runs))


def run_single_simulation(_):
    if True:

        max_S = 50
        SIM_TIME = 180000000
        num_samples = 60000000

        S = np.random.randint(1, max_S)
        s = np.random.randint(0, S)

        if sys.platform == 'linux':
            path_dists = '/home/elirans/scratch/ph_samples'
            dump_path = '/home/elirans/scratch/inv/lead_no_negative_multi_proc'
        else:
            path_dists = r'C:\Users\Eshel\workspace\data\sampled_dat'
            dump_path = r'C:\Users\Eshel\workspace\data\inv_data'

        scv_demand = np.random.choice(os.listdir(path_dists))
        scv_lead = np.random.choice(os.listdir(path_dists))

        Lead_scale = np.random.uniform(0.1, 10)

        inv_lead = leadtime_no_negative(path_dists, scv_demand, scv_lead, Lead_scale,
                                        S, s, max_S, SIM_TIME=SIM_TIME, num_samples=num_samples)

        distribution = inv_lead.run_simulation()
        data = inv_lead.num_cust_durations
        x = np.array(list(data.keys()))
        y = np.array(list(data.values())) / SIM_TIME

        fulfilrate = inv_lead.fulfilled_demand / inv_lead.total_demand
        mod_num = np.random.randint(1, 10000000)

        file_name = (str(mod_num) + '_' + str(s) + '_' + str(S) + '_' + scv_demand + '_' +
                     scv_lead + '_lead_scale_' + str(Lead_scale) + '_simtime_' + str(SIM_TIME) + 'cycle_order.pkl')
        full_path = os.path.join(dump_path, file_name)
        pkl.dump(((inv_lead.demand_moms, inv_lead.lead_moms), (fulfilrate, y, np.array(inv_lead.reordertimes).mean())),
                 open(full_path, 'wb'))

    # except:
    #     print('bad sampling')


if __name__ == "__main__":

    for ind in range(200):
        now = time.time()
        main()
        print(time.time() - now)

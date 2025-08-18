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


class leadtime_no_negative:

    def __init__(self, dist_path,  ind_demand_path, ind_lead_path, Lead_scale,S, s, max_S,num_samples = 100000, SIM_TIME = 10000):


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
        self.last_order = 0
        self.reordertimes = []

        # a_demand, T_demand = self.dist(ind_demand_path, dist_path)
        # a_lead, T_lead = self.dist(ind_lead_path, dist_path)
        # T_lead = T_lead * Lead_scale
        # print(compute_moments(a_lead, T_lead, T_lead.shape[0], 2))

        print('Start sampling')
        now = time.time()

        self.demands, self.demand_moms = self.dist_from_file(ind_demand_path, dist_path, 1)
        self.lead_times, self.lead_moms = self.dist_from_file(ind_lead_path, dist_path, Lead_scale)

        np.random.shuffle(self.demands)
        np.random.shuffle(self.lead_times)


        self.lead_times = self.lead_times * Lead_scale

        print(self.lead_times.mean(), (self.lead_times**2).mean(), Lead_scale)


        end = time.time()
        tot_time = end - now

        print('end sampling, took: ', tot_time, ' seconds')

        self.num_cust_durations = {}
        for ind in range(0, max_S + 1):
            self.num_cust_durations[ind] = 0

    def dist_from_file(self, ind, dist_path, scale=1):

        path = os.path.join(dist_path, ind)
        files = os.listdir(path)
        ind_file = np.random.randint(0, len(files))
        
        
        dat = pkl.load(open(os.path.join(path, files[ind_file]), 'rb'))
        moms = dat[-2] 
        if scale != 1:
            T = dat[1]/scale
            moms = compute_first_n_moments(dat[0],T,10)
            # moms = np.array(torch.tensor(compute_moments(torch.tensor(dat[0]), torch.tensor(T), T.shape[0], 10))    )
        return (dat[-1], moms)


    def dist(self,ind, dist_path):

        files = os.listdir(dist_path)

        try:
            a, T, mm, scv, skew, kurt = pkl.load(open(os.path.join(dist_path, files[ind]), 'rb'))
        except:
            os.remove(os.path.join(dist_path, files[0]))
            print('Error loading file: ', files[0])


        if len(a.shape) == 2:
            a = a.reshape(-1)

        return (a, T)

    def run_simulation(self,  ):



        self.env.process(self.demand_process())
        # self.env.process(self.monitor_inventory())

        self.env.run(until=self.SIM_TIME)

        # Stationary distribution
        total_time = sum(self.inventory_times.values())
        distribution = {k: v / total_time for k, v in self.inventory_times.items()}
        return distribution

    def demand_process(self,):

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

    def order_process(self,):

        yield self.env.timeout(self.lead_times[self.lead_ind % self.lead_times.shape[0]])
        self.lead_ind += 1
        self.num_cust_durations[self.inventory.level] += self.env.now - self.last_event
        self.last_event = self.env.now
        yield self.inventory.put(self.S - self.inventory.level)
        reorder_time = self.env.now - self.last_order
        self.reordertimes.append(reorder_time)
        self.last_order = self.env.now
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
    for i in range(1, n+1):
        signed_factorial *= -i
        T_powers = torch.matmul(T_powers, T_in)      # now T_powers is T^(-i)
        moms.append( signed_factorial * a @ T_powers @ one)

    return moms


def main():

    max_S = 50
    #
    SIM_TIME = 120000000
    num_samples = 50000000
    #

    for exmaple in range(0, 1000):
        try:
            ys = []
            path = r'C:\Users\Eshel\workspace\data\moment_anal\just_dists'

            S = 17 #np.random.randint(1, max_S)
            s = np.random.randint(1, S)
    
    
            if sys.platform == 'linux':
                path_dists = '/scratch/eliransc/ph_samples'
                dump_path = '/scratch/eliransc/inv/by_S'
            else:
                path_dists = r'C:\Users\Eshel\workspace\data\sampled_dat'
                dump_path = r'C:\Users\Eshel\workspace\data\inv_data'
    
            scv_demand =   np.random.choice(os.listdir(path_dists))
            scv_lead =  np.random.choice(os.listdir(path_dists))
    
            Lead_scale = np.random.uniform(0.1, 10)
    
            for jj in range(1):
    
    
                print('Running simulation for ind: ', jj, ' with SIM_TIME: ', SIM_TIME)
                inv_lead = leadtime_no_negative(path_dists, scv_demand, scv_lead, Lead_scale ,S , s, max_S, SIM_TIME=SIM_TIME, num_samples=num_samples)
    
                distribution = inv_lead.run_simulation()
    
                data = inv_lead.num_cust_durations

    
                # Separate keys and values
                x = np.array(list(data.keys()))
                y = np.array(list(data.values()))/SIM_TIME
                ys.append(y)
                fulfilrate = inv_lead.fulfilled_demand / inv_lead.total_demand
                print(np.array(inv_lead.reordertimes).mean())

                # print(fulfilrate)
            # Bar width and positions
            # width = 0.15
            # x1 = x - width / 2
            # x2 = x + width / 2
            # #
            # print('SAE: ', np.abs(ys[0]- ys[1]).sum())
            # #
            # print(100*(ys[0]- ys[1])/ ys[0])
            # # Plot stationary distribution
            # plt.figure(figsize=(20, 10))
            # # plt.bar(distribution.keys(), distribution.values())
            # plt.bar(x1, ys[0], color='red', alpha=0.9, width =width)
            # plt.bar(x2, ys[1], color='blue', alpha=0.9, width =width)
            # plt.xlabel("Inventory Level")
            # plt.ylabel("Stationary Probability")
            # plt.title(f"(S,s) Inventory Distribution (s={s}, S={S}), error in zero={100*(ys[0][3]- ys[1][3])/ ys[0][3]}")
            # plt.show()
            # 
            # Fulfillment rate
    
            mod_num = np.random.randint(1, 10000000)
    
            file_name = (str(mod_num)+ '_' + str(s) + '_' + str(S) + '_' + scv_demand + '_' + scv_lead + '_lead_scale_' + str(Lead_scale)
                        + '_simtime_'+  str(SIM_TIME) + 'cycle_order.pkl')
            full_path = os.path.join(dump_path, file_name)
            pkl.dump(((inv_lead.demand_moms, inv_lead.lead_moms),(fulfilrate, y, np.array(inv_lead.reordertimes).mean())), open(full_path, 'wb'))
            # except:
            #     print('Error in example: ')
        except:
            print('Error in example: ', exmaple, ' with s: ', s, ' and S: ', S, ' and scv_demand: ', scv_demand, ' and scv_lead: ', scv_lead)
        #     
            

if __name__ == "__main__":

    main()
import simpy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle as pkl
import os, sys
sys.path.append(os.path.abspath(r"C:\Users\Eshel\workspace\one.deep.moment"))
sys.path.append(r'C:\Users\Eshel\workspace\butools2\Python')
sys.path.append('../../one.deep.moment/')
from butools.ph import *
import torch

demand_ind = 0
lead_ind  = 0
dist_path = r'C:\Users\Eshel\workspace\data\moment_anal\just_dists'

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





def demand_process(env, inventory, monitor):
    global fulfilled_demand, total_demand, demand_ind, last_event
    while True:
        yield env.timeout(demands[demand_ind].item())
        demand_ind += 1
        total_demand += 1

        num_cust_durations[inventory.level] += env.now - last_event
        last_event = env.now

        if inventory.level > 0:
            yield inventory.get(1)
            fulfilled_demand += 1
        if inventory.level < s and not monitor['order_pending']:
            monitor['order_pending'] = True
            env.process(order_process(env, inventory, monitor))

def order_process(env, inventory, monitor):
    global lead_ind, last_event
    yield env.timeout(lead_times[lead_ind])
    lead_ind += 1
    num_cust_durations[inventory.level] += env.now - last_event
    last_event = env.now
    yield inventory.put(S - inventory.level)
    monitor['order_pending'] = False

def monitor_inventory(env, inventory, monitor):
    while True:
        yield env.timeout(0.1)
        level = inventory.level
        now = env.now
        inventory_times[level] += now - monitor['last_time']
        monitor['last_time'] = now

def run_simulation(last_event, num_cust_durations, S=10, s=4, SIM_TIME=10000):
    env = simpy.Environment()
    inventory = simpy.Container(env, init=S, capacity=S)
    monitor = {'last_time': 0, 'order_pending': False}

    env.process(demand_process(env, inventory, monitor))
    env.process(monitor_inventory(env, inventory, monitor))

    env.run(until=SIM_TIME)

    # Stationary distribution
    total_time = sum(inventory_times.values())
    distribution = {k: v / total_time for k, v in inventory_times.items()}
    return distribution

def dist(ind, dist_path):

    files = os.listdir(dist_path)


    try:
        a, T, mm, scv, skew, kurt = pkl.load(open(os.path.join(dist_path, files[ind]), 'rb'))

    except:
        os.remove(os.path.join(dist_path, files[0]))
        print('Error loading file: ', files[0])


    if len(a.shape) == 2:
        a = a.reshape(-1)

    return (a, T)

def main():

    a_demand, T_demand = dist(0, dist_path)
    a_lead, T_lead = dist(1, dist_path)


    demands = SamplesFromPH(ml.matrix(a_demand), np.array(T_demand), 90000)
    lead_times = SamplesFromPH(ml.matrix(a_lead), np.array(T_lead), 90000)
    # Scale lead times for simulation
    print(demands.mean(), lead_times.mean())
    # Parameters
    s = 4  # reorder point
    S = 10  # order-up-to level
    lambda_demand = 1.0  # demand rate

    SIM_TIME = 10000

    # Stats trackers
    inventory_times = defaultdict(float)
    fulfilled_demand = 0
    total_demand = 0

    num_cust_durations = {}
    for ind in range(0, S + 1):
        num_cust_durations[ind] = 0

    last_event = 0

    distribution = run_simulation(last_event, num_cust_durations)
    data = num_cust_durations

    # Separate keys and values
    x = list(data.keys())
    y = np.array(list(data.values()))/SIM_TIME


    # Plot stationary distribution
    plt.figure(figsize=(20, 10))
    # plt.bar(distribution.keys(), distribution.values())
    plt.bar(x, y, color='red', alpha=0.4)
    plt.xlabel("Inventory Level")
    plt.ylabel("Stationary Probability")
    plt.title(f"(S,s) Inventory Distribution (s={s}, S={S}, Lead={LEAD_TIME})")
    plt.show()

    # Fulfillment rate
    print(f"Fulfilled Demand Rate: {fulfilled_demand / total_demand:.4f}")


if __name__ == "__main__":

    main()
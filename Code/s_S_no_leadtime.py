from copyreg import pickle

import simpy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import sys
sys.path.append(os.path.abspath(r"C:\Users\Eshel\workspace\one.deep.moment"))
sys.path.append(r'C:\Users\Eshel\workspace\butools2\Python')
sys.path.append('../../one.deep.moment/')
import pickle as pkl
from butools.ph import *


demand_ind = 0
lead_ind  = 0
dist_path = r'C:\Users\Eshel\workspace\data\moment_anal\just_dists'


files = os.listdir(dist_path)
file_rand = np.random.choice(files).item()
orig_dist_type = file_rand.split('_')[3]
print('file dist: ', file_rand)
try:
    a_demand, T_demand, mm, scv, skew, kurt = pkl.load(open(os.path.join(dist_path, file_rand), 'rb'))

except:
    os.remove(os.path.join(dist_path, file_rand))
    print('Error loading file: ', file_rand)
T_demand = T_demand*10



files = os.listdir(dist_path)
file_rand = np.random.choice(files).item()
orig_dist_type = file_rand.split('_')[3]
print('file dist: ', file_rand)
try:
    a_lead, T_lead, mm, scv, skew, kurt = pkl.load(open(os.path.join(dist_path, file_rand), 'rb'))

except:
    os.remove(os.path.join(dist_path, file_rand))
    print('Error loading file: ', file_rand)
if len(a_demand.shape) == 2:
    a_demand.reshape(-1)
if len(a_lead.shape) == 2:
    a_lead.reshape(-1)

print(a_demand.shape, T_demand.shape, a_lead.shape, T_lead.shape)
demands = SamplesFromPH(ml.matrix(a_demand), np.array(T_demand), 90000)
lead_times = SamplesFromPH(ml.matrix(a_lead), np.array(T_lead), 90000)
# Parameters
s = 5  # reorder point
S = 10  # order-up-to level
SIM_TIME = 10000  # total simulation time

# Tracking time in each inventory level
inventory_times = defaultdict(float)

num_cust_durations = {}
for ind in range(s, S + 1):
    num_cust_durations[ind] = 0

last_event = 0

def demand_process(env, inventory, monitor, num_cust_durations, last_event):
    global demand_ind, lead_ind
    while True:
        yield env.timeout(demands[demand_ind].item())
        demand_ind += 1
        num_cust_durations[inventory.level] += env.now - last_event
        last_event = env.now
        if inventory.level > 0:
            inventory.get(1)
        if inventory.level <= s:
        # Place an order to replenish inventory to level S
            yield env.timeout(lead_times[lead_ind])
            lead_ind += 1
            num_cust_durations[inventory.level] += env.now - last_event
            last_event = env.now
            yield inventory.put(S - inventory.level)
        monitor['last_time'] = env.now


def monitor_inventory(env, inventory, monitor):
    while True:
        yield env.timeout(0.1)
        level = inventory.level
        now = env.now
        inventory_times[level] += now - monitor['last_time']
        monitor['last_time'] = now


def run_simulation():
    env = simpy.Environment()
    inventory = simpy.Container(env, init=S, capacity=S)
    monitor = {'last_time': 0}

    env.process(demand_process(env, inventory, monitor, num_cust_durations, last_event))

    env.run(until=SIM_TIME)

    # Normalize to get stationary distribution
    total_time = sum(inventory_times.values())
    distribution = {k: v / total_time for k, v in inventory_times.items()}
    return distribution


distribution = run_simulation()
print(num_cust_durations)
data = num_cust_durations

# Separate keys and values
x = list(data.keys())
y = np.array(list(data.values()))/SIM_TIME
plt.figure()
# Create bar chart

# Plot
plt.bar(distribution.keys(), distribution.values())
plt.bar(x, y, color='red', alpha=0.7)
plt.xlabel("Inventory Level")
plt.ylabel("Stationary Probability")
plt.title(f"Stationary Distribution of Inventory Level (s={s}, S={S})")
plt.show()


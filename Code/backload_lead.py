import simpy, random, statistics

class SingleOrderSSInventory:
    """
    (s, S) inventory with backlogging and *at most one outstanding order*.
    - Demand arrives one unit at a time (inter-demand ~ Exp(lam)).
    - Lead time for a placed order ~ Exp(mu).
    - We order only when *no order is pending* and inventory position <= s,
      and then raise the *position* to S (classic order-up-to on position).
    - Inventory position: pos = on_hand - backorders + on_order
    """

    def __init__(self, env, s, S, demand_rate, lead_rate, init_on_hand=None, seed=42):
        assert s < S, "(s,S) requires s < S"
        self.env = env
        self.s, self.S = int(s), int(S)
        self.lam, self.mu = float(demand_rate), float(lead_rate)

        self.rng = random.Random(seed)
        init = self.S if init_on_hand is None else min(int(init_on_hand), self.S)

        # On-hand stock (continuous container, we use integer amounts)
        # Keep capacity = S (physical storage). With backlogs, large puts will
        # be consumed immediately by waiting gets; SimPy resolves this at the
        # same simulation time.
        self.inv = simpy.Container(env, capacity=self.S, init=init)

        # State
        self.backorders = 0      # integer count of outstanding unit demands
        self.on_order   = 0      # integer units currently on order (0 or >0)
        self.pending    = False  # True iff there is an outstanding order

        # Wakeup event to react to state changes
        self._state_evt = env.event()

        # Metrics
        self.n_demands = 0
        self.n_immediate = 0
        self.wait_times = []

        # Time-series samples (optional)
        self.on_hand_samples = []
        self.backlog_samples = []

    # ---------- utilities ----------
    def _exp(self, rate):  # Exp(rate) sampler
        return self.rng.expovariate(rate)

    def pos(self):
        return int(self.inv.level) - self.backorders + self.on_order

    def _notify(self):
        if not self._state_evt.triggered:
            self._state_evt.succeed()
        self._state_evt = self.env.event()

    # ---------- processes ----------
    def demand_process(self):
        """Unit demands; with backlogs allowed we always 'get(1)' (may wait)."""
        while True:
            yield self.env.timeout(self._exp(self.lam))
            self.n_demands += 1
            t0 = self.env.now

            # A demand *immediately* reduces position via backlog++
            self.backorders += 1
            self._notify()  # position changed

            # If no on-hand, this get blocks until supply arrives
            yield self.inv.get(1)

            # Demand is now filled
            wait = self.env.now - t0
            self.wait_times.append(wait)
            if wait == 0:
                self.n_immediate += 1

            self.backorders -= 1
            self._notify()  # position increased by 1 (backlog--)

    def _arrival_process(self, q):
        """Order arrival after Exp(mu); deliver q units."""
        yield self.env.timeout(self._exp(self.mu))
        # On arrival: remove from 'on_order' and add to on-hand
        self.on_order -= q
        # Deposit q to on-hand; waiting 'get(1)' calls will immediately draw from it
        yield self.inv.put(q)
        self.pending = False
        self._notify()  # level and position changed

    def maybe_order(self):
        """Place an order iff no order pending and position <= s (single outstanding)."""
        if self.pending:
            return
        if self.pos() <= self.s:
            q = self.S - self.pos()   # raise *position* to S
            if q > 0:
                self.on_order += q
                self.pending = True
                self._notify()  # position jumps by +q immediately
                self.env.process(self._arrival_process(q))

    def policy_process(self):
        """Policy loop: react only on state changes; enforce single-outstanding rule."""
        while True:
            # Check once; if we ordered, pos becomes S (> s); otherwise sleep
            self.maybe_order()
            ev = self._state_evt
            yield ev

    def monitor(self, dt=1.0):
        while True:
            self.on_hand_samples.append((self.env.now, float(self.inv.level)))
            self.backlog_samples.append((self.env.now, int(self.backorders)))
            yield self.env.timeout(dt)

def run_one(s=5, S=20, demand_rate=0.8, lead_rate=0.5, T=10_000, seed=7):
    env = simpy.Environment()
    sys = SingleOrderSSInventory(env, s, S, demand_rate, lead_rate, seed=seed)

    env.process(sys.demand_process())
    env.process(sys.policy_process())
    env.process(sys.monitor(dt=1.0))

    env.run(until=T)

    # KPIs
    immediate_service = sys.n_immediate / sys.n_demands if sys.n_demands else float('nan')
    avg_wait = statistics.mean(sys.wait_times) if sys.wait_times else 0.0

    # Rough time-averages from uniform sampling
    def time_avg(series):
        if len(series) < 2: return float('nan')
        area = 0.0
        for (t0, x0), (t1, _x1) in zip(series[:-1], series[1:]):
            area += x0 * (t1 - t0)
        horizon = series[-1][0] - series[0][0]
        return area / horizon if horizon > 0 else float('nan')

    return {
        "n_demands": sys.n_demands,
        "immediate_service": immediate_service,  # P{wait=0}
        "avg_wait_time": avg_wait,
        "avg_on_hand": time_avg(sys.on_hand_samples),
        "avg_backlog": time_avg(sys.backlog_samples),
        "final_position": sys.pos(),            # should be > s unless pending/backlog
        "pending": sys.pending,
        "on_order": sys.on_order,
    }

# Example run:
if __name__ == "__main__":
    stats = run_one(s=5, S=20, demand_rate=0.8, lead_rate=0.5, T=5000, seed=123)
    for k, v in stats.items():
        print(f"{k}: {v}")

"""
Energy-efficient multi-hop routing in a wireless sensor network (WSN).

Requirements:
    Python 3.9+
    SciPy (pip install scipy)

Model:
    Maximize network lifetime T.
    f[i,j] = total bits sent over directed link i -> j during T (bits).
    r[i]   = generated traffic rate at sensor i (bits/s).
    B[i]   = battery energy at sensor i (J).
    e_tx[i,j] = transmission energy per bit (J/bit).
    e_rx       = reception energy per bit (J/bit).

The graph is fixed. Sensor-sensor radio links are bidirectional; sink-outgoing
links are omitted. Link capacities are not included in this first version.
"""

from __future__ import annotations

import argparse
import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.optimize import linprog

Node = int
Arc = Tuple[Node, Node]

# Shared first-order radio model parameters (used by both solver and dashboard).
E_ELEC_J_PER_BIT = 50e-9
EPSILON_AMP_J_PER_BIT_M2 = 100e-12
PATH_LOSS_EXPONENT = 2.0

def radio_tx_energy(distance_m: float) -> float:
    """Transmission energy per bit for the shared first-order radio model."""
    return E_ELEC_J_PER_BIT + EPSILON_AMP_J_PER_BIT_M2 * (float(distance_m) ** PATH_LOSS_EXPONENT)


@dataclass
class Network:
    positions: np.ndarray
    sink: int
    arcs: List[Arc]
    tx_energy: Dict[Arc, float]
    rx_energy: float
    generation_rate: Dict[Node, float]
    battery: Dict[Node, float]
    source: Optional[Node] = None


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def make_network(
    n_sensors: int = 10,
    seed: int = 7,
    area_size_m: float = 100.0,
    extra_link_probability: float = 0.28,
) -> Network:
    """Create a connected WSN with distributed sensor sources, one designated SOURCE node, and one SINK.

    The n_sensors ordinary sensors each generate traffic. An additional,
    battery-powered SOURCE node also generates traffic and participates in
    forwarding. The SINK is the sole destination.
    """
    if n_sensors < 2:
        raise ValueError("Use at least 2 sensor nodes.")
    rng = np.random.default_rng(seed)
    source = n_sensors
    sink = n_sensors + 1
    n_total = n_sensors + 2
    positions = rng.uniform(0.0, area_size_m, size=(n_total, 2))

    # Build a minimum spanning tree over all nodes so every sensor can reach
    # the sink. Then add extra links to create routing choices.
    candidates = []
    for i in range(n_total):
        for j in range(i + 1, n_total):
            d = euclidean(positions[i], positions[j])
            candidates.append((d, i, j))
    candidates.sort()

    parent = list(range(n_total))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    undirected_edges = set()
    for d, i, j in candidates:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            undirected_edges.add((i, j))

    # Add extra links probabilistically, excluding already selected MST links.
    for d, i, j in candidates:
        if (i, j) not in undirected_edges and rng.random() < extra_link_probability:
            undirected_edges.add((i, j))

    arcs: List[Arc] = []
    for i, j in sorted(undirected_edges):
        # Sink does not transmit in this data-collection model.
        if i != sink:
            arcs.append((i, j))
        if j != sink:
            arcs.append((j, i))

    # First-order radio energy model: E_tx = E_elec + epsilon * distance^2.
    # Values are illustrative; replace with values justified by your sources.
    tx_energy = {}
    for i, j in arcs:
        d = euclidean(positions[i], positions[j])
        tx_energy[(i, j)] = radio_tx_energy(d)

    # Every ordinary sensor generates its own measurements. The designated
    # SOURCE is an additional battery-powered node that also generates data.
    generation_rate = {i: float(rng.uniform(500.0, 1500.0)) for i in range(n_total - 1)}
    battery = {i: float(rng.uniform(0.4, 0.8)) for i in range(n_total - 1)}
    rx_energy = E_ELEC_J_PER_BIT

    return Network(
        positions=positions,
        sink=sink,
        arcs=arcs,
        tx_energy=tx_energy,
        rx_energy=rx_energy,
        generation_rate=generation_rate,
        battery=battery,
        source=source,
    )


def solve_lifetime_lp(net: Network):
    """Solve the lifetime-maximizing LP with SciPy/HiGHS."""
    arcs = net.arcs
    sensors = sorted(net.battery)
    # Decision vector: [f_arc_0, ..., f_arc_m-1, T].
    t_idx = len(arcs)
    n_vars = len(arcs) + 1
    arc_idx = {arc: k for k, arc in enumerate(arcs)}

    # Battery constraints: transmit energy + receive energy <= battery.
    A_ub = np.zeros((len(sensors), n_vars))
    b_ub = np.array([net.battery[i] for i in sensors], dtype=float)
    sensor_row = {node: row for row, node in enumerate(sensors)}
    for k, (i, j) in enumerate(arcs):
        if i in sensor_row:
            A_ub[sensor_row[i], k] += net.tx_energy[(i, j)]
        if j in sensor_row:
            A_ub[sensor_row[j], k] += net.rx_energy

    # Flow conservation for each sensor:
    # outgoing total bits - incoming total bits = generation_rate * T.
    A_eq = np.zeros((len(sensors), n_vars))
    b_eq = np.zeros(len(sensors))
    for row, i in enumerate(sensors):
        for k, (u, v) in enumerate(arcs):
            if u == i:
                A_eq[row, k] += 1.0
            if v == i:
                A_eq[row, k] -= 1.0
        A_eq[row, t_idx] = -net.generation_rate[i]

    objective = np.zeros(n_vars)
    objective[t_idx] = -1.0  # linprog minimizes, so minimize -T.
    bounds = [(0, None)] * n_vars

    result = linprog(
        objective,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"LP solver failed: {result.message}")

    flows = {arc: float(result.x[k]) for k, arc in enumerate(arcs)}
    lifetime = float(result.x[t_idx])
    energy_used = {i: 0.0 for i in sensors}
    for (i, j), flow in flows.items():
        energy_used[i] += net.tx_energy[(i, j)] * flow
        if j in energy_used:
            energy_used[j] += net.rx_energy * flow
    return {
        "lifetime": lifetime,
        "flows": flows,
        "energy_used": energy_used,
        "solver_result": result,
    }


def dijkstra_path_to_sink(net: Network, source: int) -> List[Node]:
    """Shortest path by sum of transmission energy per bit."""
    outgoing: Dict[int, List[Tuple[int, float]]] = {}
    for i, j in net.arcs:
        outgoing.setdefault(i, []).append((j, net.tx_energy[(i, j)]))

    dist = {node: math.inf for node in range(len(net.positions))}
    previous: Dict[int, int] = {}
    dist[source] = 0.0
    queue = [(0.0, source)]
    while queue:
        d, u = heapq.heappop(queue)
        if d > dist[u]:
            continue
        if u == net.sink:
            break
        for v, weight in outgoing.get(u, []):
            nd = d + weight
            if nd < dist[v]:
                dist[v] = nd
                previous[v] = u
                heapq.heappush(queue, (nd, v))

    if not math.isfinite(dist[net.sink]):
        raise RuntimeError(f"Sensor {source} cannot reach the sink.")
    path = [net.sink]
    cur = net.sink
    while cur != source:
        cur = previous[cur]
        path.append(cur)
    return list(reversed(path))


def shortest_path_baseline(net: Network):
    """Route each sensor's own traffic along its shortest-energy path."""
    rates = {arc: 0.0 for arc in net.arcs}
    for source, rate in net.generation_rate.items():
        path = dijkstra_path_to_sink(net, source)
        for u, v in zip(path[:-1], path[1:]):
            rates[(u, v)] += rate

    power = {i: 0.0 for i in net.battery}
    for (i, j), rate in rates.items():
        power[i] += net.tx_energy[(i, j)] * rate
        if j in power:
            power[j] += net.rx_energy * rate

    node_lifetimes = {
        i: (net.battery[i] / p if p > 0 else math.inf)
        for i, p in power.items()
    }
    return {
        "lifetime": min(node_lifetimes.values()),
        "rates": rates,
        "power": power,
        "node_lifetimes": node_lifetimes,
    }


def run_experiment(n_sensors: int, seed: int):
    net = make_network(n_sensors=n_sensors, seed=seed)
    lp = solve_lifetime_lp(net)
    baseline = shortest_path_baseline(net)
    improvement = 100.0 * (lp["lifetime"] - baseline["lifetime"]) / baseline["lifetime"]
    used = lp["energy_used"]
    tight = [
        i for i in sorted(net.battery)
        if abs(used[i] - net.battery[i]) <= max(1e-8, 1e-5 * net.battery[i])
    ]

    print(f"\n=== {n_sensors} sensors | seed={seed} ===")
    print(f"Directed links: {len(net.arcs)}")
    print(f"LP lifetime:             {lp['lifetime']:.6f} s")
    print(f"Shortest-path lifetime:  {baseline['lifetime']:.6f} s")
    print(f"Improvement:             {improvement:.2f}%")
    print(f"Battery constraints tight at LP optimum: {tight}")
    print(f"Solver status: {lp['solver_result'].message}")
    print("Node | generation (bit/s) | battery (J) | LP energy used (J)")
    for i in sorted(net.battery):
        print(
            f"{i:>4} | {net.generation_rate[i]:>18.2f} | "
            f"{net.battery[i]:>11.4f} | {used[i]:>18.6f}"
        )
    print("Positive LP flows (total bits over the optimized lifetime):")
    for (i, j), f in lp["flows"].items():
        if f > 1e-7:
            print(f"  {i} -> {j}: {f:.6f} bits")
    return net, lp, baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, nargs="+", default=[10, 20],
                        help="number of sensors (sink is added separately)")
    parser.add_argument("--seed", type=int, default=7,
                        help="random seed for reproducibility")
    args = parser.parse_args()
    for n in args.nodes:
        run_experiment(n, args.seed)


if __name__ == "__main__":
    main()

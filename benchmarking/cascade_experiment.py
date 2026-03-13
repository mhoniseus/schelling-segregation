"""
Cascade size distribution measurement for the Schelling model.

Measures how single-agent perturbations propagate through an equilibrated grid.
Compares measured mean cascade size to the theoretical prediction 1/(1-R(T)).

Usage: python benchmarking/cascade_experiment.py
Output: outputs/data/cascade_results.npz
"""

import numpy as np
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.schelling import SchellingModel, EMPTY, TYPE_A, TYPE_B, _MOORE_OFFSETS, _cell_satisfaction

# ── Parameters ──────────────────────────────────────────────────────
L = 80
RHO = 0.9
F_A = 0.5
N_PERTURBATIONS = 2000
T_VALUES = [0.20, 0.25, 0.275, 0.30, 0.325, 0.375, 0.40, 0.50]
MAX_EQUIL_STEPS = 500
SEED = 42


def get_neighbors(r, c, size):
    return [((r + dr) % size, (c + dc) % size) for dr, dc in _MOORE_OFFSETS]


def measure_cascade(grid, size, T, rng):
    """
    Perturb an equilibrated grid and measure cascade size via BFS.

    1. Pick a random satisfied agent
    2. Move it to a random empty cell
    3. BFS: track neighbors that become newly unsatisfied, relocate them, repeat
    4. Return total agents displaced (including initial)
    """
    # Find satisfied agents
    satisfied = []
    for r in range(size):
        for c in range(size):
            if grid[r, c] != EMPTY and _cell_satisfaction(grid, r, c) >= T:
                satisfied.append((r, c))

    if not satisfied:
        return 0

    idx = rng.integers(len(satisfied))
    sr, sc = satisfied[idx]
    agent_type = grid[sr, sc]

    # Record pre-perturbation satisfaction of neighbors
    old_neighbors = get_neighbors(sr, sc, size)
    was_satisfied = {}
    for nr, nc in old_neighbors:
        if grid[nr, nc] != EMPTY:
            was_satisfied[(nr, nc)] = _cell_satisfaction(grid, nr, nc) >= T

    # Move agent to random empty cell
    empties = list(zip(*np.where(grid == EMPTY)))
    if not empties:
        return 0
    eidx = rng.integers(len(empties))
    er, ec = empties[eidx]
    grid[sr, sc] = EMPTY
    grid[er, ec] = agent_type

    cascade_size = 1
    from collections import deque
    queue = deque()
    visited = set()

    # Check old neighbors that became newly unsatisfied
    for nr, nc in old_neighbors:
        if grid[nr, nc] != EMPTY and was_satisfied.get((nr, nc), True):
            if _cell_satisfaction(grid, nr, nc) < T:
                queue.append((nr, nc))

    # Check new location neighbors that might have become unsatisfied
    for nr, nc in get_neighbors(er, ec, size):
        if grid[nr, nc] != EMPTY and (nr, nc) not in was_satisfied:
            if _cell_satisfaction(grid, nr, nc) < T:
                queue.append((nr, nc))

    while queue:
        cr, cc = queue.popleft()
        if (cr, cc) in visited or grid[cr, cc] == EMPTY:
            continue
        if _cell_satisfaction(grid, cr, cc) >= T:
            continue
        visited.add((cr, cc))
        cascade_size += 1

        # Record neighbor satisfaction before move
        cur_neighbors = get_neighbors(cr, cc, size)
        nbr_sat = {}
        for nr, nc in cur_neighbors:
            if grid[nr, nc] != EMPTY and (nr, nc) not in visited:
                nbr_sat[(nr, nc)] = _cell_satisfaction(grid, nr, nc) >= T

        # Relocate this agent
        agent = grid[cr, cc]
        empties = list(zip(*np.where(grid == EMPTY)))
        if not empties:
            continue
        eidx = rng.integers(len(empties))
        nr2, nc2 = empties[eidx]
        grid[cr, cc] = EMPTY
        grid[nr2, nc2] = agent

        # Check old neighbors
        for nr, nc in cur_neighbors:
            if grid[nr, nc] != EMPTY and (nr, nc) not in visited:
                if nbr_sat.get((nr, nc), True) and _cell_satisfaction(grid, nr, nc) < T:
                    queue.append((nr, nc))

        # Check new neighbors
        for nr, nc in get_neighbors(nr2, nc2, size):
            if grid[nr, nc] != EMPTY and (nr, nc) not in visited:
                if _cell_satisfaction(grid, nr, nc) < T:
                    queue.append((nr, nc))

    return cascade_size


def R_theory(T, rho=0.9, fA=0.5):
    """Compute theoretical branching ratio R(T)."""
    from math import comb
    total = 0.0
    for kp in range(1, 8):
        binom_kp = comb(7, kp) * rho**kp * (1-rho)**(7-kp)
        inner = 0.0
        for jp in range(0, kp+1):
            sat_before = (jp + 1) / (kp + 1)
            sat_after = jp / kp
            if sat_before >= T and sat_after < T:
                inner += comb(kp, jp) * fA**jp * (1-fA)**(kp-jp)
        total += binom_kp * inner
    return 8 * rho * fA * total


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs('outputs/data', exist_ok=True)

    all_cascade_sizes = {}
    all_means = {}
    all_R = {}

    for T in T_VALUES:
        t0 = time.time()
        print(f"T={T:.3f}: ", end="", flush=True)
        sizes = []

        for trial in range(N_PERTURBATIONS):
            model = SchellingModel(size=L, density=RHO, fraction_a=F_A,
                                   tolerance=T, seed=rng.integers(2**31))
            model.run(max_steps=MAX_EQUIL_STEPS)
            grid = model.grid.copy()
            cs = measure_cascade(grid, L, T, rng)
            sizes.append(cs)

        sizes = np.array(sizes)
        R = R_theory(T)
        pred = 1.0 / (1.0 - R) if R < 1 else float('inf')
        mean_cs = sizes.mean()

        all_cascade_sizes[f"T_{T:.3f}"] = sizes
        all_means[f"T_{T:.3f}"] = mean_cs
        all_R[f"T_{T:.3f}"] = R

        elapsed = time.time() - t0
        print(f"mean={mean_cs:.2f}, R={R:.3f}, pred={pred:.2f}, "
              f"ratio={mean_cs/pred:.2f}, max={sizes.max()}, "
              f"({elapsed:.0f}s)")

    # Save
    save_dict = {}
    save_dict['T_values'] = np.array(T_VALUES)
    for T in T_VALUES:
        key = f"T_{T:.3f}"
        save_dict[f"sizes_{key}"] = all_cascade_sizes[key]
        save_dict[f"mean_{key}"] = all_means[key]
        save_dict[f"R_{key}"] = all_R[key]

    np.savez('outputs/data/cascade_results.npz', **save_dict)

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"{'T':>8} {'Mean CS':>10} {'R(T)':>8} {'1/(1-R)':>10} {'Ratio':>8}")
    for T in T_VALUES:
        key = f"T_{T:.3f}"
        R = all_R[key]
        pred = 1.0 / (1.0 - R) if R < 1 else float('inf')
        mean = all_means[key]
        ratio = mean / pred if pred < float('inf') else 0
        print(f"{T:8.3f} {mean:10.3f} {R:8.3f} {pred:10.3f} {ratio:8.3f}")


if __name__ == '__main__':
    main()

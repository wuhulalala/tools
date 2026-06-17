#!/usr/bin/env python3
"""Small Twill-style scheduler for the dense FlashMLA TLE split.

The production kernel is still handwritten TLE, but this script keeps the
task-assignment decision reproducible: it models the loop body as tile-level
operations and solves for a resource-feasible warp-group assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from z3 import And, Bool, If, Implies, Not, Optimize, Sum, is_true, sat


@dataclass(frozen=True)
class Op:
    name: str
    resource: str
    cycles: int
    allowed: tuple[str, ...]


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    delay: int
    spill: int = 0


BASE_OPS = {
    "load_k0": Op("load_k0", "MEM", 1, ("producer",)),
    "load_k1": Op("load_k1", "MEM", 1, ("producer",)),
    "qk0": Op("qk0", "TC", 2, ("consumer0",)),
    "qk1": Op("qk1", "TC", 2, ("consumer1",)),
    "softmax0": Op("softmax0", "SFU", 1, ("consumer0",)),
    "softmax1": Op("softmax1", "SFU", 1, ("consumer1",)),
    "pv0_left": Op("pv0_left", "TC", 1, ("consumer0",)),
    "pv1_left": Op("pv1_left", "TC", 1, ("consumer0",)),
    "pv1_right": Op("pv1_right", "TC", 1, ("consumer1",)),
    "pv0_right": Op("pv0_right", "TC", 1, ("consumer1",)),
}

EDGES = [
    Edge("load_k0", "qk0", 1),
    Edge("load_k1", "qk1", 1),
    Edge("qk0", "softmax0", 2),
    Edge("qk1", "softmax1", 2),
    Edge("softmax0", "softmax1", 1, spill=1),  # max0 -> max_next
    Edge("softmax1", "pv0_left", 1, spill=1),  # max_next -> scale prob0
    Edge("softmax0", "pv0_left", 1),
    Edge("softmax1", "pv1_right", 1),
    Edge("softmax1", "pv1_left", 1, spill=1),  # prob1 handoff
    Edge("softmax0", "pv0_right", 1, spill=1),  # prob0 handoff
    Edge("load_k1", "pv1_left", 1, spill=1),  # remote K left
    Edge("load_k0", "pv0_right", 1, spill=1),  # remote K right
]

# Independent softmax: no max exchange between consumers, only prob+max sent together
EDGES_INDEP_SOFTMAX = [
    Edge("load_k0", "qk0", 1),
    Edge("load_k1", "qk1", 1),
    Edge("qk0", "softmax0", 2),
    Edge("qk1", "softmax1", 2),
    # No softmax0->softmax1 dependency!
    Edge("softmax0", "pv0_left", 1),
    Edge("softmax1", "pv1_right", 1),
    Edge("softmax1", "pv1_left", 1, spill=1),  # prob1+max1 handoff (parallel with pv1_right)
    Edge("softmax0", "pv0_right", 1, spill=1),  # prob0+max0 handoff (parallel with pv0_left)
    Edge("load_k1", "pv1_left", 1, spill=1),  # remote K left
    Edge("load_k0", "pv0_right", 1, spill=1),  # remote K right
]

# Split-KV: each consumer handles full dim of its own block, no prob exchange
SPLITKV_OPS = {
    "load_k0": Op("load_k0", "MEM", 2, ("producer",)),  # full dim load takes 2 cycles
    "load_k1": Op("load_k1", "MEM", 2, ("producer",)),
    "qk0": Op("qk0", "TC", 2, ("consumer0",)),
    "qk1": Op("qk1", "TC", 2, ("consumer1",)),
    "softmax0": Op("softmax0", "SFU", 1, ("consumer0",)),
    "softmax1": Op("softmax1", "SFU", 1, ("consumer1",)),
    "pv0": Op("pv0", "TC", 2, ("consumer0",)),  # full dim PV takes 2 cycles
    "pv1": Op("pv1", "TC", 2, ("consumer1",)),
}

EDGES_SPLITKV = [
    Edge("load_k0", "qk0", 1),
    Edge("load_k1", "qk1", 1),
    Edge("qk0", "softmax0", 2),
    Edge("qk1", "softmax1", 2),
    Edge("softmax0", "pv0", 1),
    Edge("softmax1", "pv1", 1),
    # No cross-consumer edges at all during the loop!
]

CAPACITY = {"MEM": 1, "TC": 1, "SFU": 1}


def variant_ops(variant: str) -> tuple[dict[str, Op], tuple[str, ...], list[Edge]]:
    if variant == "3wg_prefetch":
        return BASE_OPS, ("producer", "consumer0", "consumer1"), EDGES
    if variant == "3wg_indep_softmax":
        return BASE_OPS, ("producer", "consumer0", "consumer1"), EDGES_INDEP_SOFTMAX
    if variant == "3wg_splitkv":
        return SPLITKV_OPS, ("producer", "consumer0", "consumer1"), EDGES_SPLITKV
    if variant == "2wg_no_producer":
        ops = dict(BASE_OPS)
        ops["load_k0"] = replace(ops["load_k0"], allowed=("consumer0",))
        ops["load_k1"] = replace(ops["load_k1"], allowed=("consumer1",))
        return ops, ("consumer0", "consumer1"), EDGES
    if variant == "1wg_serial":
        ops = {
            name: replace(op, allowed=("consumer0",))
            for name, op in BASE_OPS.items()
        }
        return ops, ("consumer0",), EDGES
    raise ValueError(f"unknown variant {variant}")


def edges_for(ops: dict[str, Op], base_edges: list[Edge]) -> list[Edge]:
    fixed = []
    for edge in base_edges:
        if edge.src not in ops or edge.dst not in ops:
            continue
        src = ops[edge.src].allowed[0]
        dst = ops[edge.dst].allowed[0]
        fixed.append(replace(edge, spill=edge.spill if src != dst else 0))
    return fixed


def solve_once(variant: str, ii: int, horizon: int) -> tuple[int, int, list[tuple[str, int, str]]] | None:
    ops, warps, base_edges = variant_ops(variant)
    edges = edges_for(ops, base_edges)
    opt = Optimize()
    opt.set(timeout=2000)
    start = {
        op: {t: Bool(f"{op}_{t}") for t in range(horizon)}
        for op in ops
    }
    assigned = {
        op: {w: Bool(f"{op}_{w}") for w in warps}
        for op in ops
    }

    for name, op in ops.items():
        opt.add(Sum([If(start[name][t], 1, 0) for t in range(horizon)]) == 1)
        for t in range(horizon):
            if t + op.cycles > horizon:
                opt.add(Not(start[name][t]))
        opt.add(Sum([If(assigned[name][w], 1, 0) for w in warps]) == 1)
        for w in warps:
            if w not in op.allowed:
                opt.add(Not(assigned[name][w]))

    for edge in edges:
        for ts in range(horizon):
            for td in range(horizon):
                needed = edge.delay + edge.spill
                if td < ts + needed:
                    opt.add(Implies(start[edge.src][ts], Not(start[edge.dst][td])))

    for residue in range(ii):
        for resource, cap in CAPACITY.items():
            terms = []
            for name, op in ops.items():
                if op.resource != resource:
                    continue
                for t in range(horizon):
                    for c in range(op.cycles):
                        if (t + c) % ii == residue:
                            terms.append(If(start[name][t], 1, 0))
            opt.add(Sum(terms) <= cap)

    for t in range(horizon):
        for w in warps:
            running = []
            for name, op in ops.items():
                for ts in range(max(0, t - op.cycles + 1), t + 1):
                    running.append(And(start[name][ts], assigned[name][w]))
            opt.add(Sum([If(x, 1, 0) for x in running]) <= 1)

    opt.minimize(Sum([If(start[name][t], t, 0) for name in ops for t in range(horizon)]))
    if opt.check() != sat:
        return None

    model = opt.model()
    result = []
    for name in sorted(ops, key=lambda n: next(t for t in range(horizon) if is_true(model.eval(start[n][t])))):
        t = next(t for t in range(horizon) if is_true(model.eval(start[name][t])))
        w = next(w for w in warps if is_true(model.eval(assigned[name][w])))
        result.append((name, t, w))
    return ii, horizon, result


def solve() -> None:
    impl_cost = {
        "3wg_prefetch": {"global_k_tiles": 6, "pipe_edges": 6},
        "3wg_indep_softmax": {"global_k_tiles": 6, "pipe_edges": 4},
        "3wg_splitkv": {"global_k_tiles": 6, "pipe_edges": 0},
        "2wg_no_producer": {"global_k_tiles": 8, "pipe_edges": 4},
        "1wg_serial": {"global_k_tiles": 6, "pipe_edges": 0},
    }
    search = {
        "3wg_prefetch": ((8, 14), (8, 16), (9, 14), (10, 14)),
        "3wg_indep_softmax": ((5, 10), (6, 12), (7, 12), (7, 14), (8, 14)),
        "3wg_splitkv": ((4, 10), (5, 10), (5, 12), (6, 12), (7, 14), (8, 14)),
        "2wg_no_producer": ((6, 12), (7, 12), (8, 12), (8, 14), (9, 14)),
        "1wg_serial": ((10, 14), (11, 16), (12, 18)),
    }
    for variant in ("3wg_prefetch", "3wg_indep_softmax", "3wg_splitkv", "2wg_no_producer", "1wg_serial"):
        print(f"\n== {variant} ==")
        for ii, horizon in search[variant]:
            result = solve_once(variant, ii, horizon)
            if result is None:
                print(f"  II={ii}, L={horizon}: unsat/timeout")
                continue
            found_ii, found_l, schedule = result
            print(f"II = {found_ii}")
            print(f"L  = {found_l}")
            cost = found_ii * 100 + impl_cost[variant]["global_k_tiles"] * 10 + impl_cost[variant]["pipe_edges"]
            print(
                "estimated_cost = "
                f"{cost}  "
                f"(global_k_tiles={impl_cost[variant]['global_k_tiles']}, "
                f"pipe_edges={impl_cost[variant]['pipe_edges']})"
            )
            for name, t, w in schedule:
                print(f"{name:10s} @ {t:2d}  warp_group={w}")
            break
        else:
            print("unsat")


if __name__ == "__main__":
    solve()

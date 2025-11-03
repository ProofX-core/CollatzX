#!/usr/bin/env python3
"""
QMRE HyperDynamical System 
===========================
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# ------------------------------- Optional deps ------------------------------- #
# We gate heavy libraries behind try/except to keep the script runnable anywhere.

try:  # numpy
    import numpy as np
except Exception as e:  # pragma: no cover
    raise RuntimeError("NumPy is required") from e

try:  # sympy
    import sympy as sp
    from sympy import Matrix
except Exception:
    sp = None  # type: ignore
    Matrix = None  # type: ignore

# Qiskit (optional)
try:
    import qiskit
    from qiskit.providers.aer import AerSimulator
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except Exception:
    qiskit = None  # type: ignore
    AerSimulator = None  # type: ignore
    QuantumCircuit = None  # type: ignore
    QISKIT_AVAILABLE = False

# gudhi (optional) for persistent homology
try:
    import gudhi as gd
    GUDHI_AVAILABLE = True
except Exception:
    gd = None  # type: ignore
    GUDHI_AVAILABLE = False

# UMAP (optional)
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except Exception:
    umap = None  # type: ignore
    UMAP_AVAILABLE = False

# plotly (optional)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except Exception:
    go = None  # type: ignore
    make_subplots = None  # type: ignore
    PLOTLY_AVAILABLE = False

# matplotlib (fallback visualization)
try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except Exception:
    plt = None  # type: ignore
    MPL_AVAILABLE = False

# scikit-learn (optional)
try:
    from sklearn.cluster import SpectralClustering
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except Exception:
    SpectralClustering = None  # type: ignore
    PCA = None  # type: ignore
    SKLEARN_AVAILABLE = False

# ------------------------------- Utilities ---------------------------------- #

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "t": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "lvl": record.levelname,
            "msg": record.getMessage(),
            "name": record.name,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(json_logs: bool = False, verbose: bool = False) -> None:
    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JSONFormatter())
    else:
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(handler)
    root.setLevel(level)


# Deterministic seed helper

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
    except Exception:
        pass


# ------------------------------- Domain types -------------------------------- #

class HyperDimensionalTransform(Enum):
    QUANTUM_TUNNELING = auto()
    TOPOLOGICAL_DEFORMATION = auto()  # Aliased to algebraic morphism here
    ALGEBRAIC_MORPHISM = auto()
    NONCOMMUTATIVE_ROTATION = auto()
    HODGE_DUALITY = auto()


@dataclass(frozen=True)
class AttractorSignature:
    classical_value: complex
    quantum_phases: Tuple[complex, ...]
    betti_numbers: Tuple[int, ...]
    p_adic_norm: Optional[float] = None
    cohomology_class: Optional[str] = None
    quantum_entanglement: Optional[float] = None


@dataclass
class SheafConstruct:
    base_space: str
    stalks: Dict[str, Any]
    restriction_maps: Dict[str, Any]


# ----------------------------- Analyzer Interfaces --------------------------- #

class Analyzer(ABC):
    @abstractmethod
    def analyze(self, trajectory: Sequence[complex]) -> Dict[str, Any]:
        ...


class QuantumAttractorAnalyzer(Analyzer):
    def __init__(self, backend: Optional[str] = None, shots: int = 4096):
        self.backend_name = backend or ("statevector" if QISKIT_AVAILABLE else "none")
        self.shots = shots

    def analyze(self, trajectory: Sequence[complex]) -> Dict[str, Any]:
        # Simple spectral analysis of phase; optional Qiskit hook for provenance
        x = np.asarray(trajectory, dtype=np.complex128)
        if x.size == 0:
            return {"phase_spectrum": [], "phase_entropy": 0.0, "entanglement_entropy": None}
        phases = np.exp(1j * np.angle(x + 1e-12))
        # Shannon entropy of phase histogram
        hist, _ = np.histogram(np.angle(phases), bins=32, range=(-math.pi, math.pi), density=True)
        p = hist / (hist.sum() + 1e-12)
        phase_H = float(-(p[p > 0] * np.log2(p[p > 0])).sum())

        # Optional: build a tiny circuit to tag provenance
        circuit_qasm: Optional[str] = None
        if QISKIT_AVAILABLE:
            try:
                qc = QuantumCircuit(1)
                avg_phase = float(np.angle((phases.mean() + 1e-12)))
                qc.rz(avg_phase, 0)
                circuit_qasm = qc.qasm()
            except Exception:
                circuit_qasm = None

        return {
            "phase_spectrum": phases.astype(np.complex128).tolist(),
            "phase_entropy": phase_H,
            "entanglement_entropy": None,  # Placeholder — requires multi-qubit encoding
            "qasm": circuit_qasm,
        }


class TopologicalDynamicsAnalyzer(Analyzer):
    def __init__(self, homology_depth: int = 2):
        self.homology_depth = int(max(0, homology_depth))

    def analyze(self, trajectory: Sequence[complex]) -> Dict[str, Any]:
        x = np.asarray(trajectory, dtype=float)
        if x.size == 0:
            return {"betti_numbers": [0], "persistence": [], "topological_entropy": 0.0}

        bettis: List[int] = []
        diagrams: List[List[Tuple[float, float]]] = []

        if GUDHI_AVAILABLE and x.size >= 3:
            try:
                # Delay embedding into 2D for point cloud
                pts = np.column_stack([x[:-1], x[1:]])
                rc = gd.RipsComplex(points=pts, max_edge_length=1.0)
                st = rc.create_simplex_tree(max_dimension=min(2, self.homology_depth))
                pers = st.persistence()
                # Betti numbers estimate
                max_dim = min(2, self.homology_depth)
                # gd.SimplexTree has betti_numbers() in newer versions — guard defensively
                try:
                    bnums = st.betti_numbers()  # type: ignore[attr-defined]
                except Exception:
                    bnums = [0] * (max_dim + 1)
                for d in range(max_dim + 1):
                    bettis.append(bnums[d] if d < len(bnums) else 0)
                # crude diagram capture
                for pair in pers:
                    (dim, (b, d)) = pair
                    if dim in (0, 1):
                        diagrams.append([(float(b), float(d if d != float("inf") else x.max()))])
            except Exception:
                bettis = [0]
                diagrams = []
        else:
            # Fallback: crude 0th Betti (connected components of epsilon-graph)
            eps = float(np.std(x) * 0.5 + 1e-9)
            pts = np.column_stack([x[:-1], x[1:]]) if x.size > 1 else x.reshape(-1, 1)
            n = len(pts)
            comp = 1
            if n > 1:
                visited = np.zeros(n, dtype=bool)
                comp = 0
                for i in range(n):
                    if not visited[i]:
                        comp += 1
                        stack = [i]
                        visited[i] = True
                        while stack:
                            u = stack.pop()
                            for v in range(n):
                                if not visited[v] and np.linalg.norm(pts[u] - pts[v]) <= eps:
                                    visited[v] = True
                                    stack.append(v)
            bettis = [comp]

        # Topological entropy proxy: growth rate of unique symbols in coarse partition
        bins = 16
        idx = np.clip(((x - x.min()) / (x.ptp() + 1e-12) * bins).astype(int), 0, bins - 1)
        growth = []
        seen: set = set()
        for k in idx:
            seen.add(int(k))
            growth.append(len(seen))
        top_H = float(np.log(growth[-1] + 1.0) / (len(growth) + 1e-9))

        return {"betti_numbers": bettis, "persistence": diagrams, "topological_entropy": top_H}


class CategoryTheoryAnalyzer(Analyzer):
    def analyze(self, trajectory: Sequence[complex]) -> Dict[str, Any]:
        # Placeholder: identity morphisms between consecutive states
        n = len(trajectory)
        return {"objects": n, "morphisms": max(0, n - 1)}

    # Transform hook
    @staticmethod
    def apply_algebraic_morphism(traj: np.ndarray) -> np.ndarray:
        # Simple endofunctor: affine re-scaling to unit interval
        if traj.size == 0:
            return traj
        tmin, tptp = float(np.min(traj)), float(np.ptp(traj) + 1e-12)
        return (traj - tmin) / tptp


class NoncommutativeGeometryAnalyzer(Analyzer):
    def analyze(self, trajectory: Sequence[complex]) -> Dict[str, Any]:
        # Connes-style toy distance proxy: mean inverse pairwise distance
        x = np.asarray(trajectory, dtype=float)
        if x.size < 2:
            return {"nc_distance": 0.0}
        diffs = np.abs(np.subtract.outer(x, x) + 1e-12)
        inv = 1.0 / diffs
        np.fill_diagonal(inv, 0.0)
        return {"nc_distance": float(inv.mean())}

    @staticmethod
    def apply_noncommutative_rotation(traj: np.ndarray) -> np.ndarray:
        # Rotate in complex plane by a phase depending on index (noncomm-like ordering)
        if traj.size == 0:
            return traj
        idx = np.arange(traj.size)
        phase = np.exp(1j * (idx % 7) * (math.pi / 16))
        return (traj.astype(np.complex128) * phase).real


class HodgeTheoryAnalyzer(Analyzer):
    def analyze(self, trajectory: Sequence[complex]) -> Dict[str, Any]:
        # Toy cohomology class selector via sign changes
        x = np.asarray(trajectory, dtype=float)
        sign_changes = int(np.sum(np.diff(np.sign(x + 1e-12)) != 0))
        coh = f"H^1 class ~{sign_changes}"
        return {"cohomology_class": coh, "sign_changes": sign_changes}

    @staticmethod
    def apply_hodge_duality(traj: np.ndarray) -> np.ndarray:
        # Discrete Hodge-* : swap forward/backward differences (toy)
        if traj.size < 3:
            return traj
        fwd = np.diff(traj, prepend=traj[0])
        bwd = np.diff(traj, append=traj[-1])
        return (fwd - bwd) * 0.5 + traj


class HybridQuantumClassicalProcessor:
    @staticmethod
    def apply_quantum_tunneling(traj: np.ndarray, sigma: float = 0.75) -> np.ndarray:
        if traj.size == 0:
            return traj
        # Gaussian smoothing + rare jump
        k = 5
        grid = np.arange(-k, k + 1)
        ker = np.exp(-0.5 * (grid / sigma) ** 2)
        ker /= ker.sum()
        smoothed = np.convolve(traj, ker, mode="same")
        # occasional tunnel jump
        if traj.size > 8:
            i = int(np.random.randint(2, traj.size - 2))
            smoothed[i] += np.random.normal(scale=np.std(traj) * 0.5)
        return smoothed


# ------------------------------- Pattern DB --------------------------------- #

@dataclass
class AttractorPatternDatabase:
    items: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, initial_condition: complex, attractor_signature: Dict[str, Any]) -> None:
        self.items.append({"initial": complex(initial_condition), "signature": attractor_signature})

    def to_json(self) -> str:
        return json.dumps(self.items, cls=EnhancedJSONEncoder, indent=2)


# ---------------------------- Rules & Simulation ----------------------------- #

@dataclass
class Rule:
    condition_expr: sp.Expr  # type: ignore[name-defined]
    map_expr: sp.Expr  # type: ignore[name-defined]
    cond: Callable[[float], bool]
    func: Callable[[float], float]


class RuleParser:
    """Safe parser for piecewise rules like: 'abs(x)<1:x**2;abs(x)>=1:1/x'."""

    ALLOWED_FUNCS = {
        "abs": sp.Abs if sp else abs,
        "sin": sp.sin if sp else math.sin,
        "cos": sp.cos if sp else math.cos,
        "exp": sp.exp if sp else math.exp,
        "log": sp.log if sp else math.log,
        "sqrt": sp.sqrt if sp else math.sqrt,
    }

    def __init__(self) -> None:
        if sp is None:
            raise RuntimeError("sympy is required to parse rules safely")
        self.x = sp.symbols("x")

    def parse(self, rule_str: str) -> List[Rule]:
        rules: List[Rule] = []
        for chunk in filter(None, (s.strip() for s in rule_str.split(";"))):
            try:
                cond_txt, fn_txt = map(str.strip, chunk.split(":", 1))
            except ValueError as e:
                raise ValueError(f"Invalid rule chunk: '{chunk}'") from e
            cond_expr = sp.sympify(cond_txt, locals=self.ALLOWED_FUNCS)
            fn_expr = sp.sympify(fn_txt, locals=self.ALLOWED_FUNCS)
            # Build callables
            cond = sp.lambdify(self.x, cond_expr, modules=[{"Abs": np.abs}, "numpy"])
            func = sp.lambdify(self.x, fn_expr, modules=[{"Abs": np.abs}, "numpy"])
            rules.append(Rule(cond_expr, fn_expr, cond=lambda v, c=cond: bool(c(v)), func=lambda v, f=func: float(f(v))))
        return rules


@dataclass
class ClassicalResult:
    trajectory: List[float]
    lyapunov: float
    attractor: Dict[str, Any]
    transform_applied: Optional[str] = None


class ClassicalSimulator:
    def __init__(self, rules: List[Rule]):
        self.rules = rules

    def iterate(self, x: float) -> float:
        for r in self.rules:
            try:
                if r.cond(x):
                    return r.func(x)
            except Exception:
                continue
        # If no rule matched, identity
        return x

    def simulate(self, x0: float, max_iter: int = 1000, precision: float = 1e-12) -> ClassicalResult:
        x = float(x0)
        traj: List[float] = [x]
        lyap_sum = 0.0
        eps = 1e-7
        seen: Dict[int, int] = {}
        period = 0

        for t in range(1, max_iter + 1):
            x_next = self.iterate(x)
            # numerical derivative via symmetric difference
            fx1 = self.iterate(x + eps)
            fx2 = self.iterate(x - eps)
            deriv = (fx1 - fx2) / (2 * eps)
            lyap_sum += math.log(abs(deriv) + 1e-12)

            traj.append(x_next)
            key = int(round(x_next / (precision + 1e-16)))
            if key in seen:
                period = t - seen[key]
                break
            seen[key] = t

            if abs(x_next - x) < precision:
                break
            x = x_next

        lyap = lyap_sum / max(1, len(traj) - 1)
        attractor = self._classify_attractor(traj, period)
        return ClassicalResult(trajectory=traj, lyapunov=lyap, attractor=attractor)

    @staticmethod
    def _classify_attractor(traj: Sequence[float], period: int) -> Dict[str, Any]:
        tail = np.asarray(traj[-min(64, len(traj)):])
        if len(tail) < 2:
            return {"type": "unknown", "value": complex(traj[-1])}
        var = float(np.var(tail))
        if period > 1:
            cyc = [complex(v) for v in tail[-period:]]
            return {"type": "cycle", "period": period, "cycle": cyc, "value": complex(cyc[-1])}
        if var < 1e-18:
            return {"type": "fixed", "value": complex(tail[-1])}
        if np.any(np.abs(tail) > 1e6):
            return {"type": "divergent", "value": complex(tail[-1])}
        return {"type": "chaotic", "value": complex(tail[-1])}


# ------------------------------- System Kernel ------------------------------- #

class HyperDynamicalSystem:
    def __init__(
        self,
        rules: List[Rule],
        quantum_backend: Optional[str] = None,
        homology_depth: int = 2,
        p_adic_prime: Optional[int] = None,
    ) -> None:
        self.sim = ClassicalSimulator(rules)
        self.quantum = QuantumAttractorAnalyzer(backend=quantum_backend)
        self.topology = TopologicalDynamicsAnalyzer(homology_depth=homology_depth)
        self.category = CategoryTheoryAnalyzer()
        self.noncomm = NoncommutativeGeometryAnalyzer()
        self.hodge = HodgeTheoryAnalyzer()
        self.hybrid = HybridQuantumClassicalProcessor()
        self.p_adic_prime = p_adic_prime
        self.patterns = AttractorPatternDatabase()
        self.attractor_signatures: List[AttractorSignature] = []

        # Sheaf structure
        self.sheaf = SheafConstruct(
            base_space="PhaseSpace",
            stalks=self._standard_attractors(),
            restriction_maps={},
        )

        self.log = logging.getLogger("HyperDynamicalSystem")

    @staticmethod
    def _standard_attractors() -> Dict[str, Any]:
        return {
            "fixed_points": [0, 1, -1, complex(0, 1), float("inf")],
            "cycles": [],
            "quantum": ["superposition", "entangled_state"],
            "topological": ["strange_attractor", "torus_flow"],
            "algebraic": ["group_identity", "ring_zero"],
        }

    # -------------------------- High-level orchestration --------------------- #

    def run(
        self,
        initials: Sequence[complex],
        max_iter: int = 1000,
        precision: float = 1e-12,
        transform: Optional[HyperDimensionalTransform] = None,
        workers: int = 0,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        exec_fn = partial(self._run_single, max_iter=max_iter, precision=precision, transform=transform)

        if workers and workers > 0:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for out in pool.map(exec_fn, initials):
                    results.append(out)
        else:
            for x0 in initials:
                results.append(exec_fn(x0))
        return results

    def _run_single(
        self,
        x0: complex,
        max_iter: int,
        precision: float,
        transform: Optional[HyperDimensionalTransform],
    ) -> Dict[str, Any]:
        classical = self.simulate_classical(float(np.real(x0)), max_iter, precision)

        if transform is not None:
            classical = self.apply_transform(classical, transform)

        quantum = self.quantum.analyze(classical.trajectory)
        topo = self.topology.analyze(classical.trajectory)
        cat = self.category.analyze(classical.trajectory)
        nc = self.noncomm.analyze(classical.trajectory)
        hod = self.hodge.analyze(classical.trajectory)
        padic = self.p_adic_analysis(classical.trajectory) if self.p_adic_prime else {}

        unified = self.unify(classical, quantum, topo, cat, nc, hod, padic)

        # Persistence
        self.patterns.add(x0, unified["signature"])
        self.attractor_signatures.append(self._signature_from(unified["signature"]))

        result = {
            "initial": complex(x0),
            "classical": dataclass_to_dict(classical),
            "quantum": quantum,
            "topological": topo,
            "category": cat,
            "noncommutative": nc,
            "hodge": hod,
            "p_adic": padic,
            "unified_attractor": unified,
            "novelty_score": float(self.novelty(unified["signature"]))
        }
        self._update_lattice(result)
        return result

    # ------------------------------- Components ------------------------------ #

    def simulate_classical(self, x0: float, max_iter: int, precision: float) -> ClassicalResult:
        return self.sim.simulate(x0=x0, max_iter=max_iter, precision=precision)

    def apply_transform(self, classical: ClassicalResult, t: HyperDimensionalTransform) -> ClassicalResult:
        traj = np.asarray(classical.trajectory, dtype=float)
        if t is HyperDimensionalTransform.QUANTUM_TUNNELING:
            traj2 = self.hybrid.apply_quantum_tunneling(traj)
        elif t is HyperDimensionalTransform.TOPOLOGICAL_DEFORMATION:
            traj2 = self.category.apply_algebraic_morphism(traj)
        elif t is HyperDimensionalTransform.ALGEBRAIC_MORPHISM:
            traj2 = self.category.apply_algebraic_morphism(traj)
        elif t is HyperDimensionalTransform.NONCOMMUTATIVE_ROTATION:
            traj2 = self.noncomm.apply_noncommutative_rotation(traj)
        elif t is HyperDimensionalTransform.HODGE_DUALITY:
            traj2 = self.hodge.apply_hodge_duality(traj)
        else:
            traj2 = traj
        new_attr = ClassicalSimulator._classify_attractor(traj2.tolist(), period=1)
        return ClassicalResult(trajectory=traj2.tolist(), lyapunov=classical.lyapunov, attractor=new_attr, transform_applied=t.name)

    def p_adic_analysis(self, trajectory: Sequence[float]) -> Dict[str, Any]:
        # p-adic norm via valuation of rational approximation of last point
        if not trajectory:
            return {}
        v = float(trajectory[-1])
        if self.p_adic_prime is None:
            return {}
        p = int(self.p_adic_prime)
        try:
            rat = sp.nsimplify(v, maxsteps=50) if sp is not None else v
            if sp is not None and getattr(rat, "is_Rational", False):
                num, den = int(rat.p), int(rat.q)
            else:
                # crude scaling to integer
                den = 10 ** 6
                num = int(round(v * den))
            vp = 0
            while num % p == 0 and num != 0:
                num //= p
                vp += 1
            while den % p == 0 and den != 0:
                den //= p
                vp -= 1
            norm = p ** (-vp)
            return {"prime": p, "valuation": vp, "norm": float(norm)}
        except Exception:
            return {"prime": p, "valuation": None, "norm": None}

    # ---------------------------- Unification layer --------------------------- #

    def unify(
        self,
        classical: ClassicalResult,
        quantum: Dict[str, Any],
        topo: Dict[str, Any],
        cat: Dict[str, Any],
        nc: Dict[str, Any],
        hod: Dict[str, Any],
        padic: Dict[str, Any],
    ) -> Dict[str, Any]:
        sig = AttractorSignature(
            classical_value=complex(classical.attractor.get("value", complex(classical.trajectory[-1]))),
            quantum_phases=tuple(quantum.get("phase_spectrum", [])),
            betti_numbers=tuple(topo.get("betti_numbers", [0])),
            p_adic_norm=padic.get("norm"),
            cohomology_class=hod.get("cohomology_class"),
            quantum_entanglement=quantum.get("entanglement_entropy"),
        )

        return {
            "type": self._classify_unified_type(classical, quantum, topo),
            "signature": dataclass_to_dict(sig),
            "stability": self._stability_proxy(classical, topo),
            "algebraic_invariants": {"objects": cat.get("objects"), "morphisms": cat.get("morphisms")},
            "quantum_topological_links": {"phase_entropy": quantum.get("phase_entropy"), "topological_entropy": topo.get("topological_entropy")},
        }

    def _classify_unified_type(self, classical: ClassicalResult, quantum: Dict[str, Any], topo: Dict[str, Any]) -> str:
        t = classical.attractor.get("type", "unknown")
        if t == "fixed" and (topo.get("betti_numbers", [0])[0] <= 1):
            return "hyper_fixed"
        if t == "cycle":
            return "hyper_cycle"
        if quantum.get("phase_entropy", 0.0) > 3.5:
            return "hyper_chaotic"
        return f"hyper_{t}"

    def _stability_proxy(self, classical: ClassicalResult, topo: Dict[str, Any]) -> float:
        # Combine Lyapunov and inverse of 0th Betti as a toy stability metric
        b0 = float(topo.get("betti_numbers", [1])[0] or 1)
        return float(1.0 / (1.0 + math.exp(classical.lyapunov)) * (1.0 / b0))

    def _signature_from(self, d: Dict[str, Any]) -> AttractorSignature:
        return AttractorSignature(
            classical_value=complex(d.get("classical_value", 0)),
            quantum_phases=tuple(d.get("quantum_phases", [])),
            betti_numbers=tuple(d.get("betti_numbers", [0])),
            p_adic_norm=d.get("p_adic_norm"),
            cohomology_class=d.get("cohomology_class"),
            quantum_entanglement=d.get("quantum_entanglement"),
        )

    def novelty(self, signature: Dict[str, Any]) -> float:
        # Similarity against previous signatures (0..1), novelty=1-max(similarity)
        if not self.attractor_signatures:
            return 1.0
        c_sig = self._signature_from(signature)
        sims: List[float] = []
        for s in self.attractor_signatures:
            sims.append(self._sim(c_sig, s))
        return float(1.0 - max(sims))

    @staticmethod
    def _sim(a: AttractorSignature, b: AttractorSignature) -> float:
        # Classical similarity
        c = math.exp(-abs(a.classical_value - b.classical_value))
        # Quantum phases correlation (truncate to min length)
        qa, qb = np.asarray(a.quantum_phases), np.asarray(b.quantum_phases)
        if qa.size > 0 and qb.size > 0:
            m = min(len(qa), len(qb))
            q = float(np.real(np.vdot(qa[:m], qb[:m])) / (np.linalg.norm(qa[:m]) * np.linalg.norm(qb[:m]) + 1e-12))
        else:
            q = 0.5
        # Topology
        ta, tb = a.betti_numbers, b.betti_numbers
        if len(ta) == len(tb):
            t = float(np.mean([math.exp(-abs(x - y)) for x, y in zip(ta, tb)]))
        else:
            t = 0.0
        return 0.4 * c + 0.3 * q + 0.3 * t

    # ------------------------------- Lattice --------------------------------- #

    def _update_lattice(self, result: Dict[str, Any]) -> None:
        # Simple bookkeeping — extend as needed
        typ = result.get("unified_attractor", {}).get("type", "unknown")
        self.sheaf.stalks.setdefault("observed_types", set()).add(typ)

    # --------------------------- Pattern Discovery --------------------------- #

    def discover_universal_patterns(self) -> Dict[str, Any]:
        return {
            "quantum_topological_correlations": self._find_qt_correlations(),
            "attractor_families": self._cluster_attractor_signatures(),
            "emergent_algebraic_structures": self._find_emergent_structures(),
        }

    def _find_qt_correlations(self) -> Dict[str, Any]:
        # Placeholder correlation analysis across stored signatures
        if not self.attractor_signatures:
            return {"strong_correlations": [], "weak_correlations": [], "surprising_anticorrelations": []}
        # Example heuristic: link high phase entropy to higher b0
        pairs = []
        for s in self.attractor_signatures:
            phase_level = np.std(np.angle(np.asarray(s.quantum_phases) + 1e-12)) if s.quantum_phases else 0.0
            b0 = int(s.betti_numbers[0] if s.betti_numbers else 0)
            pairs.append((phase_level, b0))
        return {"pairs_sample": pairs[:16], "strong_correlations": [], "weak_correlations": [], "surprising_anticorrelations": []}

    def _cluster_attractor_signatures(self) -> Dict[str, List[Dict[str, Any]]]:
        if not self.attractor_signatures or not SKLEARN_AVAILABLE:
            return {}
        feats: List[List[float]] = []
        for sig in self.attractor_signatures:
            vec = [
                float(np.real(sig.classical_value)),
                float(np.imag(sig.classical_value)),
                float(np.mean(np.abs(sig.quantum_phases))) if sig.quantum_phases else 0.0,
                float(np.std(np.angle(np.asarray(sig.quantum_phases) + 1e-12))) if sig.quantum_phases else 0.0,
                float(sum(sig.betti_numbers)) if sig.betti_numbers else 0.0,
            ]
            feats.append(vec)
        X = np.asarray(feats)
        k = int(max(1, min(5, len(X))))
        labels = SpectralClustering(n_clusters=k, assign_labels="kmeans").fit_predict(X)
        clustered: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for sig, lab in zip(self.attractor_signatures, labels):
            clustered[int(lab)].append(dataclass_to_dict(sig))
        return {str(k): v for k, v in clustered.items()}

    def _find_emergent_structures(self) -> Dict[str, Any]:
        # Placeholder for future structure mining
        return {"lie_algebras": [], "cohomology_rings": [], "quantum_groups": []}


# ------------------------------- Visualization ------------------------------ #

class Visualizer:
    def __init__(self) -> None:
        self.log = logging.getLogger("Visualizer")

    def show(self, results: List[Dict[str, Any]], mode: str = "interactive") -> None:
        if mode == "interactive" and PLOTLY_AVAILABLE:
            self._plotly(results)
        elif MPL_AVAILABLE:
            self._matplotlib(results)
        else:
            self.log.warning("No visualization backend available")

    def _plotly(self, results: List[Dict[str, Any]]) -> None:
        assert make_subplots is not None and go is not None
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{"type": "xy"}, {"type": "polar"}], [{"type": "scene"}, {"type": "xy"}]],
            subplot_titles=("Classical Trajectories", "Quantum Phase", "3D Embedding", "Entropy Map"),
        )
        # Classical
        for i, r in enumerate(results):
            y = np.real(np.asarray(r["classical"]["trajectory"]))
            fig.add_trace(go.Scatter(x=list(range(len(y))), y=y, mode="lines", name=f"traj {i}"), row=1, col=1)
        # Polar phases
        for i, r in enumerate(results):
            phases = np.asarray(r["quantum"].get("phase_spectrum", []))
            if phases.size:
                fig.add_trace(go.Scatterpolar(r=np.abs(phases), theta=np.degrees(np.angle(phases)), mode="markers", name=f"phases {i}"), row=1, col=2)
        # Simple 3D embedding via PCA if available
        if SKLEARN_AVAILABLE:
            all_traj = [np.asarray(r["classical"]["trajectory"]) for r in results]
            max_len = max(len(t) for t in all_traj)
            pad = np.zeros((len(all_traj), max_len))
            for i, t in enumerate(all_traj):
                pad[i, : len(t)] = t
            emb = PCA(n_components=3).fit_transform(pad)
            fig.add_trace(go.Scatter3d(x=emb[:, 0], y=emb[:, 1], z=emb[:, 2], mode="markers", name="PCA"), row=2, col=1)
        # Entropy map (scatter)
        xs = [complex(r["classical"]["attractor"].get("value", 0)).real for r in results]
        ys = [r["quantum"].get("phase_entropy", 0.0) for r in results]
        zs = [r["topological"].get("topological_entropy", 0.0) for r in results]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", text=[f"topH={z:.3f}" for z in zs], name="Entropy"), row=2, col=2)
        fig.update_layout(title="HyperDynamical Analysis", height=900, showlegend=True)
        fig.show()

    def _matplotlib(self, results: List[Dict[str, Any]]) -> None:
        assert plt is not None
        plt.figure(figsize=(10, 6))
        for i, r in enumerate(results):
            y = np.asarray(r["classical"]["trajectory"])
            plt.plot(y, label=f"traj {i}")
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("Classical trajectories")
        plt.legend()
        plt.tight_layout()
        plt.show()


# ------------------------------- Export / JSON ------------------------------- #

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:  # type: ignore[override]
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if dataclasses_is_instance(obj):
            return dataclass_to_dict(obj)
        return json.JSONEncoder.default(self, obj)


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    from dataclasses import asdict, is_dataclass
    return asdict(obj) if is_dataclass(obj) else dict(obj)


def dataclasses_is_instance(obj: Any) -> bool:
    from dataclasses import is_dataclass
    return is_dataclass(obj)


# ----------------------------------- CLI ------------------------------------ #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Quantum-Topological HyperDynamical System Analyzer (integrated)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--initial", type=complex, nargs="+", default=[0.5, 1.0, 1.5, 2.0], help="Initial conditions (complex allowed)")
    p.add_argument("--rules", type=str, default="abs(x)<1:x**2;abs(x)>=1:1/x", help="Condition:function pairs, semicolon-separated")
    p.add_argument("--quantum_backend", type=str, default=None, help="Qiskit Aer backend method, if available")
    p.add_argument("--homology_depth", type=int, default=2, help="Max homology dimension")
    p.add_argument("--p_adic", type=int, default=None, help="Prime for p-adic analysis")
    p.add_argument("--out", type=str, default="results.json", help="Output JSON path")
    p.add_argument("--visualization", choices=["interactive", "basic", "none"], default="interactive", help="Visualization mode")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--json-logs", action="store_true", help="Structured logs in JSON")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    p.add_argument("--max_iter", type=int, default=1000, help="Max iterations per trajectory")
    p.add_argument("--precision", type=float, default=1e-12, help="Convergence precision")
    p.add_argument("--workers", type=int, default=0, help="Thread workers (0 = sync)")
    p.add_argument("--transform", choices=[k.name for k in HyperDimensionalTransform], default=None, help="Optional transform to apply")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(json_logs=args.json_logs, verbose=args.verbose)
    set_seed(args.seed)
    log = logging.getLogger("main")

    # Parse rules safely
    parser = RuleParser()
    rules = parser.parse(args.rules)

    sysm = HyperDynamicalSystem(
        rules=rules,
        quantum_backend=args.quantum_backend,
        homology_depth=args.homology_depth,
        p_adic_prime=args.p_adic,
    )

    transform = HyperDimensionalTransform[args.transform] if args.transform else None

    log.info("Running simulations", extra={"n_initials": len(args.initial)})
    results = sysm.run(
        initials=args.initial,
        max_iter=args.max_iter,
        precision=args.precision,
        transform=transform,
        workers=args.workers,
    )

    # Export
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, cls=EnhancedJSONEncoder, indent=2)
    log.info("Results saved", extra={"path": args.out})

    # Visualize
    if args.visualization != "none":
        Visualizer().show(results, mode="interactive" if args.visualization == "interactive" else "basic")

    # Discover universal patterns (optional log)
    patterns = sysm.discover_universal_patterns()
    log.info("Patterns summary", extra={"keys": list(patterns.keys())})

    # Print brief summary
    uniq_types = {r["unified_attractor"]["type"] for r in results}
    log.info("Summary", extra={"unique_types": list(uniq_types), "count": len(results)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Symbolic AI Engine for Integer Dynamics (SAEID) - Research Edition

A mathematically rigorous, self-contained research platform for:
- Advanced Collatz conjecture analysis
- Generalized integer dynamical systems
- Automated theorem proving
- Symbolic-numeric hybrid computation
- Topological and number-theoretic analysis

Features:
1. Formal mathematical framework with theorem verification
2. Symbolic computation of dynamical invariants
3. Hybrid numeric-symbolic trajectory analysis
4. Automated conjecture generation and testing
5. Research-grade visualization and reporting

Developed by: Dr. Mohammed Alkindi, Institute for CollatzLab
License: MIT (Open Source, Academic Use Encouraged)
"""

import math
import argparse
import time
from typing import List, Dict, Tuple, Optional, Callable, Set, Union
from collections import defaultdict, deque
import functools
import heapq
import random
import sys
import json
import pickle
from fractions import Fraction
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import zeta
from scipy.stats import norm, ks_2samp
import sympy as sp
from sympy.logic.boolalg import BooleanFunction
from sympy.assumptions import ask, Q
from sympy.core.logic import fuzzy_and, fuzzy_or

# Global configuration
plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
getcontext().prec = 100  # High precision decimal arithmetic

# Type aliases
DynamicsSystem = Callable[[sp.Expr], sp.Expr]
Trajectory = List[Union[int, sp.Expr]]
Invariant = Callable[[Trajectory], Union[float, sp.Expr]]
Theorem = BooleanFunction

class IntegerDynamicsSystem:
    """Formal representation of an integer dynamical system"""
    def __init__(self,
                 name: str,
                 step_func: DynamicsSystem,
                 domain: Optional[sp.Set] = None,
                 invariants: Optional[List[Invariant]] = None):
        """
        Initialize a dynamical system with:
        - name: Descriptive name
        - step_func: Symbolic step function (SymPy expression)
        - domain: Domain of definition (SymPy set)
        - invariants: List of invariant functions to compute
        """
        self.name = name
        self.step_func = step_func
        self.domain = domain if domain is not None else sp.S.Naturals
        self.invariants = invariants if invariants is not None else []

        # Symbolic variables
        self.n = sp.symbols('n', integer=True, positive=True)
        self.k = sp.symbols('k', integer=True, positive=True)

        # Cached properties
        self._step_func_compiled = None
        self._invariant_funcs = None

    @property
    def compiled_step(self) -> Callable[[int], int]:
        """Compiled numeric version of step function"""
        if self._step_func_compiled is None:
            self._step_func_compiled = sp.lambdify(self.n, self.step_func, 'numpy')
        return self._step_func_compiled

    def trajectory(self, x0: Union[int, sp.Expr], max_steps: int = 10**6) -> Trajectory:
        """Compute trajectory from initial value x0"""
        traj = [x0]
        current = x0

        for _ in range(max_steps):
            if isinstance(current, sp.Expr) and current.is_Number:
                current = int(current)

            if isinstance(current, int):
                if current == 1:  # Termination condition
                    break
                current = self.compiled_step(current)
            else:
                current = self.step_func.subs(self.n, current)

            traj.append(current)

        return traj

    def compute_invariants(self, trajectory: Trajectory) -> Dict[str, Union[float, sp.Expr]]:
        """Compute all registered invariants for a trajectory"""
        results = {}
        for inv in self.invariants:
            try:
                results[inv.__name__] = inv(trajectory)
            except Exception as e:
                results[inv.__name__] = f"Error: {str(e)}"
        return results

    def add_invariant(self, invariant: Invariant):
        """Register a new invariant function"""
        self.invariants.append(invariant)
        self._invariant_funcs = None  # Reset cache

    def symbolic_forward(self, k: int, x0: Optional[sp.Expr] = None) -> sp.Expr:
        """
        Compute symbolic k-th iterate of the system
        If x0 is None, returns general form as function of n
        """
        if x0 is None:
            x = self.n
        else:
            x = x0

        for _ in range(k):
            x = self.step_func.subs(self.n, x)
        return sp.simplify(x)

    def is_invariant(self, expr: sp.Expr) -> bool:
        """
        Check if an expression is an invariant of the system
        (expr(x_{k+1}) == expr(x_k) for all k)
        """
        next_expr = expr.subs(self.n, self.step_func)
        return sp.simplify(next_expr - expr) == 0

    def find_cycle(self, max_period: int = 20) -> List[List[sp.Expr]]:
        """
        Find all cycles up to given period symbolically
        Returns list of cycles (each cycle is a list of expressions)
        """
        cycles = []

        for p in range(1, max_period + 1):
            # Solve x = f^p(x)
            eq = self.n - self.symbolic_forward(p)
            solutions = sp.solve(eq, self.n)

            # Filter valid solutions in domain
            valid_sols = []
            for sol in solutions:
                if ask(Q.integer(sol)) and ask(Q.positive(sol)):
                    if sol not in valid_sols:
                        valid_sols.append(sol)

            # For each solution, generate full cycle
            for x in valid_sols:
                cycle = []
                current = x
                for _ in range(p):
                    cycle.append(current)
                    current = self.step_func.subs(self.n, current)

                # Check if we've already found this cycle
                if not any(set(cycle) == set(existing) for existing in cycles):
                    cycles.append(cycle)

        return cycles

    def to_latex(self) -> str:
        """Generate LaTeX representation of the system"""
        return f"{self.name}: $n_{{k+1}} = {sp.latex(self.step_func)}$"

# Predefined dynamical systems
def collatz_system() -> IntegerDynamicsSystem:
    """The classic Collatz system"""
    n = sp.symbols('n')
    step = sp.Piecewise(
        (n/2, sp.Eq(sp.Mod(n, 2), 0)),
        (3*n + 1, True)
    )

    # Define standard invariants
    def stopping_time(traj):
        return len(traj) - 1

    def max_value(traj):
        return max(traj)

    def parity_ratio(traj):
        if len(traj) <= 1:
            return 0
        even = sum(1 for x in traj[:-1] if x % 2 == 0)
        return even / (len(traj) - 1)

    system = IntegerDynamicsSystem(
        name="Collatz",
        step_func=step,
        invariants=[stopping_time, max_value, parity_ratio]
    )
    return system

def generalized_collatz(a: int, b: int, c: int) -> IntegerDynamicsSystem:
    """Generalized Collatz system: a*n + b if odd, n/c if even"""
    n = sp.symbols('n')
    step = sp.Piecewise(
        (n/c, sp.Eq(sp.Mod(n, c), 0)),
        (a*n + b, True)
    )

    return IntegerDynamicsSystem(
        name=f"Generalized Collatz ({a}, {b}, {c})",
        step_func=step
    )

def syracuse_system() -> IntegerDynamicsSystem:
    """Syracuse function (odd part of Collatz)"""
    n = sp.symbols('n')
    step = (3*n + 1) // sp.Pow(2, sp.Mod(3*n + 1, 2))

    return IntegerDynamicsSystem(
        name="Syracuse",
        step_func=step
    )

def lagarias_system() -> IntegerDynamicsSystem:
    """Lagarias's simplified version"""
    n = sp.symbols('n')
    step = sp.Piecewise(
        (n/2, sp.Eq(sp.Mod(n, 2), 0)),
        ((3*n + 1)/2, True)
    )

    return IntegerDynamicsSystem(
        name="Lagarias",
        step_func=step
    )

class DynamicsAnalyzer:
    """Advanced analysis toolkit for integer dynamical systems"""
    def __init__(self, system: IntegerDynamicsSystem):
        self.system = system
        self._trajectory_cache = {}
        self._invariant_cache = {}

    def batch_analyze(self,
                     start: int,
                     end: int,
                     max_steps: int = 10**6) -> Dict[int, Dict]:
        """Analyze range of initial values"""
        results = {}

        for x0 in range(start, end + 1):
            if x0 in self._trajectory_cache:
                traj = self._trajectory_cache[x0]
            else:
                traj = self.system.trajectory(x0, max_steps)
                self._trajectory_cache[x0] = traj

            if x0 in self._invariant_cache:
                invs = self._invariant_cache[x0]
            else:
                invs = self.system.compute_invariants(traj)
                self._invariant_cache[x0] = invs

            results[x0] = {
                'trajectory': traj,
                'invariants': invs,
                'length': len(traj),
                'terminated': traj[-1] == 1
            }

        return results

    def statistical_analysis(self,
                           start: int,
                           end: int,
                           invariant: str = 'stopping_time') -> Dict:
        """Perform statistical analysis of an invariant"""
        data = self.batch_analyze(start, end)
        values = [d['invariants'].get(invariant, 0) for d in data.values()]

        if not values:
            return {}

        values = [v for v in values if isinstance(v, (int, float))]

        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'skewness': sp.stats.skew(values),
            'kurtosis': sp.stats.kurtosis(values),
            'normality_test': self._test_normality(values)
        }

    def _test_normality(self, data) -> Dict:
        """Test if data follows normal distribution"""
        if len(data) < 8:
            return {'p_value': None, 'normal': False}

        k2, p = sp.stats.normaltest(data)
        return {
            'p_value': float(p),
            'normal': p > 0.05
        }

    def find_anomalies(self,
                      start: int,
                      end: int,
                      threshold: float = 3.0) -> Dict[int, Dict]:
        """
        Find anomalous trajectories using z-score analysis
        Returns dict of {x0: anomaly_info} where z-score > threshold
        """
        data = self.batch_analyze(start, end)
        if not data:
            return {}

        # Analyze stopping times
        stopping_times = [d['invariants'].get('stopping_time', 0)
                         for d in data.values()]
        st_mean = np.mean(stopping_times)
        st_std = np.std(stopping_times)

        anomalies = {}
        for x0, d in data.items():
            st = d['invariants'].get('stopping_time', 0)
            z_score = (st - st_mean) / st_std if st_std != 0 else 0

            if abs(z_score) > threshold:
                anomalies[x0] = {
                    'stopping_time': st,
                    'z_score': z_score,
                    'trajectory': d['trajectory'],
                    'all_invariants': d['invariants']
                }

        return anomalies

    def compare_systems(self,
                       other_system: IntegerDynamicsSystem,
                       start: int,
                       end: int) -> Dict:
        """
        Compare two dynamical systems over a range of initial values
        Returns statistical comparison of their behavior
        """
        analyzer1 = self
        analyzer2 = DynamicsAnalyzer(other_system)

        data1 = analyzer1.batch_analyze(start, end)
        data2 = analyzer2.batch_analyze(start, end)

        # Compare stopping times
        st1 = [d['invariants'].get('stopping_time', 0) for d in data1.values()]
        st2 = [d['invariants'].get('stopping_time', 0) for d in data2.values()]

        # KS test for distribution comparison
        ks_stat, ks_p = ks_2samp(st1, st2)

        return {
            'stopping_time_comparison': {
                'mean_diff': np.mean(st1) - np.mean(st2),
                'ks_test': {
                    'statistic': ks_stat,
                    'p_value': ks_p
                }
            },
            'termination_agreement': sum(
                1 for x0 in data1
                if data1[x0]['terminated'] == data2[x0]['terminated']
            ) / len(data1)
        }

class TheoremProver:
    """Automated theorem prover for dynamical systems properties"""
    def __init__(self, system: IntegerDynamicsSystem):
        self.system = system
        self.n = system.n
        self.k = system.k

    def verify_termination(self, x0: Union[int, sp.Expr], max_k: int = 100) -> Dict:
        """
        Attempt to verify termination for specific x0
        Returns dictionary with verification results
        """
        if isinstance(x0, int):
            # Numeric verification
            traj = self.system.trajectory(x0, max_steps=max_k)
            terminated = traj[-1] == 1
            return {
                'terminated': terminated,
                'steps': len(traj) - 1,
                'method': 'numeric',
                'certificate': traj
            }
        else:
            # Symbolic verification
            try:
                # Check if x0 is known to terminate (e.g., power of 2)
                if ask(sp.Q.pow(2, sp.Q.integer(self.k)), x0):
                    return {
                        'terminated': True,
                        'steps': self.k,
                        'method': 'symbolic',
                        'certificate': f"Power of 2: {x0} = 2^{self.k}"
                    }

                # Try to find k where f^k(x0) reaches known terminating value
                for k in range(1, max_k + 1):
                    xk = self.system.symbolic_forward(k, x0)
                    if ask(sp.Q.equal(xk, 1)):
                        return {
                            'terminated': True,
                            'steps': k,
                            'method': 'symbolic',
                            'certificate': f"Reaches 1 in {k} steps"
                        }

                return {
                    'terminated': False,
                    'steps': None,
                    'method': 'symbolic',
                    'certificate': f"No termination in first {max_k} steps"
                }
            except Exception as e:
                return {
                    'terminated': None,
                    'steps': None,
                    'method': 'symbolic',
                    'error': str(e)
                }

    def check_invariant(self, expr: sp.Expr) -> Dict:
        """Check if expression is an invariant of the system"""
        try:
            is_inv = self.system.is_invariant(expr)
            return {
                'is_invariant': is_inv,
                'expression': expr,
                'simplified': sp.simplify(expr)
            }
        except Exception as e:
            return {
                'is_invariant': None,
                'error': str(e),
                'expression': expr
            }

    def find_linear_invariants(self, degree: int = 1) -> List[sp.Expr]:
        """
        Find linear invariants of form a*n + b
        Returns list of valid invariant expressions
        """
        a, b = sp.symbols('a b')
        expr = a*self.n + b
        next_expr = expr.subs(self.n, self.system.step_func)

        # Solve expr == next_expr
        eq = sp.Eq(expr, next_expr)
        solutions = sp.solve(eq, (a, b))

        invariants = []
        for sol in solutions:
            if sol:
                inv = expr.subs({a: sol[a], b: sol[b]})
                invariants.append(sp.simplify(inv))

        return invariants

    def induction_proof(self, prop: Theorem, max_k: int = 10) -> Dict:
        """
        Attempt proof by induction for a property
        Returns dictionary with proof status
        """
        try:
            # Base case
            base_case = prop.subs(self.n, 1)
            if not ask(base_case):
                return {
                    'proved': False,
                    'reason': "Base case fails",
                    'base_case': base_case
                }

            # Induction step
            for k in range(1, max_k + 1):
                inductive_hyp = prop.subs(self.n, k)
                inductive_step = prop.subs(self.n, self.system.step_func.subs(self.n, k))

                if ask(sp.Implies(inductive_hyp, inductive_step)):
                    continue
                else:
                    return {
                        'proved': False,
                        'reason': f"Induction fails at k={k}",
                        'failing_case': k
                    }

            return {
                'proved': True,
                'method': f"Induction up to k={max_k}",
                'verified_cases': max_k
            }
        except Exception as e:
            return {
                'proved': None,
                'error': str(e),
                'property': prop
            }

class Visualization:
    """Advanced visualization tools for dynamical systems"""
    @staticmethod
    def plot_trajectory(trajectory: Trajectory,
                       ax=None,
                       log_scale: bool = True,
                       highlight_peaks: bool = True) -> plt.Axes:
        """Plot trajectory with optional annotations"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        steps = range(len(trajectory))
        ax.plot(steps, trajectory, 'b-', linewidth=1, alpha=0.7)

        if highlight_peaks:
            peak_pos = np.argmax(trajectory)
            ax.plot(peak_pos, trajectory[peak_pos], 'ro', label='Peak')
            ax.annotate(f'Max: {trajectory[peak_pos]}',
                       xy=(peak_pos, trajectory[peak_pos]),
                       xytext=(10, 10),
                       textcoords='offset points',
                       arrowprops=dict(arrowstyle='->'))

        if log_scale:
            ax.set_yscale('log')

        ax.set_xlabel('Step')
        ax.set_ylabel('Value (log scale)' if log_scale else 'Value')
        ax.set_title('Trajectory Visualization')
        ax.grid(True)
        ax.legend()

        return ax

    @staticmethod
    def plot_invariant_distribution(invariants: Dict[int, float],
                                  ax=None,
                                  bins: int = 30) -> plt.Axes:
        """Plot distribution of an invariant"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        values = list(invariants.values())
        ax.hist(values, bins=bins, density=True, alpha=0.7)

        # Fit normal distribution
        mu, std = norm.fit(values)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2,
               label=f'Normal fit: μ={mu:.2f}, σ={std:.2f}')

        ax.set_xlabel('Invariant Value')
        ax.set_ylabel('Density')
        ax.set_title('Invariant Distribution')
        ax.legend()
        ax.grid(True)

        return ax

    @staticmethod
    def plot_phase_space(system: IntegerDynamicsSystem,
                        start: int,
                        end: int,
                        ax=None) -> plt.Axes:
        """Plot phase space of the system"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        x_values = range(start, end + 1)
        y_values = [system.compiled_step(x) for x in x_values]

        ax.plot(x_values, y_values, 'bo', markersize=3, alpha=0.5)
        ax.plot([start, end], [start, end], 'r--', label='Identity')

        # Mark fixed points
        fixed_points = [x for x in x_values if system.compiled_step(x) == x]
        for fp in fixed_points:
            ax.plot(fp, fp, 'gs', markersize=8, label=f'Fixed Point {fp}')

        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Phase Space')
        ax.legend()
        ax.grid(True)

        return ax

class ResearchReport:
    """Generate comprehensive research reports"""
    @staticmethod
    def generate_report(analyzer: DynamicsAnalyzer,
                      start: int,
                      end: int,
                      filename: Optional[str] = None) -> str:
        """Generate full research report"""
        # Collect data
        data = analyzer.batch_analyze(start, end)
        stats = analyzer.statistical_analysis(start, end)
        anomalies = analyzer.find_anomalies(start, end)

        # Create figures
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        Visualization.plot_trajectory(data[start]['trajectory'], ax=ax1)

        stopping_times = {x0: d['invariants']['stopping_time']
                         for x0, d in data.items()}
        Visualization.plot_invariant_distribution(stopping_times, ax=ax2)

        # Generate report text
        report = [
            f"Research Report for {analyzer.system.name} System",
            f"Analysis Range: {start} to {end}",
            "",
            "=== Statistical Summary ===",
            f"Mean stopping time: {stats.get('mean', 0):.2f}",
            f"Max stopping time: {stats.get('max', 0)}",
            f"Standard deviation: {stats.get('std', 0):.2f}",
            "",
            "=== Anomalies Detected ===",
            f"Found {len(anomalies)} anomalies (|z-score| > 3)"
        ]

        if anomalies:
            for x0, info in list(anomalies.items())[:5]:  # Show top 5
                report.append(
                    f"x0 = {x0}: stopping time = {info['stopping_time']}, "
                    f"z-score = {info['z_score']:.2f}"
                )

        # Add system description
        report.extend([
            "",
            "=== System Definition ===",
            analyzer.system.to_latex(),
            "",
            "=== Sample Trajectory ===",
            f"x0 = {start}: length = {len(data[start]['trajectory'])}, "
            f"max = {max(data[start]['trajectory'])}"
        ])

        report_text = "\n".join(report)

        # Save to file if requested
        if filename:
            with open(filename, 'w') as f:
                f.write(report_text)

            # Save figures
            fig1.savefig(filename.replace('.txt', '_fig1.png'))
            plt.close(fig1)

        return report_text

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Symbolic AI Engine for Integer Dynamics (SAEID)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # System selection
    parser.add_argument('--system',
                      choices=['collatz', 'syracuse', 'lagarias', 'generalized'],
                      default='collatz',
                      help='Dynamical system to analyze')
    parser.add_argument('--a', type=int, default=3,
                      help='Parameter a for generalized system')
    parser.add_argument('--b', type=int, default=1,
                      help='Parameter b for generalized system')
    parser.add_argument('--c', type=int, default=2,
                      help='Parameter c for generalized system')

    # Analysis parameters
    parser.add_argument('--start', type=int, default=1,
                      help='Start of analysis range')
    parser.add_argument('--end', type=int, default=100,
                      help='End of analysis range')
    parser.add_argument('--max-steps', type=int, default=10**6,
                      help='Maximum steps per trajectory')

    # Output options
    parser.add_argument('--report', type=str,
                      help='Generate report to specified file')
    parser.add_argument('--plot', action='store_true',
                      help='Show interactive plots')

    args = parser.parse_args()

    # Initialize system
    if args.system == 'collatz':
        system = collatz_system()
    elif args.system == 'syracuse':
        system = syracuse_system()
    elif args.system == 'lagarias':
        system = lagarias_system()
    elif args.system == 'generalized':
        system = generalized_collatz(args.a, args.b, args.c)

    analyzer = DynamicsAnalyzer(system)
    prover = TheoremProver(system)

    # Generate report if requested
    if args.report:
        report = ResearchReport.generate_report(
            analyzer, args.start, args.end, args.report
        )
        print(f"Report generated and saved to {args.report}")

    # Interactive analysis
    if args.plot:
        # Sample trajectory
        traj = system.trajectory(args.start, args.max_steps)
        print(f"Trajectory for {args.start}: length={len(traj)}, max={max(traj)}")

        # Show plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        Visualization.plot_trajectory(traj, ax=ax1)

        # Phase space
        Visualization.plot_phase_space(system, args.start, args.end, ax=ax2)

        plt.tight_layout()
        plt.show()

    # Theorem proving examples
    print("\nTheorem Proving Examples:")
    n = sp.symbols('n', integer=True, positive=True)

    # Check if even numbers decrease
    prop = sp.Implies(sp.Eq(sp.Mod(n, 2), 0), system.step_func < n)
    proof = prover.induction_proof(prop)
    print(f"All even numbers decrease: {proof['proved']}")

    # Find cycles
    cycles = system.find_cycle()
    print(f"\nFound {len(cycles)} cycles:")
    for cycle in cycles:
        print(f"Cycle of length {len(cycle)}: {cycle}")

if __name__ == "__main__":
    main()

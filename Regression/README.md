# ULTRA: Unified Learning & Theory-Rich Architecture for Integer Dynamics

A once-in-a-generation symbolic-empirical framework for modeling, analyzing, and discovering mathematical structure in integer dynamics. ULTRA fuses empirical machine learning, symbolic regression, topological analysis, and neural-symbolic integration into a unified research-grade engine capable of:

* Automated conjecture generation
* Topological attractor analysis
* Symbolic feature synthesis
* Neural-symbolic hybrid modeling
* Fully interactive, publication-ready visualizations

> "This isn't just a Collatz engine. It's an operating system for integer phenomena."

---

## 🤖 Key Features

### ⚛️ Symbolic Reasoning & Conjecture Generation

* Integrated `PySR`, `SymPy`, and optional `Z3` SMT solver
* Auto-converts empirical patterns into symbolic equations
* Theorem objects include confidence, proof status, and dependency tracking

### 🌍 Topological Analysis of Dynamical Sequences

* Constructs directed transition graphs of sequence paths
* Computes spectral gap, mixing time, attractor basins
* Outputs a `TopologicalAnalysis` dataclass with basin sizes

### 🧠 Empirical Modeling (Ensemble + Neural)

* Gradient Boosting, Random Forests, MLP, Gaussian Process
* Automatic feature selection via Random Forest feature importances
* Ensemble voting model with cross-validated weights

### ⚖️ Neural-Symbolic Hybridization

* Extracts features from symbolic formulas
* Trains empirical models over combined symbolic and statistical features
* Produces `VotingRegressor` hybrids for interpretability + performance

### 🌀 Feature Space of Integer Dynamics

* Arithmetic: log(n), sqrt(n), n/log(n)
* Number theory: parity, modular signatures, totient, ω(n), Ω(n)
* Sequence features: stopping time, entropy, peak ratios
* Spectral: FFT-based parity signature analysis

### 📊 Interactive Dashboard

* Built with `Plotly` and `Seaborn`
* 3D dynamics landscape
* Residual plots, feature importances, and symbolic equations
* Exported as a standalone `.html` for sharing or publishing

---

## 🚀 Getting Started

### Requirements

```
pip install -r requirements.txt
```

Key dependencies:

* `numpy`, `scipy`, `sympy`, `pysr`
* `sklearn`, `torch`, `plotly`, `networkx`
* `hydra-core`, `omegaconf`, `beartype`, `cloudpickle`

### Running ULTRA

```bash
python ultra.py
```

Hydra-based config allows runtime overrides:

```bash
python ultra.py mode=HYBRID max_n=100000 symbolic_complexity=5
```

---

## 📑 Outputs

| File                     | Description                             |
| ------------------------ | --------------------------------------- |
| `results/results.json`   | Main results: models, metrics, theorems |
| `results/results.yaml`   | YAML equivalent of outputs              |
| `results/ultra.log`      | Logging trace of pipeline               |
| `results/dashboard.html` | Full interactive Plotly dashboard       |
| `results/*.pkl`          | Trained models saved via cloudpickle    |

---

## 🎓 Academic Utility

ULTRA is designed for:

* Research labs studying integer dynamics (e.g., Collatz, 5x+1, parity dynamics)
* Papers on neural-symbolic theorem synthesis
* Educational tools for number theory, dynamical systems, or AI x Math
* Computational mathematics competitions and grant proposals

---

## 🔧 Architecture Overview

```
ultra/
├── ultra.py               # Entry point (Hydra-managed)
├── engine/                # UltraEngine and orchestration logic
├── symbolic/              # SymbolicReasoner, Theorem, parsing
├── empirical/             # EmpiricalModeler + Ensemble logic
├── hybrid/                # HybridIntegrator (neural-symbolic)
├── features/              # FeatureSpace definitions
├── topology/              # SequenceAnalyzer + attractor logic
├── viz/                   # UltraVisualizer + dashboard composer
├── config/                # Hydra + OmegaConf defaults
└── results/               # Auto-generated output folder
```

---

## 🤝 Acknowledgements

* `PySR`: Symbolic regression powered by Julia under the hood
* `SymPy`, `Z3`, `scikit-learn`: Math meets AI
* `Hydra`, `OmegaConf`: Clean experiment management
* `Plotly`, `Seaborn`: Research-grade visualizations

---

## 🎯 Future Directions

* Quantum-enhanced symbolic search
* Reinforcement theorem discovery (proof as RL)
* Collatz across moduli and 2-adic trees
* Integration with Lean, Coq, or Isabelle for formal verification

---

## 🌟 Author

Built by **Alkindi**, a research engineer in symbolic AI and computational mathematics.

For collaborations, licensing, or publishing:

* Email: [alkindi.research@gmail.com](mailto:alkindi.research@gmail.com)
* GitHub: [github.com/AlkindiProjects](https://github.com/AlkindiProjects)
* Research Site: Coming soon

---

> "ULTRA doesn’t just simulate integer systems. It invites them to explain themselves."

---

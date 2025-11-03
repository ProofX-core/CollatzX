# Collatz Research Platform

**Advanced Computational Framework for Symbolic Dynamics, ML Analysis, and Visualization of Generalized Collatz Sequences**

---

## 🧠 Overview

This project is a **hybrid simulation-analysis platform** designed to explore the dynamics of **generalized Collatz sequences** using state-of-the-art tooling. It combines symbolic logic, parallel processing, feature engineering, and machine learning models into a unified CLI-driven research engine.

Key capabilities include:

* Fast simulation of `(a * x + b) / d` variants
* Cycle detection, divergence analysis, entropy calculations
* 100+ feature extractors: parity, jump, modular, topological, spectral
* ML classification, clustering, and visualization (via sklearn, TensorFlow, PyTorch)
* Interactive Dash-based dashboards and static Plotly visualizations
* CLI batch processing with support for caching, parallelism, and export

---

## 🔎 Use Cases

* Empirical exploration of Collatz-style functions (3x+1, 5x+1, custom a/b/d)
* Symbolic feature extraction and entropy analysis of integer sequences
* Cycle detection, parity signature extraction, jump metrics
* Machine learning prediction of termination outcomes or entropy classes
* Topological/spectral pattern discovery across large seed batches

---

## ⚖️ Core Modules

### `CollatzSequenceSimulator`

Generalized simulation engine with caching, peak tracking, cycle detection, and entropy calculation.

### `AdvancedFeatureExtractor`

Modular, registry-based system extracting parity transitions, jump statistics, entropy, modular properties, spectral FFT features, and topological metrics.

### `CollatzMLModel`

Trainable ML suite with:

* Random Forest + feature importances
* Neural nets with auto-architecture
* KMeans, PCA, t-SNE clustering
* Visualization via Plotly

### `CollatzVisualizer`

Tools for visualizing sequences, feature distributions, and correlation matrices in interactive or static forms.

### `CollatzResearchCLI`

Fully CLI-operable research interface supporting batch runs, ML analysis, visual output, and config overrides.

---

## 🔄 CLI Usage

```bash
python3 main.py --range 1:100 --a 3 --b 1 --d 2 \
--feature-sets all --ml --visualize --cache --parallel --format parquet
```

### Options

* `--range`: seed range to simulate
* `--a`, `--b`, `--d`: generalized Collatz parameters
* `--feature-sets`: choose specific sets (basic, jump, spectral...)
* `--ml`: enable ML pipeline (classification + clustering)
* `--visualize`: generate HTML plots
* `--interactive`: launch Dash dashboard (port 8050)
* `--format`: export format (csv/json/parquet/hdf5)

---

## 🎓 Dependencies

* Python 3.8+
* `numpy`, `pandas`, `matplotlib`, `plotly`, `seaborn`, `sympy`
* `scikit-learn`, `tensorflow`, `torch`
* `tqdm`, `argparse`, `dash`

Install via:

```bash
pip install -r requirements.txt
```

---

## 🎯 Output

* **Results**: Feature data per seed (CSV, JSON, Parquet)
* **Visualizations**: HTML plots (feature dist, sequence curves)
* **ML Reports**: Feature importances, classification reports, clustering visuals
* **Logs**: Full experiment logs per run

---

## ✨ Vision

This platform is designed as a launchpad for **next-generation Collatz analysis** that blends symbolic logic, statistical learning, and algorithmic visualization. From numerical recursion to topological inference, it treats integer dynamics as data-rich, pattern-bearing systems.

*"In every divergent path lies a signature of recursion. This engine reads that signature."*

---

## ⚛️ License

MIT

## 📱 Author

Mohammed Alkindi
UCL Engineering | CollatzLab | AI Systems Architect

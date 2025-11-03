# Quantum-Classical Collatz Analyzer v3.0

A hybrid, production-grade research platform for analyzing the Collatz Conjecture using classical computation, machine learning, and quantum circuit execution. Built for precision, scale, and symbolic experimentation.

---

##  Features

* **Modular Architecture** – Cleanly separated logic for ML, quantum, and classical analysis
* **Machine Learning Integration** – Fast heuristic predictions with confidence scores
* **Quantum Backend Support** – Qiskit-compatible execution and backend toggling
* **Caching Systems** – Thread-safe LRU and LFU caching for high-performance reuse
* **Telemetry & Logging** – Built-in observability with Prometheus-compatible metrics
* **Error Handling** – Fault-tolerant fallback system with structured logs
* **FastAPI + CLI Ready** – Can be run as API service or CLI tool

---

##  Setup

### Requirements

```bash
Python 3.9+
Qiskit
SymPy
NumPy
FastAPI
Uvicorn
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run via CLI

```bash
python main.py --number 27 --method hybrid --visualize sequence
```

### Run API

```bash
uvicorn app:app --reload
```

---

## 📁 Directory Structure

```
├── main.py
├── analyzer/
│   ├── __init__.py
│   ├── core.py
│   ├── ml/
│   │   └── predictor.py
│   ├── quantum/
│   │   └── executor.py
│   ├── utils/
│   │   └── cache.py
├── config/
│   └── settings.yaml
├── app/
│   └── api.py
├── tests/
│   └── test_analyzer.py
├── docs/
│   └── architecture.md
```

---

## 📊 Example Output

```json
{
  "number": 27,
  "steps": 111,
  "method": "ML",
  "is_prime": false,
  "ml_confidence": 0.82,
  "sequence": null,
  "quantum_metrics": null
}
```

---

## 📘 Citations

* Alkindi, M. (2025). *Empirical and Theoretical Explorations of the Collatz Conjecture*. \[arXiv preprint pending]
* Qiskit contributors. IBM Quantum.

---

## 🧠 Author

Mohammed Alkindi
*Industrial-Grade Symbolic Systems, Cognitive Infrastructure, and Hybrid Simulation Engineering.*

[GitHub](https://github.com/alkindimath) | [LinkedIn](https://linkedin.com/in/mohammed-alkindi-51a5a62b2) | [Website](https://nimble-mind.com)

---

## 📄 License

MIT License – Open for research, learning, and symbolic innovation.

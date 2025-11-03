# Collatz Variations Calculator

A Python command-line tool to explore different variations of the Collatz conjecture by applying customizable iteration rules to integers within a user-defined range.

---

## Features

- Supports multiple Collatz-like variations with customizable odd/even rules.
- Memoization for efficient step calculations.
- Handles ranges of natural numbers with user input validation.
- Limits maximum steps to avoid infinite loops.
- Outputs results to both `.txt` and `.csv` files for easy analysis.
- Provides progress feedback and timing information.
- Allows repeated runs with different variations and ranges.

---

## Collatz Variations Included

| Variation Name      | Odd Rule         | Even Rule | Description            |
|---------------------|------------------|-----------|------------------------|
| Standard Collatz    | 3x + 1           | x / 2     | Classic Collatz rules  |
| Reduced Collatz     | (x + 1) / 2      | x / 2     | Accelerated odd steps  |
| Your Original       | x + 1            | x / 2     | Simple increment odd   |
| Accelerated Collatz | (3x + 1) / 2     | x / 2     | Faster convergence?    |
| 3x-1 Variant        | 3x - 1           | x / 2     | A variation on odd step|

---

## Requirements

- Python 3.7 or higher

---

## Usage

1. Clone or download the repository.

2. Run the script:

   ```bash
   python collatz_variations.py

## Variations

quantum_collatz/
├── core/                      # Core computation engines
│   ├── symbolic_engine.py     # Rule system with plugin support
│   ├── quantum_alu.py         # Quantum arithmetic implementations
│   ├── hybrid_controller.py   # Adaptive execution logic
│   └── sequence_analyzer.py   # Convergence metrics & analysis
├── api/                       # Web service layer
│   ├── fastapi_app.py         # REST/WebSocket interface
│   ├── schemas.py             # Pydantic models
│   └── client.py              # Python API client
├── plugins/                   # Extensibility system
│   ├── rules/                 # Custom sequence rules
│   ├── visualizations/        # Plotting alternatives
│   └── quantum_backends/      # Pennylane alternatives
├── sdk/                       # Developer tools
│   ├── client.py              # QuantumMathClient class
│   └── rule_registrar.py      # Custom rule injection API
├── gui/                       # PyQt interface
│   ├── main_window.py         # Primary application window
│   ├── widgets/               # Modular UI components
│   └── themes/                # Style system
├── telemetry/                 # Monitoring system
│   ├── dashboard.py           # Real-time metrics
│   └── exporters.py           # CSV/JSON logging
├── presets/                   # Preconfigured systems
│   ├── collatz_variants/      # 5x+1, 3x-1, etc.
│   └── quantum_optimized/     # Gate-efficient versions
├── tests/                     # Comprehensive test suite
└── utils/                     # Shared utilities
    ├── logging.py             # Structured logging
    └── config.py             # Settings management

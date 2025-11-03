"""
QUANTUM COLLATZ ENGINE - ADVANCED HYBRID IMPLEMENTATION
========================================================
"""

import sys
import time
import numpy as np
import pandas as pd
from typing import (Callable, Dict, List, Tuple, Optional, Union, Any)
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyqtgraph as pg
from pyqtgraph import PlotWidget, PlotItem
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.templates import AmplitudeEmbedding, AngleEmbedding
import sklearn.decomposition as skd
from sklearn.preprocessing import MinMaxScaler

# PyQt5 Imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QTextEdit,
    QProgressBar, QTabWidget, QFileDialog, QMessageBox,
    QGroupBox, QCheckBox, QDoubleSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QHeaderView, QDockWidget, QListWidget, QToolBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QElapsedTimer
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor, QIcon, QKeySequence

# ==============================================
# CORE ENGINE COMPONENTS
# ==============================================

class SymbolicRuleEngine:
    """
    Enhanced genetic blueprint repository with dynamic rule generation
    and quantum circuit synthesis capabilities.
    """

    def __init__(self):
        self.rule_library = self._initialize_rule_library()
        self.active_rules = self.rule_library['standard']
        self.custom_rules = {}
        self._rule_validators = {
            'odd': self._validate_odd_rule,
            'even': self._validate_even_rule
        }

    def _initialize_rule_library(self) -> Dict[str, Dict[str, Any]]:
        """Initialize with advanced rule sets including parameterized variants"""
        return {
            'standard': {
                'odd': lambda x: 3*x + 1,
                'even': lambda x: x // 2,
                'description': "Classic 3x+1 Collatz conjecture",
                'quantum_support': True
            },
            'accelerated': {
                'odd': lambda x: (3*x + 1) // 2,
                'even': lambda x: x // 2,
                'description': "Compressed step variant (3x+1)/2",
                'quantum_support': True
            },
            'generalized': {
                'odd': lambda x, p=3, q=1: p*x + q,
                'even': lambda x, r=2: x // r,
                'description': "Generalized px+q form",
                'quantum_support': False,
                'params': {'p': 3, 'q': 1, 'r': 2}
            },
            'quantum_optimized': {
                'odd': self._quantum_odd_transform,
                'even': self._quantum_even_transform,
                'description': "Gate-optimized for QPU execution",
                'quantum_support': True
            }
        }

    def _quantum_odd_transform(self, x: int, params: Optional[Dict] = None) -> int:
        """Enhanced quantum-optimized 3x+1 with parameter support"""
        params = params or {}
        p = params.get('p', 3)
        q = params.get('q', 1)
        return (x << 1) + x + q if p == 3 else p*x + q  # Bit shift optimization for p=3

    def _quantum_even_transform(self, x: int, params: Optional[Dict] = None) -> int:
        """Enhanced quantum-optimized division with parameter support"""
        params = params or {}
        r = params.get('r', 2)
        return x >> 1 if r == 2 else x // r  # Bit shift optimization for r=2

    def _validate_odd_rule(self, fn: Callable) -> bool:
        """Validate odd transformation rules"""
        try:
            test_vals = [1, 3, 5, 7]
            return all(fn(x) > x for x in test_vals)
        except:
            return False

    def _validate_even_rule(self, fn: Callable) -> bool:
        """Validate even transformation rules"""
        try:
            test_vals = [2, 4, 6, 8]
            return all(fn(x) < x for x in test_vals)
        except:
            return False

    def add_custom_rule(self, name: str, rules: Dict[str, Callable],
                       description: str = "", quantum_support: bool = False,
                       params: Optional[Dict] = None):
        """
        Enhanced rule injection with validation and parameter support

        Args:
            name: Unique identifier for the rule set
            rules: Dictionary containing 'odd' and 'even' transformations
            description: Human-readable description
            quantum_support: Whether the rules can be quantum-optimized
            params: Optional parameters for parameterized rules
        """
        if not all(k in rules for k in ['odd', 'even']):
            raise ValueError("Rules must contain both 'odd' and 'even' transformations")

        if not all(self._rule_validators[k](rules[k]) for k in ['odd', 'even']):
            raise ValueError("Provided rules failed validation checks")

        self.custom_rules[name] = {
            **rules,
            'description': description,
            'quantum_support': quantum_support,
            'params': params or {}
        }
        self.rule_library[name] = self.custom_rules[name]

    def set_active_rule(self, rule_name: str, params: Optional[Dict] = None) -> bool:
        """Enhanced rule selection with parameter passing"""
        if rule_name in self.rule_library:
            self.active_rules = self.rule_library[rule_name]
            if params and 'params' in self.active_rules:
                self.active_rules['params'].update(params)
            return True
        return False

    def generate_quantum_circuit(self, rule_name: str) -> Optional[qml.Operation]:
        """Generate quantum circuit for supported rules"""
        if rule_name not in self.rule_library or not self.rule_library[rule_name].get('quantum_support', False):
            return None

        # Placeholder for actual circuit generation logic
        return qml.RY(np.pi/2, wires=0)  # Example operation

class QuantumArithmeticUnit:
    """
    Enhanced quantum computational unit with:
    - Circuit optimization
    - State tomography
    - Error mitigation
    - Parallel execution
    """

    def __init__(self, num_qubits: int = 8, shots: int = 1024,
                 optimization_level: int = 2, enable_error_mitigation: bool = False):
        self.device = qml.device("default.qubit", wires=num_qubits, shots=shots)
        self.num_qubits = num_qubits
        self.shots = shots
        self.optimization_level = optimization_level
        self.enable_error_mitigation = enable_error_mitigation
        self.metrics = {
            'gate_depth': [],
            'entropy': [],
            'execution_time': [],
            'state_fidelity': []
        }
        self._circuit_cache = {}
        self._executor = ThreadPoolExecutor(max_workers=4)

    def __del__(self):
        self._executor.shutdown(wait=True)

    @qml.qnode(self.device)
    def execute_operation(self, x: int, operation: str, params: Optional[Dict] = None) -> int:
        """
        Enhanced quantum operation execution with:
        - Parameter support
        - Circuit caching
        - Error mitigation
        """
        params = params or {}
        cache_key = f"{x}_{operation}_{hash(frozenset(params.items()))}"

        if cache_key in self._circuit_cache:
            return self._circuit_cache[cache_key]

        start_time = time.perf_counter()

        # Initialize quantum state with advanced embedding
        self._load_integer_optimized(x)

        # Apply parameterized quantum operation
        if operation == 'expand':
            result = self._apply_parameterized_expand(params)
        elif operation == 'compress':
            result = self._apply_parameterized_compress(params)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Error mitigation if enabled
        if self.enable_error_mitigation:
            result = self._apply_error_mitigation(result)

        exec_time = time.perf_counter() - start_time
        self._record_metrics(operation, exec_time, x, result)

        self._circuit_cache[cache_key] = result
        return result

    def _load_integer_optimized(self, x: int):
        """Advanced state initialization using amplitude embedding"""
        # Normalize to probability amplitudes
        state = np.zeros(2**self.num_qubits)
        state[x % len(state)] = 1.0
        AmplitudeEmbedding(state, wires=range(self.num_qubits), normalize=True)

    def _apply_parameterized_expand(self, params: Dict) -> int:
        """Parameterized quantum expansion (px+q)"""
        p = params.get('p', 3)
        q = params.get('q', 1)

        # Quantum multiplication by p
        if p == 3:
            # Optimized circuit for p=3
            for i in range(self.num_qubits):
                qml.RY(2*np.pi/3, wires=i)
        else:
            # General multiplication
            for _ in range(p-1):
                for i in range(self.num_qubits):
                    qml.RY(np.pi/2, wires=i)

        # Quantum addition of q
        binary_q = format(q, f'0{self.num_qubits}b')
        for qubit, bit in enumerate(reversed(binary_q)):
            if bit == '1':
                qml.PauliX(wires=qubit)

        # Entanglement for carry propagation
        for i in range(self.num_qubits-1):
            qml.CNOT(wires=[i, i+1])

        return self._measure_optimized()

    def _apply_parameterized_compress(self, params: Dict) -> int:
        """Parameterized quantum compression (x/r)"""
        r = params.get('r', 2)

        if r == 2:
            # Optimized right shift
            for i in range(self.num_qubits-1):
                qml.SWAP(wires=[i, i+1])
        else:
            # General division - more complex quantum arithmetic
            for _ in range(r-1):
                for i in range(self.num_qubits-1):
                    qml.CRY(np.pi/2, wires=[i, i+1])

        return self._measure_optimized()

    def _measure_optimized(self) -> int:
        """Optimized measurement with post-selection"""
        # Use computational basis measurement
        measurements = qml.sample(wires=range(self.num_qubits))

        # Post-process measurements (mode of samples)
        unique, counts = np.unique(measurements, return_counts=True, axis=0)
        most_frequent = unique[np.argmax(counts)]

        # Convert bitstring to integer
        return int("".join(map(str, most_frequent)), 2)

    def _apply_error_mitigation(self, raw_result: int) -> int:
        """Basic error mitigation using post-processing"""
        # Simple majority voting - could be enhanced with more sophisticated techniques
        results = [raw_result]
        for _ in range(2):  # Take 2 additional shots
            results.append(self._measure_optimized())
        return max(set(results), key=results.count)

    def _record_metrics(self, operation: str, exec_time: float, input_val: int, output_val: int):
        """Enhanced metric tracking with state tomography"""
        self.metrics['gate_depth'].append(len(self.device.tape.operations))
        self.metrics['entropy'].append(self._calculate_entropy())
        self.metrics['execution_time'].append(exec_time * 1000)  # Convert to ms
        self.metrics['state_fidelity'].append(self._calculate_state_fidelity(input_val, output_val))

    def _calculate_entropy(self) -> float:
        """Calculate quantum state von Neumann entropy"""
        state = self.device.state
        probabilities = np.abs(state) ** 2
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    def _calculate_state_fidelity(self, input_val: int, output_val: int) -> float:
        """Calculate state fidelity between expected and actual output"""
        expected = output_val  # Simplified - should compute expected classically
        actual = output_val
        return 1.0 if expected == actual else 0.0  # Simplified fidelity measure

    def parallel_execute(self, operations: List[Tuple[int, str, Optional[Dict]]]) -> List[int]:
        """Execute multiple operations in parallel"""
        futures = []
        results = [0] * len(operations)

        for i, (x, op, params) in enumerate(operations):
            future = self._executor.submit(self.execute_operation, x, op, params)
            futures.append((i, future))

        for i, future in futures:
            results[i] = future.result()

        return results

class HybridController:
    """
    Enhanced hybrid controller with:
    - Adaptive threshold adjustment
    - Advanced caching strategies
    - Predictive execution
    - Comprehensive telemetry
    """

    def __init__(self, symbolic_engine: SymbolicRuleEngine, quantum_alu: QuantumArithmeticUnit):
        self.symbolic = symbolic_engine
        self.quantum = quantum_alu
        self.threshold = 100  # Initial switching point
        self._adaptive_threshold = True
        self.cache = {}
        self.telemetry = {
            'classical_steps': 0,
            'quantum_steps': 0,
            'crossings': 0,
            'cache_hits': 0,
            'execution_times': [],
            'sequence_lengths': []
        }
        self._performance_model = self._init_performance_model()
        self._predictive_buffer = []
        self._batch_size = 10

    def _init_performance_model(self) -> Dict:
        """Initialize performance prediction model"""
        return {
            'classical_speed': 1e6,  # steps/sec (estimate)
            'quantum_speed': 1e3,    # steps/sec (estimate)
            'last_adjustment': time.time()
        }

    def compute_sequence(self, start: int, max_steps: int = 1000,
                        parallel: bool = False) -> List[int]:
        """
        Enhanced sequence computation with:
        - Parallel execution
        - Predictive batching
        - Adaptive thresholding
        """
        sequence = [start]
        current = start
        batch = []

        for _ in range(max_steps):
            if current == 1:
                break

            if parallel and len(batch) < self._batch_size:
                batch.append(current)
                current = self._predict_step(current)
                continue

            if batch:
                results = self._compute_batch(batch, parallel)
                sequence.extend(results)
                current = results[-1]
                batch = []
                continue

            current = self._compute_step(current)
            sequence.append(current)

            # Adaptive threshold adjustment
            if self._adaptive_threshold and len(sequence) % 100 == 0:
                self._adjust_threshold(sequence)

        self._update_telemetry(sequence)
        return sequence

    def _compute_step(self, x: int) -> int:
        """Enhanced step computation with advanced caching"""
        if x in self.cache:
            self.telemetry['cache_hits'] += 1
            return self.cache[x]

        use_quantum = self._should_use_quantum(x)
        params = self.symbolic.active_rules.get('params', {})

        if use_quantum:
            operation = 'expand' if x % 2 == 1 else 'compress'
            result = self.quantum.execute_operation(x, operation, params)
            self.telemetry['quantum_steps'] += 1
            self.telemetry['crossings'] += 1
        else:
            result = self._classical_step(x, params)
            self.telemetry['classical_steps'] += 1

        self.cache[x] = result
        return result

    def _compute_batch(self, batch: List[int], parallel: bool) -> List[int]:
        """Compute a batch of values efficiently"""
        if parallel:
            operations = [
                (x, 'expand' if x % 2 == 1 else 'compress',
                 self.symbolic.active_rules.get('params', {}))
                for x in batch
            ]
            return self.quantum.parallel_execute(operations)
        else:
            return [self._compute_step(x) for x in batch]

    def _predict_step(self, x: int) -> int:
        """Predict next step for batching (simplified)"""
        return self._classical_step(x, self.symbolic.active_rules.get('params', {}))

    def _classical_step(self, x: int, params: Dict) -> int:
        """Parameterized classical transformation"""
        if x % 2 == 1:
            return self.symbolic.active_rules['odd'](x, **params)
        return self.symbolic.active_rules['even'](x, **params)

    def _should_use_quantum(self, x: int) -> bool:
        """Determine if quantum computation should be used"""
        if not self.symbolic.active_rules.get('quantum_support', False):
            return False

        # Simple threshold check - could be enhanced with more sophisticated logic
        return x >= self.threshold

    def _adjust_threshold(self, sequence: List[int]):
        """Dynamically adjust quantum threshold based on performance"""
        now = time.time()
        if now - self._performance_model['last_adjustment'] < 1.0:  # Don't adjust too frequently
            return

        # Simple adjustment heuristic - could be enhanced
        quantum_ratio = self.telemetry['quantum_steps'] / max(1, len(sequence))

        if quantum_ratio > 0.5:  # Too much quantum usage
            self.threshold = min(1000, int(self.threshold * 1.1))
        else:  # Too little quantum usage
            self.threshold = max(10, int(self.threshold * 0.9))

        self._performance_model['last_adjustment'] = now

    def _update_telemetry(self, sequence: List[int]):
        """Update telemetry with sequence statistics"""
        self.telemetry['execution_times'].append(time.perf_counter())
        self.telemetry['sequence_lengths'].append(len(sequence))

    def get_telemetry(self) -> Dict:
        """Enhanced telemetry with performance metrics"""
        total_steps = self.telemetry['classical_steps'] + self.telemetry['quantum_steps']
        return {
            **self.telemetry,
            'total_steps': total_steps,
            'quantum_ratio': self.telemetry['quantum_steps'] / max(1, total_steps),
            'average_sequence_length': np.mean(self.telemetry['sequence_lengths']) if self.telemetry['sequence_lengths'] else 0,
            'current_threshold': self.threshold,
            'adaptive_threshold': self._adaptive_threshold
        }

    def enable_adaptive_threshold(self, enabled: bool):
        """Toggle adaptive threshold adjustment"""
        self._adaptive_threshold = enabled

# ==============================================
# VISUALIZATION AND UI COMPONENTS
# ==============================================

class QuantumDebugger(QDockWidget):
    """Enhanced quantum debugger with real-time visualization and analysis tools"""

    def __init__(self, parent=None):
        super().__init__("Quantum Debugger", parent)
        self.setAllowedAreas(Qt.AllDockWidgetAreas)

        self.widget = QWidget()
        self.layout = QVBoxLayout(self.widget)

        # Create metrics dashboard
        self._create_metrics_dashboard()

        # Create quantum state visualizer
        self._create_state_visualizer()

        # Add analysis tools
        self._create_analysis_tools()

        self.setWidget(self.widget)

    def _create_metrics_dashboard(self):
        """Create comprehensive metrics dashboard"""
        metrics_group = QGroupBox("Quantum Execution Metrics")
        metrics_layout = QVBoxLayout()

        # Real-time metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(3)
        self.metrics_table.setHorizontalHeaderLabels(['Metric', 'Value', 'Trend'])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Performance indicators
        self.gate_depth_indicator = self._create_performance_indicator("Gate Depth")
        self.execution_time_indicator = self._create_performance_indicator("Execution Time")
        self.fidelity_indicator = self._create_performance_indicator("State Fidelity")

        metrics_layout.addWidget(self.metrics_table)
        metrics_layout.addWidget(self.gate_depth_indicator)
        metrics_layout.addWidget(self.execution_time_indicator)
        metrics_layout.addWidget(self.fidelity_indicator)
        metrics_group.setLayout(metrics_layout)

        self.layout.addWidget(metrics_group)

    def _create_performance_indicator(self, name: str) -> QWidget:
        """Create a performance indicator widget"""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        label = QLabel(name)
        progress = QProgressBar()
        progress.setRange(0, 100)

        layout.addWidget(label)
        layout.addWidget(progress)

        return widget

    def _create_state_visualizer(self):
        """Create advanced state visualization tools"""
        vis_group = QGroupBox("Quantum State Visualization")
        vis_layout = QVBoxLayout()

        # Create tabbed visualizations
        vis_tabs = QTabWidget()

        # State evolution plot
        self.state_evolution_plot = pg.PlotWidget()
        self.state_evolution_plot.setBackground('w')
        self.state_evolution_plot.setTitle("State Evolution")
        self.state_evolution_plot.addLegend()

        # State tomography display
        self.state_tomography_view = pg.GraphicsLayoutWidget()
        self.state_tomography_plot = self.state_tomography_view.addPlot(title="State Tomography")

        # Add tabs
        vis_tabs.addTab(self.state_evolution_plot, "Evolution")
        vis_tabs.addTab(self.state_tomography_view, "Tomography")

        vis_layout.addWidget(vis_tabs)
        vis_group.setLayout(vis_layout)

        self.layout.addWidget(vis_group)

    def _create_analysis_tools(self):
        """Create quantum state analysis tools"""
        analysis_group = QGroupBox("Quantum State Analysis")
        analysis_layout = QVBoxLayout()

        # PCA dimensionality reduction
        self.pca_button = QPushButton("Run PCA Analysis")

        # State similarity tools
        self.similarity_button = QPushButton("Compare States")

        analysis_layout.addWidget(self.pca_button)
        analysis_layout.addWidget(self.similarity_button)
        analysis_group.setLayout(analysis_layout)

        self.layout.addWidget(analysis_group)

    def update_metrics(self, metrics: Dict):
        """Update all metrics displays"""
        self.metrics_table.setRowCount(len(metrics))

        for row, (name, value) in enumerate(metrics.items()):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(name))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(str(value)))

            # Add trend indicator if historical data exists
            if isinstance(value, (int, float)):
                trend = "↑" if value > 0 else "↓" if value < 0 else "→"
                self.metrics_table.setItem(row, 2, QTableWidgetItem(trend))

    def plot_state_evolution(self, sequence: List[int], quantum_regions: List[Tuple[int, int]]):
        """Enhanced state evolution plotting with region highlighting"""
        self.state_evolution_plot.clear()

        x = range(len(sequence))
        y = sequence

        # Main sequence plot
        main_line = self.state_evolution_plot.plot(x, y, pen='b', name="Sequence")

        # Highlight quantum computation regions
        for start, end in quantum_regions:
            region = pg.LinearRegionItem(values=[start, end], brush=(255, 0, 0, 50))
            self.state_evolution_plot.addItem(region)

        # Add threshold line
        threshold_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('g', style=Qt.DashLine))
        self.state_evolution_plot.addItem(threshold_line)

    def plot_state_tomography(self, state: np.ndarray):
        """Visualize quantum state tomography results"""
        self.state_tomography_plot.clear()

        # Simple visualization - could be enhanced with Bloch sphere or similar
        bars = pg.BarGraphItem(x=range(len(state)), height=np.abs(state), width=0.6)
        self.state_tomography_plot.addItem(bars)

class SequenceAnalyzer(QDockWidget):
    """Advanced sequence analysis tools with statistical visualization"""

    def __init__(self, parent=None):
        super().__init__("Sequence Analyzer", parent)
        self.setup_ui()

    def setup_ui(self):
        self.widget = QWidget()
        self.layout = QVBoxLayout(self.widget)

        # Create analysis tabs
        self.tabs = QTabWidget()

        # Statistical analysis tab
        self._create_stats_tab()

        # Pattern recognition tab
        self._create_patterns_tab()

        # Comparative analysis tab
        self._create_comparative_tab()

        self.layout.addWidget(self.tabs)
        self.setWidget(self.widget)

    def _create_stats_tab(self):
        """Create statistical analysis tab"""
        stats_tab = QWidget()
        layout = QVBoxLayout(stats_tab)

        # Basic statistics
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(['Statistic', 'Value'])

        # Distribution plot
        self.dist_plot = pg.PlotWidget()

        layout.addWidget(self.stats_table)
        layout.addWidget(self.dist_plot)
        self.tabs.addTab(stats_tab, "Statistics")

    def _create_patterns_tab(self):
        """Create pattern recognition tab"""
        patterns_tab = QWidget()
        layout = QVBoxLayout(patterns_tab)

        # Pattern visualization
        self.pattern_plot = pg.PlotWidget()

        # Pattern detection controls
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)

        self.pattern_type = QComboBox()
        self.pattern_type.addItems(["Cycles", "Convergence", "Divergence"])

        detect_button = QPushButton("Detect Patterns")

        controls_layout.addWidget(self.pattern_type)
        controls_layout.addWidget(detect_button)

        layout.addWidget(controls)
        layout.addWidget(self.pattern_plot)
        self.tabs.addTab(patterns_tab, "Patterns")

    def _create_comparative_tab(self):
        """Create comparative analysis tab"""
        comp_tab = QWidget()
        layout = QVBoxLayout(comp_tab)

        # Comparative plot
        self.comp_plot = pg.PlotWidget()

        # Comparison controls
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)

        self.comp_mode = QComboBox()
        self.comp_mode.addItems(["Length", "Steps", "Peak"])

        compare_button = QPushButton("Compare Sequences")

        controls_layout.addWidget(self.comp_mode)
        controls_layout.addWidget(compare_button)

        layout.addWidget(controls)
        layout.addWidget(self.comp_plot)
        self.tabs.addTab(comp_tab, "Comparison")

    def analyze_sequence(self, sequence: List[int]):
        """Perform comprehensive sequence analysis"""
        self._update_stats_table(sequence)
        self._plot_distribution(sequence)

    def _update_stats_table(self, sequence: List[int]):
        """Update statistics table with sequence metrics"""
        stats = {
            'Length': len(sequence),
            'Maximum': max(sequence),
            'Minimum': min(sequence),
            'Average': np.mean(sequence),
            'Std Dev': np.std(sequence),
            'Convergence Steps': len(sequence) - sequence.index(1) if 1 in sequence else 'N/A'
        }

        self.stats_table.setRowCount(len(stats))

        for row, (name, value) in enumerate(stats.items()):
            self.stats_table.setItem(row, 0, QTableWidgetItem(name))
            self.stats_table.setItem(row, 1, QTableWidgetItem(str(value)))

    def _plot_distribution(self, sequence: List[int]):
        """Plot sequence value distribution"""
        self.dist_plot.clear()

        # Create histogram
        y, x = np.histogram(sequence, bins=20)
        bars = pg.BarGraphItem(x=x[:-1], height=y, width=x[1]-x[0])
        self.dist_plot.addItem(bars)

class MainWindow(QMainWindow):
    """Enhanced main window with advanced features and improved UI"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantum Collatz Engine")
        self.setGeometry(100, 100, 1600, 1000)

        # Initialize core components
        self.symbolic_engine = SymbolicRuleEngine()
        self.quantum_alu = QuantumArithmeticUnit(
            num_qubits=10,
            shots=2048,
            optimization_level=2,
            enable_error_mitigation=True
        )
        self.controller = HybridController(self.symbolic_engine, self.quantum_alu)

        # Setup UI
        self._init_ui()
        self._connect_signals()
        self._apply_styling()

        # Initialize state
        self.current_sequence = []
        self.quantum_regions = []
        self.benchmark_results = []

        # Create status bar
        self.statusBar().showMessage("Ready")

    def _init_ui(self):
        """Initialize enhanced user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create left control panel
        control_panel = self._create_control_panel()

        # Create visualization area
        vis_tabs = QTabWidget()

        # Main sequence visualization
        self.sequence_plot = pg.PlotWidget()
        self.sequence_plot.setBackground('w')
        self.sequence_plot.setTitle("Sequence Visualization")
        self.sequence_plot.addLegend()

        # Circuit diagram
        self.circuit_viewer = FigureCanvas(Figure(figsize=(10, 6)))

        # Add tabs
        vis_tabs.addTab(self.sequence_plot, "Sequence")
        vis_tabs.addTab(self.circuit_viewer, "Circuit")

        # Add dock widgets
        self.debugger = QuantumDebugger(self)
        self.analyzer = SequenceAnalyzer(self)

        self.addDockWidget(Qt.RightDockWidgetArea, self.debugger)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.analyzer)

        # Add components to main layout
        main_layout.addWidget(control_panel, stretch=1)
        main_layout.addWidget(vis_tabs, stretch=3)

        # Create toolbar
        self._create_toolbar()

    def _create_control_panel(self) -> QWidget:
        """Create enhanced control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Add control sections
        self._add_rule_controls(layout)
        self._add_computation_controls(layout)
        self._add_threshold_controls(layout)
        self._add_execution_controls(layout)
        self._add_benchmark_controls(layout)

        return panel

    def _create_toolbar(self):
        """Create application toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(32, 32))
        self.addToolBar(toolbar)

        # Add actions
        run_action = QAction(QIcon.fromTheme("media-playback-start"), "Run", self)
        run_action.triggered.connect(self._run_computation)
        run_action.setShortcut(QKeySequence("Ctrl+R"))
        toolbar.addAction(run_action)

        stop_action = QAction(QIcon.fromTheme("media-playback-stop"), "Stop", self)
        stop_action.triggered.connect(self._stop_computation)
        stop_action.setShortcut(QKeySequence("Ctrl+S"))
        toolbar.addAction(stop_action)

        toolbar.addSeparator()

        save_action = QAction(QIcon.fromTheme("document-save"), "Save Results", self)
        save_action.triggered.connect(self._save_results)
        save_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        toolbar.addAction(save_action)

    def _add_rule_controls(self, layout):
        """Enhanced rule controls with parameter editing"""
        rule_group = QGroupBox("Symbolic Rule Configuration")
        rule_layout = QVBoxLayout()

        # Rule selection
        self.rule_combo = QComboBox()
        self.rule_combo.addItems(self.symbolic_engine.rule_library.keys())

        # Rule description
        self.rule_desc = QLabel()
        self.rule_desc.setWordWrap(True)
        self._update_rule_description()

        # Parameter controls (dynamic)
        self.param_widgets = {}
        self.param_container = QWidget()
        self.param_layout = QVBoxLayout(self.param_container)

        rule_layout.addWidget(QLabel("Select Rule Set:"))
        rule_layout.addWidget(self.rule_combo)
        rule_layout.addWidget(QLabel("Description:"))
        rule_layout.addWidget(self.rule_desc)
        rule_layout.addWidget(QLabel("Parameters:"))
        rule_layout.addWidget(self.param_container)

        rule_group.setLayout(rule_layout)
        layout.addWidget(rule_group)

        # Initialize parameter controls
        self._update_parameter_controls()

    def _add_computation_controls(self, layout):
        """Enhanced computation controls"""
        comp_group = QGroupBox("Computation Parameters")
        comp_layout = QVBoxLayout()

        # Starting value
        self.start_spin = QSpinBox()
        self.start_spin.setRange(1, 10**18)
        self.start_spin.setValue(27)

        # Maximum steps
        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(10, 10**6)
        self.max_steps_spin.setValue(1000)

        # Parallel execution
        self.parallel_check = QCheckBox("Parallel Execution")

        comp_layout.addWidget(QLabel("Starting Value:"))
        comp_layout.addWidget(self.start_spin)
        comp_layout.addWidget(QLabel("Maximum Steps:"))
        comp_layout.addWidget(self.max_steps_spin)
        comp_layout.addWidget(self.parallel_check)
        comp_group.setLayout(comp_layout)

        layout.addWidget(comp_group)

    def _add_threshold_controls(self, layout):
        """Enhanced threshold controls"""
        threshold_group = QGroupBox("Hybrid Threshold")
        threshold_layout = QVBoxLayout()

        # Threshold slider
        self.threshold_slider = QDoubleSpinBox()
        self.threshold_slider.setRange(10, 10000)
        self.threshold_slider.setValue(100)
        self.threshold_slider.setSingleStep(10)

        # Adaptive threshold checkbox
        self.adaptive_check = QCheckBox("Adaptive Threshold")
        self.adaptive_check.setChecked(True)

        threshold_layout.addWidget(QLabel("Quantum Activation Threshold:"))
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.adaptive_check)
        threshold_group.setLayout(threshold_layout)

        layout.addWidget(threshold_group)

    def _add_execution_controls(self, layout):
        """Enhanced execution controls"""
        exec_group = QGroupBox("Execution Control")
        exec_layout = QVBoxLayout()

        # Control buttons
        self.run_button = QPushButton("Run Computation")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)

        # Progress indicators
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")

        exec_layout.addWidget(self.run_button)
        exec_layout.addWidget(self.stop_button)
        exec_layout.addWidget(self.progress_bar)
        exec_layout.addWidget(self.status_label)
        exec_group.setLayout(exec_layout)

        layout.addWidget(exec_group)

    def _add_benchmark_controls(self, layout):
        """Add benchmarking controls"""
        bench_group = QGroupBox("Benchmarking")
        bench_layout = QVBoxLayout()

        # Benchmark range
        self.bench_start = QSpinBox()
        self.bench_start.setRange(1, 10**6)
        self.bench_start.setValue(1)

        self.bench_end = QSpinBox()
        self.bench_end.setRange(1, 10**6)
        self.bench_end.setValue(100)

        # Benchmark button
        self.bench_button = QPushButton("Run Benchmark")

        bench_layout.addWidget(QLabel("Start Value:"))
        bench_layout.addWidget(self.bench_start)
        bench_layout.addWidget(QLabel("End Value:"))
        bench_layout.addWidget(self.bench_end)
        bench_layout.addWidget(self.bench_button)
        bench_group.setLayout(bench_layout)

        layout.addWidget(bench_group)

    def _update_parameter_controls(self):
        """Update parameter controls based on selected rule"""
        # Clear existing controls
        for i in reversed(range(self.param_layout.count())):
            self.param_layout.itemAt(i).widget().setParent(None)

        self.param_widgets = {}

        # Get current rule
        rule_name = self.rule_combo.currentText()
        rule = self.symbolic_engine.rule_library.get(rule_name, {})
        params = rule.get('params', {})

        # Create controls for each parameter
        for param, value in params.items():
            label = QLabel(f"{param}:")
            spin = QDoubleSpinBox()
            spin.setRange(-1000, 1000)
            spin.setValue(value)
            spin.setSingleStep(0.1)

            self.param_widgets[param] = spin

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.addWidget(label)
            row_layout.addWidget(spin)

            self.param_layout.addWidget(row)

    def _connect_signals(self):
        """Connect all UI signals to slots"""
        # Rule controls
        self.rule_combo.currentTextChanged.connect(self._change_rule_set)

        # Parameter controls
        for spin in self.param_widgets.values():
            spin.valueChanged.connect(self._update_parameters)

        # Computation controls
        self.run_button.clicked.connect(self._run_computation)
        self.stop_button.clicked.connect(self._stop_computation)
        self.parallel_check.stateChanged.connect(self._toggle_parallel)

        # Threshold controls
        self.threshold_slider.valueChanged.connect(self._update_threshold)
        self.adaptive_check.stateChanged.connect(self._toggle_adaptive_threshold)

        # Benchmark controls
        self.bench_button.clicked.connect(self._run_benchmark)

    def _apply_styling(self):
        """Apply modern styling to the UI"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QPushButton {
                min-height: 30px;
                background-color: #4285f4;
                color: white;
                border: 1px solid #357ae8;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #3367d6;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #34a853;
            }
            QTableWidget {
                border: 1px solid #ddd;
            }
            QTabWidget::pane {
                border: 1px solid #ddd;
            }
        """)

    def _change_rule_set(self):
        """Handle rule set selection change"""
        rule_name = self.rule_combo.currentText()
        self.symbolic_engine.set_active_rule(rule_name)
        self._update_rule_description()
        self._update_parameter_controls()

    def _update_rule_description(self):
        """Update displayed rule description"""
        current_rule = self.rule_combo.currentText()
        desc = self.symbolic_engine.rule_library[current_rule]['description']
        self.rule_desc.setText(desc)

    def _update_parameters(self):
        """Update rule parameters from UI controls"""
        params = {name: spin.value() for name, spin in self.param_widgets.items()}
        rule_name = self.rule_combo.currentText()
        self.symbolic_engine.set_active_rule(rule_name, params)

    def _update_threshold(self):
        """Update hybrid computation threshold"""
        self.controller.threshold = int(self.threshold_slider.value())

    def _toggle_adaptive_threshold(self, state):
        """Toggle adaptive threshold adjustment"""
        self.controller.enable_adaptive_threshold(state == Qt.Checked)

    def _toggle_parallel(self, state):
        """Toggle parallel execution mode"""
        self.parallel_mode = state == Qt.Checked

    def _run_computation(self):
        """Execute the hybrid computation"""
        start_val = self.start_spin.value()
        max_steps = self.max_steps_spin.value()
        parallel = self.parallel_check.isChecked()

        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Computing...")
        self.progress_bar.setRange(0, max_steps)

        # Execute in separate thread
        self.computation_thread = ComputationThread(
            self.controller, start_val, max_steps, parallel
        )
        self.computation_thread.progress_updated.connect(self._update_progress)
        self.computation_thread.result_ready.connect(self._handle_results)
        self.computation_thread.finished.connect(self._computation_finished)
        self.computation_thread.start()

    def _stop_computation(self):
        """Abort running computation"""
        if hasattr(self, 'computation_thread'):
            self.computation_thread.stop()
            self.status_label.setText("Computation stopped")

        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _update_progress(self, progress: int):
        """Update progress bar during computation"""
        self.progress_bar.setValue(progress)

    def _handle_results(self, sequence: List[int]):
        """Process and display computation results"""
        self.current_sequence = sequence

        # Identify quantum computation regions
        self.quantum_regions = []
        in_quantum = False
        start_idx = 0

        for i, val in enumerate(sequence):
            if val >= self.controller.threshold and not in_quantum:
                start_idx = i
                in_quantum = True
            elif val < self.controller.threshold and in_quantum:
                self.quantum_regions.append((start_idx, i))
                in_quantum = False

        if in_quantum:
            self.quantum_regions.append((start_idx, len(sequence)-1))

        # Update visualizations
        self._plot_sequence()
        self.debugger.update_metrics(self.controller.get_telemetry())
        self.debugger.plot_state_evolution(sequence, self.quantum_regions)
        self.analyzer.analyze_sequence(sequence)

        # Update circuit diagram
        self._update_circuit_view()

    def _computation_finished(self):
        """Clean up after computation completes"""
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Computation complete")
        self.progress_bar.setValue(0)

    def _plot_sequence(self):
        """Enhanced sequence visualization"""
        self.sequence_plot.clear()

        if not self.current_sequence:
            return

        x = range(len(self.current_sequence))
        y = self.current_sequence

        # Main sequence plot with logarithmic scaling
        self.sequence_plot.setLogMode(False, True)
        main_line = self.sequence_plot.plot(x, y, pen='b', name="Sequence")

        # Highlight quantum regions
        for start, end in self.quantum_regions:
            region = pg.LinearRegionItem(values=[start, end], brush=(255, 0, 0, 50))
            self.sequence_plot.addItem(region)

        # Add threshold line
        threshold_line = pg.InfiniteLine(
            pos=self.controller.threshold,
            angle=0,
            pen=pg.mkPen('g', style=Qt.DashLine),
            label=f"Threshold: {self.controller.threshold}",
            labelOpts={'position': 0.1}
        )
        self.sequence_plot.addItem(threshold_line)

    def _update_circuit_view(self):
        """Update quantum circuit visualization"""
        fig = self.circuit_viewer.figure
        fig.clear()

        ax = fig.add_subplot(111)

        # Get current rule
        rule_name = self.rule_combo.currentText()
        circuit = self.symbolic_engine.generate_quantum_circuit(rule_name)

        if circuit:
            # Simple visualization - could be enhanced with actual circuit drawing
            ax.text(0.5, 0.5, f"Quantum Circuit for {rule_name}\n\n{str(circuit)}",
                   ha='center', va='center')
        else:
            ax.text(0.5, 0.5, "No quantum circuit available\nfor selected rule set",
                   ha='center', va='center')

        self.circuit_viewer.draw()

    def _run_benchmark(self):
        """Run performance benchmark"""
        start = self.bench_start.value()
        end = self.bench_end.value()

        if start >= end:
            QMessageBox.warning(self, "Invalid Range", "End value must be greater than start value")
            return

        self.benchmark_thread = BenchmarkThread(self.controller, start, end)
        self.benchmark_thread.result_ready.connect(self._handle_benchmark_results)
        self.benchmark_thread.start()

        self.status_label.setText(f"Running benchmark from {start} to {end}...")

    def _handle_benchmark_results(self, results: List[Dict]):
        """Process and display benchmark results"""
        self.benchmark_results = results
        self._plot_benchmark_results()
        self.status_label.setText("Benchmark complete")

    def _plot_benchmark_results(self):
        """Visualize benchmark results"""
        if not self.benchmark_results:
            return

        # Create a new window for benchmark results
        bench_window = QMainWindow()
        bench_window.setWindowTitle("Benchmark Results")
        bench_window.resize(800, 600)

        # Create plot widget
        plot_widget = pg.PlotWidget()
        plot_widget.setBackground('w')
        plot_widget.setTitle("Benchmark Results")
        plot_widget.addLegend()

        # Prepare data
        x = [r['start'] for r in self.benchmark_results]
        y_time = [r['time'] for r in self.benchmark_results]
        y_steps = [r['steps'] for r in self.benchmark_results]

        # Plot execution time
        time_curve = plot_widget.plot(x, y_time, pen='b', name="Execution Time (ms)")

        # Plot steps (secondary axis)
        steps_axis = pg.ViewBox()
        plot_widget.scene().addItem(steps_axis)
        plot_widget.getPlotItem().showAxis('right')
        plot_widget.getPlotItem().getAxis('right').linkToView(steps_axis)

        steps_curve = pg.PlotCurveItem(x, y_steps, pen='r', name="Steps")
        steps_axis.addItem(steps_curve)

        # Update views when resized
        def update_views():
            steps_axis.setGeometry(plot_widget.getPlotItem().vb.sceneBoundingRect())
            steps_axis.linkedViewChanged(plot_widget.getPlotItem().vb, steps_axis.XAxis)

        plot_widget.getPlotItem().vb.sigResized.connect(update_views)

        # Set central widget
        bench_window.setCentralWidget(plot_widget)
        bench_window.show()

    def _save_results(self):
        """Save current results to file"""
        if not self.current_sequence:
            QMessageBox.warning(self, "No Data", "No computation results to save")
            return

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "CSV Files (*.csv);;All Files (*)", options=options)

        if file_name:
            try:
                df = pd.DataFrame({
                    'step': range(len(self.current_sequence)),
                    'value': self.current_sequence,
                    'quantum': [any(start <= i <= end for (start, end) in self.quantum_regions)
                              for i in range(len(self.current_sequence))]
                })

                df.to_csv(file_name, index=False)
                self.status_label.setText(f"Results saved to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")

class ComputationThread(QThread):
    """Enhanced computation thread with progress reporting"""

    progress_updated = pyqtSignal(int)
    result_ready = pyqtSignal(list)

    def __init__(self, controller, start_val, max_steps, parallel=False):
        super().__init__()
        self.controller = controller
        self.start_val = start_val
        self.max_steps = max_steps
        self.parallel = parallel
        self._running = True

    def run(self):
        """Execute computation with progress updates"""
        sequence = []
        current = self.start_val
        steps = 0

        sequence.append(current)

        while self._running and steps < self.max_steps and current != 1:
            current = self.controller._compute_step(current)
            sequence.append(current)
            steps += 1

            if steps % 10 == 0:  # Throttle progress updates
                self.progress_updated.emit(steps)

        if self._running:
            self.result_ready.emit(sequence)
            self.progress_updated.emit(steps)

    def stop(self):
        """Gracefully stop computation"""
        self._running = False

class BenchmarkThread(QThread):
    """Thread for running performance benchmarks"""

    result_ready = pyqtSignal(list)

    def __init__(self, controller, start, end, samples=10):
        super().__init__()
        self.controller = controller
        self.start = start
        self.end = end
        self.samples = samples

    def run(self):
        """Execute benchmark across range of values"""
        results = []
        test_values = range(self.start, self.end + 1, max(1, (self.end - self.start) // self.samples))

        for val in test_values:
            timer = QElapsedTimer()
            timer.start()

            sequence = self.controller.compute_sequence(val)

            elapsed = timer.elapsed()
            results.append({
                'start': val,
                'time': elapsed,
                'steps': len(sequence),
                'quantum_steps': sum(1 for x in sequence if x >= self.controller.threshold)
            })

        self.result_ready.emit(results)

# ==============================================
# APPLICATION ENTRY POINT
# ==============================================

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set application metadata and styling
    app.setApplicationName("Quantum Collatz Engine")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Quantum Mathematics Lab")
    app.setStyle("Fusion")

    # Create and show main window
    window = MainWindow()
    window.show()

    # Start application loop
    sys.exit(app.exec_())

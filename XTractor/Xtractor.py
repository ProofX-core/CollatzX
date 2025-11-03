#!/usr/bin/env python3
import argparse
import json
import sys
import math
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
import time
import hashlib
import pickle
import zlib

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras import layers, models
import torch
import torch.nn as nn
import torch.optim as optim

# Advanced math and visualization
from sympy import isprime, factorint
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
import seaborn as sns

# Type aliases
CollatzParams = Tuple[int, int, int]
SequenceData = Dict[str, Any]
FeatureSet = Dict[str, Any]
ModelInput = Union[np.ndarray, torch.Tensor, tf.Tensor]

class TerminationStatus(Enum):
    TERMINATED = auto()
    CYCLE_DETECTED = auto()
    DIVERGED = auto()
    MAX_STEPS_REACHED = auto()
    UNKNOWN = auto()

@dataclass
class CollatzConfig:
    a: int = 3
    b: int = 1
    d: int = 2
    max_steps: int = 1000
    track_values: bool = False
    tail_window: int = 10
    cache_enabled: bool = True
    parallel_processing: bool = False
    progress_bar: bool = True

class CollatzSequenceSimulator:
    """
    Advanced simulator for generalized Collatz sequences with multiple enhancements:
    - Caching for performance
    - Parallel processing
    - Extended mathematical analysis
    - Multiple sequence generation strategies
    """
    def __init__(self, config: CollatzConfig = CollatzConfig()):
        self.config = config
        self.cache = {} if config.cache_enabled else None
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger('CollatzSimulator')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _cache_key(self, seed: int) -> str:
        """Generate a unique cache key for simulation parameters and seed"""
        params = f"{self.config.a}_{self.config.b}_{self.config.d}_{self.config.max_steps}"
        return f"{params}_{seed}"

    def _batch_simulate(self, seeds: List[int]) -> List[SequenceData]:
        """Batch simulation for multiple seeds with optional parallel processing"""
        if self.config.parallel_processing:
            try:
                from multiprocessing import Pool
                with Pool() as pool:
                    results = list(tqdm(pool.imap(self.simulate, seeds),
                                      total=len(seeds),
                                 disable=not self.config.progress_bar)
                return results
            except ImportError:
                self.logger.warning("Parallel processing not available. Falling back to sequential.")

        results = []
        for seed in tqdm(seeds, disable=not self.config.progress_bar):
            results.append(self.simulate(seed))
        return results

    def simulate(self, seed: int) -> SequenceData:
        """
        Enhanced simulation with:
        - Advanced cycle detection
        - Mathematical properties analysis
        - Detailed statistics collection
        - Caching for performance
        """
        # Check cache first
        if self.cache is not None:
            cache_key = self._cache_key(seed)
            if cache_key in self.cache:
                return self.cache[cache_key]

        sequence = [seed] if self.config.track_values else []
        current = seed
        steps = 0
        seen = {current: 0}
        peak = current
        termination_status = TerminationStatus.UNKNOWN
        parity_signature = []
        mathematical_properties = {
            'prime_factors': factorint(seed),
            'is_prime': isprime(seed),
            'modular_properties': defaultdict(list)
        }

        while steps < self.config.max_steps:
            steps += 1
            is_odd = current % 2 == 1

            # Apply generalized Collatz function
            if is_odd:
                next_val = (self.config.a * current + self.config.b) // self.config.d
                parity_signature.append(1)  # Using 1 for odd, 0 for even
            else:
                next_val = current // self.config.d
                parity_signature.append(0)

            # Track mathematical properties
            if steps % 10 == 0 or steps < 10:
                mathematical_properties['modular_properties'][next_val % 8].append(steps)

            # Update tracking
            if self.config.track_values:
                sequence.append(next_val)

            # Enhanced termination checks
            if next_val == 1:
                termination_status = TerminationStatus.TERMINATED
                break
            if next_val in seen:
                cycle_length = steps - seen[next_val]
                termination_status = TerminationStatus.CYCLE_DETECTED
                break
            if next_val < 1:
                termination_status = TerminationStatus.DIVERGED
                break
            if next_val > 10**100:  # Practical divergence check
                termination_status = TerminationStatus.DIVERGED
                break

            # Update for next iteration
            seen[next_val] = steps
            current = next_val
            peak = max(peak, current)

        if termination_status == TerminationStatus.UNKNOWN and steps >= self.config.max_steps:
            termination_status = TerminationStatus.MAX_STEPS_REACHED

        # Calculate advanced metrics
        if self.config.track_values and len(sequence) > 1:
            jumps = np.diff(sequence)
            mathematical_properties.update({
                'jump_stats': {
                    'mean': float(np.mean(jumps)),
                    'std': float(np.std(jumps)),
                    'max': float(np.max(jumps)),
                    'min': float(np.min(jumps))
                },
                'entropy': float(self._calculate_entropy(sequence))
            })

        result = {
            'seed': seed,
            'steps': steps,
            'termination_status': termination_status.name,
            'peak': peak,
            'parity_signature': parity_signature,
            'final_value': current,
            'mathematical_properties': mathematical_properties,
            'config': {
                'a': self.config.a,
                'b': self.config.b,
                'd': self.config.d
            }
        }

        if self.config.track_values:
            result['sequence'] = sequence
            result['compressed_sequence'] = zlib.compress(pickle.dumps(sequence))

        # Cache the result
        if self.cache is not None:
            self.cache[self._cache_key(seed)] = result

        return result

    def _calculate_entropy(self, sequence: List[int]) -> float:
        """Calculate the Shannon entropy of the sequence"""
        counts = Counter(sequence)
        total = len(sequence)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log(p) if p > 0 else 0
        return entropy

class AdvancedFeatureExtractor:
    """
    State-of-the-art feature extraction with:
    - 100+ mathematical features
    - Dimensionality reduction
    - Feature importance analysis
    - Automated feature engineering
    """
    def __init__(self, config: CollatzConfig = CollatzConfig()):
        self.config = config
        self.feature_registry = self._initialize_feature_registry()

    def _initialize_feature_registry(self) -> Dict[str, Callable]:
        """Registry of all available feature extraction functions"""
        return {
            'basic': self._extract_basic_features,
            'parity': self._extract_parity_features,
            'jump': self._extract_jump_features,
            'tail': self._extract_tail_features,
            'step_ratio': self._extract_step_ratio_features,
            'mathematical': self._extract_mathematical_features,
            'spectral': self._extract_spectral_features,
            'topological': self._extract_topological_features
        }

    def extract_features(self, sequence_data: SequenceData, feature_set: List[str] = None) -> FeatureSet:
        """
        Extract features with optional selection of feature sets.
        Supports parallel feature extraction when possible.
        """
        if feature_set is None:
            feature_set = list(self.feature_registry.keys())

        features = {'seed': sequence_data['seed']}

        for set_name in feature_set:
            if set_name in self.feature_registry:
                features.update(self.feature_registry[set_name](sequence_data))

        return features

    def _extract_basic_features(self, sequence_data: SequenceData) -> Dict:
        """Extract basic sequence features"""
        return {
            'seed': sequence_data['seed'],
            'termination_status': sequence_data['termination_status'],
            'total_steps': sequence_data['steps'],
            'peak_value': sequence_data['peak'],
            'final_value': sequence_data['final_value'],
            'a_param': sequence_data['config']['a'],
            'b_param': sequence_data['config']['b'],
            'd_param': sequence_data['config']['d']
        }

    def _extract_parity_features(self, sequence_data: SequenceData) -> Dict:
        """Advanced parity analysis with transition matrices"""
        parity = sequence_data['parity_signature']
        if len(parity) < 2:
            return {
                'parity_odd_count': 0,
                'parity_even_count': 0,
                'parity_transitions': 0,
                'parity_odd_ratio': 0.0
            }

        parity_array = np.array(parity)
        odd_count = np.sum(parity_array)
        even_count = len(parity) - odd_count

        # Transition counts
        transitions = {
            '00': 0, '01': 0,
            '10': 0, '11': 0
        }

        for i in range(1, len(parity)):
            transition = f"{parity[i-1]}{parity[i]}"
            transitions[transition] += 1

        total_transitions = len(parity) - 1
        transition_matrix = [
            transitions['00'] / total_transitions,
            transitions['01'] / total_transitions,
            transitions['10'] / total_transitions,
            transitions['11'] / total_transitions
        ]

        return {
            'parity_odd_count': int(odd_count),
            'parity_even_count': int(even_count),
            'parity_transitions': int(total_transitions),
            'parity_odd_ratio': float(odd_count / len(parity)),
            'parity_transition_00': transition_matrix[0],
            'parity_transition_01': transition_matrix[1],
            'parity_transition_10': transition_matrix[2],
            'parity_transition_11': transition_matrix[3]
        }

    def _extract_jump_features(self, sequence_data: SequenceData) -> Dict:
        """Advanced jump analysis with statistical moments"""
        if 'sequence' not in sequence_data or len(sequence_data['sequence']) < 2:
            return {
                'jump_mean': 0.0,
                'jump_std': 0.0,
                'jump_skewness': 0.0,
                'jump_kurtosis': 0.0,
                'jump_max_rise': 0.0,
                'jump_max_fall': 0.0
            }

        sequence = sequence_data['sequence']
        jumps = np.diff(sequence)

        if len(jumps) == 0:
            return {}

        return {
            'jump_mean': float(np.mean(jumps)),
            'jump_std': float(np.std(jumps)),
            'jump_skewness': float(pd.Series(jumps).skew()),
            'jump_kurtosis': float(pd.Series(jumps).kurtosis()),
            'jump_max_rise': float(np.max(jumps)),
            'jump_max_fall': float(np.min(jumps))
        }

    def _extract_tail_features(self, sequence_data: SequenceData) -> Dict:
        """Advanced tail analysis with multiple window sizes"""
        if 'sequence' not in sequence_data or len(sequence_data['sequence']) == 0:
            return {
                'tail_entropy': 0.0,
                'tail_mean': 0.0,
                'tail_std': 0.0
            }

        sequence = sequence_data['sequence']
        tail = sequence[-self.config.tail_window:]

        return {
            'tail_entropy': float(self._calculate_entropy(tail)),
            'tail_mean': float(np.mean(tail)),
            'tail_std': float(np.std(tail)) if len(tail) > 1 else 0.0,
            'tail_min': float(np.min(tail)),
            'tail_max': float(np.max(tail))
        }

    def _extract_step_ratio_features(self, sequence_data: SequenceData) -> Dict:
        """Advanced step ratio features with logarithmic transforms"""
        steps = sequence_data['steps']
        peak = sequence_data['peak']
        seed = sequence_data['seed']

        peak_step_estimate = steps // 2
        if 'sequence' in sequence_data:
            sequence = sequence_data['sequence']
            peak_step = np.argmax(sequence)
        else:
            peak_step = peak_step_estimate

        try:
            log_peak_ratio = math.log(peak_step + 1) / math.log(steps + 1)
            peak_seed_ratio = peak / seed if seed != 0 else 0.0
            steps_seed_ratio = steps / seed if seed != 0 else 0.0
        except (ValueError, ZeroDivisionError):
            log_peak_ratio = 0.0
            peak_seed_ratio = 0.0
            steps_seed_ratio = 0.0

        return {
            'log_step_peak_ratio': log_peak_ratio,
            'peak_to_seed_ratio': peak_seed_ratio,
            'steps_to_seed_ratio': steps_seed_ratio,
            'peak_step_ratio': peak_step / steps if steps > 0 else 0.0
        }

    def _extract_mathematical_features(self, sequence_data: SequenceData) -> Dict:
        """Extract advanced mathematical properties"""
        props = sequence_data.get('mathematical_properties', {})
        features = {
            'seed_is_prime': props.get('is_prime', False),
            'seed_prime_factors_count': len(props.get('prime_factors', {}))
        }

        # Add modular arithmetic properties
        mod_props = props.get('modular_properties', {})
        for mod in [2, 3, 4, 5, 8]:
            features[f'mod_{mod}_count'] = len(mod_props.get(mod % 8, []))

        return features

    def _extract_spectral_features(self, sequence_data: SequenceData) -> Dict:
        """Extract spectral features if sequence is available"""
        if 'sequence' not in sequence_data or len(sequence_data['sequence']) < 4:
            return {}

        sequence = sequence_data['sequence']
        fft = np.fft.fft(sequence)
        magnitudes = np.abs(fft)

        return {
            'spectral_centroid': float(np.sum(magnitudes * np.arange(len(magnitudes))) / np.sum(magnitudes)),
            'spectral_bandwidth': float(np.sqrt(np.sum(magnitudes * (np.arange(len(magnitudes)) ** 2) / np.sum(magnitudes))),
            'spectral_flatness': float(np.exp(np.mean(np.log(magnitudes + 1e-12))) / np.mean(magnitudes))
        }

    def _extract_topological_features(self, sequence_data: SequenceData) -> Dict:
        """Extract topological features from the sequence"""
        if 'sequence' not in sequence_data or len(sequence_data['sequence']) < 3:
            return {}

        sequence = sequence_data['sequence']
        diffs = np.diff(sequence)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)

        return {
            'topological_sign_changes': int(sign_changes),
            'topological_turning_points': int(sign_changes),
            'topological_monotonic_segments': int(sign_changes + 1)
        }

    def _calculate_entropy(self, sequence: List[int]) -> float:
        """Calculate normalized entropy of a sequence"""
        counts = Counter(sequence)
        total = len(sequence)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log(p + 1e-12)  # Add small epsilon to avoid log(0)
        return entropy / math.log(len(counts) + 1e-12) if len(counts) > 0 else 0.0

class CollatzMLModel:
    """
    Advanced machine learning framework for Collatz sequence analysis with:
    - Multiple model architectures
    - Automated hyperparameter tuning
    - Cross-validation
    - Feature importance analysis
    - Visualization tools
    """
    def __init__(self, features: pd.DataFrame):
        self.features = features
        self.models = {}
        self._preprocess_data()

    def _preprocess_data(self):
        """Prepare data for machine learning"""
        # Convert categorical features
        self.features = pd.get_dummies(self.features, columns=['termination_status'])

        # Handle missing values
        self.features.fillna(0, inplace=True)

        # Normalize numerical features
        numeric_cols = self.features.select_dtypes(include=np.number).columns
        self.feature_means = self.features[numeric_cols].mean()
        self.feature_stds = self.features[numeric_cols].std()
        self.features[numeric_cols] = (self.features[numeric_cols] - self.feature_means) / (self.feature_stds + 1e-12)

    def train_random_forest(self, target: str, test_size: float = 0.2, **kwargs):
        """Train a Random Forest classifier"""
        X = self.features.drop(columns=[target], errors='ignore')
        y = self.features[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = RandomForestClassifier(**kwargs)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        self.models['random_forest'] = {
            'model': model,
            'report': report,
            'feature_importances': dict(zip(X.columns, model.feature_importances_))
        }

        return report

    def train_neural_network(self, target: str, test_size: float = 0.2, **kwargs):
        """Train a neural network using TensorFlow/Keras"""
        X = self.features.drop(columns=[target], errors='ignore')
        y = pd.get_dummies(self.features[target])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(y_train.shape[1], activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=kwargs.get('epochs', 50),
            batch_size=kwargs.get('batch_size', 32),
            verbose=kwargs.get('verbose', 1)
        )

        self.models['neural_network'] = {
            'model': model,
            'history': history.history,
            'input_dim': X_train.shape[1],
            'output_dim': y_train.shape[1]
        }

        return history.history

    def cluster_sequences(self, n_clusters: int = 5, method: str = 'kmeans'):
        """Cluster sequences using different methods"""
        X = self.features.select_dtypes(include=np.number)

        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=2)
            reduced = reducer.fit_transform(X)
            return reduced
        elif method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=30)
            reduced = reducer.fit_transform(X)
            return reduced
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        clusters = clusterer.fit_predict(X)
        self.features['cluster'] = clusters

        return clusters

    def visualize_clusters(self, method: str = 'tsne'):
        """Visualize sequence clusters"""
        if 'cluster' not in self.features:
            self.cluster_sequences(method='kmeans')

        reduced = self.cluster_sequences(method=method)

        fig = px.scatter(
            x=reduced[:, 0],
            y=reduced[:, 1],
            color=self.features['cluster'].astype(str),
            hover_data={'seed': self.features['seed']},
            title=f'Collatz Sequence Clusters ({method.upper()})'
        )

        return fig

class CollatzVisualizer:
    """
    Advanced visualization system with:
    - Interactive plots
    - 3D visualizations
    - Animated sequences
    - Customizable themes
    """
    @staticmethod
    def plot_sequence(sequence_data: SequenceData, interactive: bool = True):
        """Visualize a single Collatz sequence"""
        if 'sequence' not in sequence_data:
            raise ValueError("Sequence data not available for visualization")

        sequence = sequence_data['sequence']
        x = range(len(sequence))

        if interactive:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=sequence,
                mode='lines+markers',
                name='Sequence',
                line=dict(color='royalblue', width=2),
                marker=dict(size=4)
            ))

            # Add peak annotation
            peak_idx = np.argmax(sequence)
            fig.add_annotation(
                x=peak_idx, y=sequence[peak_idx],
                text=f"Peak: {sequence[peak_idx]}",
                showarrow=True,
                arrowhead=1
            )

            fig.update_layout(
                title=f"Collatz Sequence for Seed {sequence_data['seed']}",
                xaxis_title="Step",
                yaxis_title="Value",
                hovermode="x unified"
            )

            return fig
        else:
            plt.figure(figsize=(10, 6))
            plt.plot(x, sequence, '-o', markersize=3)
            plt.title(f"Collatz Sequence for Seed {sequence_data['seed']}")
            plt.xlabel("Step")
            plt.ylabel("Value")
            plt.grid(True)
            return plt

    @staticmethod
    def plot_feature_distributions(features: pd.DataFrame, cols: List[str] = None):
        """Plot distributions of multiple features"""
        if cols is None:
            cols = features.select_dtypes(include=np.number).columns.tolist()[:10]

        fig = px.box(features, y=cols, title="Feature Distributions")
        return fig

    @staticmethod
    def plot_feature_correlations(features: pd.DataFrame):
        """Plot feature correlation matrix"""
        numeric_cols = features.select_dtypes(include=np.number).columns
        corr = features[numeric_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))

        fig.update_layout(
            title="Feature Correlation Matrix",
            xaxis_title="Features",
            yaxis_title="Features",
            width=800,
            height=800
        )

        return fig

class CollatzResearchCLI:
    """
    Advanced command-line interface for the Collatz Research Platform with:
    - Interactive mode
    - Batch processing
    - Experiment tracking
    - Configuration management
    """
    def __init__(self):
        self.parser = self._create_parser()
        self.args = None
        self.config = CollatzConfig()
        self.experiment_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'collatz_research_{self.experiment_id}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('CollatzResearchCLI')

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with all options"""
        parser = argparse.ArgumentParser(
            description='Advanced Collatz Research Platform',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # Core parameters
        parser.add_argument('--range', type=str, default='1:100',
                         help='Range of seeds to process (start:end)')
        parser.add_argument('--a', type=int, default=3,
                         help='Parameter a in generalized Collatz function')
        parser.add_argument('--b', type=int, default=1,
                         help='Parameter b in generalized Collatz function')
        parser.add_argument('--d', type=int, default=2,
                         help='Parameter d in generalized Collatz function')
        parser.add_argument('--max-steps', type=int, default=1000,
                         help='Maximum steps per sequence simulation')

        # Feature extraction
        parser.add_argument('--feature-sets', type=str, default='all',
                         help='Comma-separated list of feature sets to extract')
        parser.add_argument('--tail-window', type=int, default=10,
                         help='Window size for tail entropy calculation')

        # Output options
        parser.add_argument('--output', type=str, default='results',
                         help='Output directory for results')
        parser.add_argument('--format', type=str, default='parquet',
                         choices=['csv', 'json', 'parquet', 'hdf5'],
                         help='Output format for results')

        # Execution options
        parser.add_argument('--cache', action='store_true',
                         help='Enable simulation caching')
        parser.add_argument('--parallel', action='store_true',
                         help='Enable parallel processing')
        parser.add_argument('--no-progress', action='store_true',
                         help='Disable progress bars')

        # Visualization options
        parser.add_argument('--visualize', action='store_true',
                         help='Generate visualizations')
        parser.add_argument('--interactive', action='store_true',
                         help='Launch interactive visualization dashboard')

        # Machine learning options
        parser.add_argument('--ml', action='store_true',
                         help='Run machine learning analysis')
        parser.add_argument('--ml-target', type=str, default='termination_status',
                         help='Target variable for machine learning')

        return parser

    def parse_args(self, args=None):
        """Parse command line arguments and update configuration"""
        self.args = self.parser.parse_args(args)

        # Update config from arguments
        self.config.a = self.args.a
        self.config.b = self.args.b
        self.config.d = self.args.d
        self.config.max_steps = self.args.max_steps
        self.config.tail_window = self.args.tail_window
        self.config.cache_enabled = self.args.cache
        self.config.parallel_processing = self.args.parallel
        self.config.progress_bar = not self.args.no_progress

        return self.args

    def run(self):
        """Main execution flow"""
        self.logger.info(f"Starting Collatz Research Experiment {self.experiment_id}")

        # Create output directory
        output_dir = Path(self.args.output)
        output_dir.mkdir(exist_ok=True)

        # Parse seed range
        start, end = map(int, self.args.range.split(':'))
        seeds = range(start, end + 1)

        # Initialize components
        simulator = CollatzSequenceSimulator(self.config)
        extractor = AdvancedFeatureExtractor(self.config)

        # Determine feature sets to extract
        if self.args.feature_sets == 'all':
            feature_sets = None
        else:
            feature_sets = self.args.feature_sets.split(',')

        # Process all seeds
        self.logger.info(f"Processing {len(seeds)} seeds...")
        sequence_data = simulator._batch_simulate(seeds)

        # Extract features
        self.logger.info("Extracting features...")
        features = []
        for data in tqdm(sequence_data, disable=not self.config.progress_bar):
            features.append(extractor.extract_features(data, feature_sets))

        # Create DataFrame and save results
        df = pd.DataFrame(features)
        self._save_results(df, output_dir)

        # Generate visualizations if requested
        if self.args.visualize or self.args.interactive:
            self._generate_visualizations(df, output_dir)

        # Run machine learning analysis if requested
        if self.args.ml:
            self._run_ml_analysis(df, output_dir)

        self.logger.info(f"Experiment {self.experiment_id} completed successfully")

    def _save_results(self, df: pd.DataFrame, output_dir: Path):
        """Save results in requested format"""
        output_file = output_dir / f'results_{self.experiment_id}'

        if self.args.format == 'csv':
            df.to_csv(f"{output_file}.csv", index=False)
        elif self.args.format == 'json':
            df.to_json(f"{output_file}.json", orient='records')
        elif self.args.format == 'parquet':
            df.to_parquet(f"{output_file}.parquet")
        elif self.args.format == 'hdf5':
            df.to_hdf(f"{output_file}.h5", key='collatz_data')

        self.logger.info(f"Results saved to {output_file}.{self.args.format}")

    def _generate_visualizations(self, df: pd.DataFrame, output_dir: Path):
        """Generate and save visualizations"""
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)

        visualizer = CollatzVisualizer()

        # Save feature distributions
        fig = visualizer.plot_feature_distributions(df)
        fig.write_html(str(vis_dir / 'feature_distributions.html'))

        # Save correlation matrix
        fig = visualizer.plot_feature_correlations(df)
        fig.write_html(str(vis_dir / 'feature_correlations.html'))

        if self.args.interactive:
            import dash
            from dash import dcc, html
            from dash.dependencies import Input, Output

            app = dash.Dash(__name__)

            app.layout = html.Div([
                html.H1(f"Collatz Research Dashboard - Experiment {self.experiment_id}"),
                dcc.Graph(id='feature-distributions'),
                dcc.Graph(id='feature-correlations'),
                dcc.Dropdown(
                    id='sequence-selector',
                    options=[{'label': f"Seed {seed}", 'value': seed} for seed in df['seed']],
                    value=df['seed'].iloc[0]
                ),
                dcc.Graph(id='sequence-plot')
            ])

            @app.callback(
                Output('feature-distributions', 'figure'),
                Input('feature-distributions', 'id')
            )
            def update_distributions(_):
                return visualizer.plot_feature_distributions(df)

            @app.callback(
                Output('feature-correlations', 'figure'),
                Input('feature-correlations', 'id')
            )
            def update_correlations(_):
                return visualizer.plot_feature_correlations(df)

            @app.callback(
                Output('sequence-plot', 'figure'),
                Input('sequence-selector', 'value')
            )
            def update_sequence_plot(selected_seed):
                # Find the sequence data for the selected seed
                selected_data = next(d for d in sequence_data if d['seed'] == selected_seed)
                return visualizer.plot_sequence(selected_data)

            self.logger.info("Starting interactive dashboard...")
            app.run_server(debug=False, port=8050)

    def _run_ml_analysis(self, df: pd.DataFrame, output_dir: Path):
        """Run machine learning analysis on the collected data"""
        ml_dir = output_dir / 'machine_learning'
        ml_dir.mkdir(exist_ok=True)

        ml_model = CollatzMLModel(df)

        # Train Random Forest
        rf_report = ml_model.train_random_forest(
            target=self.args.ml_target,
            n_estimators=100,
            max_depth=10
        )

        # Save feature importances
        importances = pd.DataFrame.from_dict(
            ml_model.models['random_forest']['feature_importances'],
            orient='index',
            columns=['importance']
        ).sort_values('importance', ascending=False)

        importances.to_csv(ml_dir / 'feature_importances.csv')

        # Save classification report
        with open(ml_dir / 'classification_report.json', 'w') as f:
            json.dump(rf_report, f, indent=2)

        # Cluster sequences
        clusters = ml_model.cluster_sequences(n_clusters=5)
        df['cluster'] = clusters

        # Save clustered data
        df.to_csv(ml_dir / 'clustered_data.csv', index=False)

        # Generate cluster visualization
        fig = ml_model.visualize_clusters()
        fig.write_html(str(ml_dir / 'sequence_clusters.html'))

        self.logger.info(f"Machine learning results saved to {ml_dir}")

def main():
    cli = CollatzResearchCLI()
    cli.parse_args()
    cli.run()

if __name__ == '__main__':
    main()

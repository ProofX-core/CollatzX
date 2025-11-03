"""
Collatz Conjecture Advanced Analysis Framework — Monolithic, Cleaned Version
- Visualization stack: Matplotlib + Plotly (no Dash)
- Preserves analytical quality, models, caching, and outputs
- Deterministic: NumPy + PyTorch seeded; stable multiprocessing
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
import torch
import torch.nn as nn
from matplotlib import cm
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from statsmodels.tsa.stattools import acf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ============================================================
# Logging
# ============================================================
LOG_DIR = Path(".logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(LOG_DIR / "collatz_debug.log", maxBytes=10 * 1024 * 1024, backupCount=5),
        logging.FileHandler(LOG_DIR / "collatz_analysis.log"),
    ],
)
logger = logging.getLogger("collatz")

# ============================================================
# Constants & Config
# ============================================================
MAX_ITERATIONS = 10_000_000
MAX_VALUE = 10**100
DEFAULT_OUTPUT_DIR = Path("collatz_results")
CACHE_DIR = Path(".collatz_cache")
CACHE_DIR.mkdir(exist_ok=True)
PLOT_STYLE = "seaborn-v0_8-darkgrid"  # still available via matplotlib

class Config:
    USE_GPU = torch.cuda.is_available()
    MAX_PROCESSES = max(1, (os_cpu := __import__("os").cpu_count() or 2) - 2)
    PRECISION = np.float64
    ENABLE_CACHE = True
    ENABLE_PROGRESS_BAR = True
    DEFAULT_TEST_SIZE = 0.2
    RANDOM_SEED = 42
    EARLY_STOPPING_PATIENCE = 5
    # Grid sizes tuned to be robust but not excessive
    RF_N_EST = [100, 200]
    RF_MAX_DEPTH = [None, 10, 20]
    RF_MIN_SAMPLES_SPLIT = [2, 5]

config = Config()

# Determinism
np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# Enums & Registries
# ============================================================
class ModelType(Enum):
    EXPONENTIAL = auto()
    POWER_LAW = auto()
    LOGARITHMIC = auto()
    LINEAR = auto()
    RANDOM_FOREST = auto()
    NEURAL_NET = auto()

MODEL_REGISTRY: Dict[ModelType, Any] = {}

def register_model(model_type: ModelType):
    def decorator(cls):
        MODEL_REGISTRY[model_type] = cls
        return cls
    return decorator

class CollatzVariant(Enum):
    CLASSIC = auto()
    GENERALIZED = auto()
    FRACTAL = auto()
    MODULAR = auto()

    @classmethod
    def get_function(cls, variant: "CollatzVariant", **params):
        if variant == cls.CLASSIC:
            return lambda n: 3 * n + 1 if n % 2 else n // 2
        if variant == cls.GENERALIZED:
            p = params.get("p", 3)
            q = params.get("q", 1)
            d = params.get("d", 2)
            return lambda n: p * n + q if n % 2 else n // d
        if variant == cls.FRACTAL:
            return lambda n: (3 * n + 1) // 2 if n % 2 else n // 2
        if variant == cls.MODULAR:
            mod = params.get("mod", 3)
            return lambda n: (2 * n + 1) if n % mod == 0 else (n // 2)
        raise ValueError(f"Unsupported variant: {variant.name}")

# ============================================================
# Data Structures
# ============================================================
@dataclass
class CollatzSequence:
    starting_value: int
    sequence: List[int] = field(default_factory=list)
    stopping_time: Optional[int] = None
    max_value: Optional[int] = None
    features: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.sequence:
            self.generate_sequence()

    def generate_sequence(self):
        n = self.starting_value
        seq = [n]
        steps = 0
        while n != 1 and steps < MAX_ITERATIONS:
            if n > MAX_VALUE:
                raise OverflowError(f"Value {n} exceeds MAX_VALUE")
            n = 3 * n + 1 if n % 2 else n // 2
            seq.append(n)
            steps += 1
        if steps >= MAX_ITERATIONS:
            warnings.warn(f"Max iterations reached for start={self.starting_value}")
        self.sequence = seq
        self.stopping_time = len(seq) - 1
        self.max_value = max(seq) if seq else None

@dataclass
class FitResult:
    parameters: np.ndarray
    errors: np.ndarray
    r_squared: float
    adjusted_r_squared: float
    aic: float
    bic: float
    model_type: ModelType
    fit_time: float
    cross_val_scores: Optional[np.ndarray] = None
    feature_importances: Optional[np.ndarray] = None
    model: Optional[Any] = None
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        as_arr = lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        return {
            "model_type": self.model_type.name,
            "parameters": as_arr(self.parameters),
            "errors": as_arr(self.errors),
            "r_squared": self.r_squared,
            "adjusted_r_squared": self.adjusted_r_squared,
            "aic": self.aic,
            "bic": self.bic,
            "fit_time": self.fit_time,
            "cross_val_scores": as_arr(self.cross_val_scores) if self.cross_val_scores is not None else None,
            "feature_importances": as_arr(self.feature_importances) if self.feature_importances is not None else None,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
        }

# ============================================================
# Feature Extraction
# ============================================================
class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, sequence: List[int]) -> Dict[str, float]:
        ...

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        ...

class SequenceTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, extractor: FeatureExtractor):
        self.extractor = extractor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self.extractor.extract(seq) for seq in X])

class StatisticalFeatureExtractor(FeatureExtractor):
    def __init__(self):
        self._names = [
            "length", "max_value", "min_value", "mean_value", "median_value",
            "std_dev", "skewness", "kurtosis", "entropy", "parity_ratio",
            "even_streak_max", "odd_streak_max", "growth_rate", "volatility",
            "autocorrelation", "hurst_exponent", "lyapunov_exponent",
        ]

    @property
    def feature_names(self) -> List[str]:
        return self._names

    def extract(self, sequence: List[int]) -> Dict[str, float]:
        if not sequence:
            return {n: 0.0 for n in self._names}

        arr = np.array(sequence, dtype=np.float64)
        parity = [v % 2 for v in sequence]

        out = {
            "length": float(len(arr)),
            "max_value": float(np.max(arr)),
            "min_value": float(np.min(arr)),
            "mean_value": float(np.mean(arr)),
            "median_value": float(np.median(arr)),
            "std_dev": float(np.std(arr)),
            "skewness": float(self._skew(arr)),
            "kurtosis": float(self._kurt(arr)),
            "entropy": float(self._entropy(sequence)),
            "parity_ratio": float(np.mean(parity)),
            "even_streak_max": float(self._max_streak(parity, 0)),
            "odd_streak_max": float(self._max_streak(parity, 1)),
            "growth_rate": float(self._growth_rate(sequence)),
            "volatility": float(self._volatility(sequence)),
            "autocorrelation": float(self._autocorr(sequence, 1)),
            "hurst_exponent": float(self._hurst(sequence)),
            "lyapunov_exponent": float(self._lyap(sequence)),
        }
        return out

    @staticmethod
    def _max_streak(seq: List[int], target: int) -> int:
        m = c = 0
        for v in seq:
            if v == target:
                c += 1
                m = max(m, c)
            else:
                c = 0
        return m

    @staticmethod
    def _entropy(sequence: List[int]) -> float:
        vals, cnts = np.unique(np.abs(sequence), return_counts=True)
        p = cnts.astype(np.float64) / cnts.sum()
        return float(-(p * np.log2(p + 1e-12)).sum())

    @staticmethod
    def _growth_rate(sequence: List[int]) -> float:
        if len(sequence) < 2:
            return 0.0
        diffs = np.diff(sequence)
        return float(np.mean(diffs) / (np.max(sequence) + 1e-12))

    @staticmethod
    def _volatility(sequence: List[int]) -> float:
        if len(sequence) < 2:
            return 0.0
        prev = np.array(sequence[:-1], dtype=np.float64)
        ret = np.diff(sequence) / np.where(prev == 0, 1.0, prev)
        return float(np.std(ret))

    @staticmethod
    def _autocorr(sequence: List[int], lag: int) -> float:
        if len(sequence) < lag + 1:
            return 0.0
        return float(acf(sequence, nlags=lag, fft=True)[lag])

    @staticmethod
    def _hurst(sequence: List[int]) -> float:
        if len(sequence) < 10:
            return 0.5
        lags = range(2, min(20, len(sequence) // 2))
        tau = [np.mean(np.abs(np.diff(sequence[0::lag]))) for lag in lags]
        slope, _ = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return float(slope)

    @staticmethod
    def _lyap(sequence: List[int]) -> float:
        if len(sequence) < 10:
            return 0.0
        dv = []
        for i in range(1, min(10, len(sequence))):
            a, b = sequence[i - 1], sequence[i]
            if a != 0:
                dv.append(np.log(abs(b / a)))
        return float(np.mean(dv)) if dv else 0.0

    @staticmethod
    def _skew(x: np.ndarray) -> float:
        if len(x) < 3:
            return 0.0
        mu, sd = np.mean(x), np.std(x)
        if sd == 0:
            return 0.0
        return float(np.mean((x - mu) ** 3) / (sd**3))

    @staticmethod
    def _kurt(x: np.ndarray) -> float:
        if len(x) < 4:
            return 0.0
        mu, sd = np.mean(x), np.std(x)
        if sd == 0:
            return 0.0
        return float(np.mean((x - mu) ** 4) / (sd**4) - 3)

class AlgebraicFeatureExtractor(FeatureExtractor):
    def __init__(self):
        self._names = [
            "prime_factors_count",
            "distinct_primes",
            "prime_multiplicity",
            "is_power_of_two",
            "log2_ratio",
            "modular_pattern",
            "binary_density",
            "binary_transitions",
            "binary_entropy",
            "gcd_with_max",
            "lcm_with_max",
            "divisor_count",
        ]

    @property
    def feature_names(self) -> List[str]:
        return self._names

    @staticmethod
    @lru_cache(maxsize=100000)
    def _factorint_cached(x: int) -> Dict[int, int]:
        return sp.factorint(x)

    def extract(self, sequence: List[int]) -> Dict[str, float]:
        if not sequence:
            return {n: 0.0 for n in self._names}

        features = {
            "prime_factors_count": float(self._total_prime_factors(sequence)),
            "distinct_primes": float(self._distinct_prime_factors(sequence)),
            "prime_multiplicity": float(self._avg_prime_multiplicity(sequence)),
            "is_power_of_two": float(self._power2_ratio(sequence)),
            "log2_ratio": float(self._log2_ratio(sequence)),
            "modular_pattern": float(self._modular_pattern(sequence)),
            "binary_density": float(self._binary_density(sequence)),
            "binary_transitions": float(self._binary_transitions(sequence)),
            "binary_entropy": float(self._binary_entropy(sequence)),
            "gcd_with_max": float(self._gcd_with_max(sequence)),
            "lcm_with_max": float(self._lcm_with_max(sequence)),
            "divisor_count": float(self._avg_divisors(sequence)),
        }
        return features

    def _total_prime_factors(self, seq: List[int]) -> float:
        return sum(len(self._factorint_cached(abs(int(n)))) for n in seq if n)

    def _distinct_prime_factors(self, seq: List[int]) -> float:
        primes = set()
        for n in seq:
            if n:
                primes.update(self._factorint_cached(abs(int(n))).keys())
        return float(len(primes))

    def _avg_prime_multiplicity(self, seq: List[int]) -> float:
        total = count = 0
        for n in seq:
            if n:
                fac = self._factorint_cached(abs(int(n)))
                total += sum(fac.values())
                count += len(fac)
        return total / count if count else 0.0

    @staticmethod
    def _power2_ratio(seq: List[int]) -> float:
        c = sum(1 for n in seq if n > 0 and (n & (n - 1)) == 0)
        return c / len(seq) if seq else 0.0

    @staticmethod
    def _log2_ratio(seq: List[int]) -> float:
        if len(seq) < 2:
            return 0.0
        vals = []
        for i in range(1, len(seq)):
            a, b = seq[i - 1], seq[i]
            if a > 0 and b > 0:
                vals.append(np.log2(b) - np.log2(a))
        return float(np.mean(vals)) if vals else 0.0

    @staticmethod
    def _modular_pattern(seq: List[int]) -> float:
        if len(seq) < 3:
            return 0.0
        sigmas = []
        for m in (2, 3, 4, 5):
            sigmas.append(np.std([n % m for n in seq]))
        return float(np.mean(sigmas))

    @staticmethod
    def _binary_density(seq: List[int]) -> float:
        acc = cnt = 0
        for n in seq:
            if n > 0:
                b = bin(int(n))[2:]
                acc += b.count("1") / len(b)
                cnt += 1
        return acc / cnt if cnt else 0.0

    @staticmethod
    def _binary_transitions(seq: List[int]) -> float:
        acc = cnt = 0
        for n in seq:
            if n > 0:
                b = bin(int(n))[2:]
                acc += sum(1 for a, c in zip(b, b[1:]) if a != c) / max(1, len(b) - 1)
                cnt += 1
        return acc / cnt if cnt else 0.0

    @staticmethod
    def _binary_entropy(seq: List[int]) -> float:
        acc = cnt = 0
        for n in seq:
            if n > 0:
                b = bin(int(n))[2:]
                c0 = b.count("0")
                c1 = len(b) - c0
                tot = len(b)
                ent = 0.0
                for c in (c0, c1):
                    p = c / tot if tot else 0.0
                    if p > 0:
                        ent -= p * np.log2(p)
                acc += ent
                cnt += 1
        return acc / cnt if cnt else 0.0

    @staticmethod
    def _gcd_with_max(seq: List[int]) -> float:
        if not seq:
            return 0.0
        mx = int(max(seq))
        vals = [sp.gcd(int(n), mx) for n in seq if n]
        return float(np.mean(vals)) if vals else 0.0

    @staticmethod
    def _lcm_with_max(seq: List[int]) -> float:
        if not seq:
            return 0.0
        mx = int(max(seq))
        vals = [sp.lcm(int(n), mx) for n in seq if n]
        return float(np.mean(vals)) if vals else 0.0

    @staticmethod
    def _avg_divisors(seq: List[int]) -> float:
        total = cnt = 0
        for n in seq:
            if n:
                total += len(sp.divisors(abs(int(n))))
                cnt += 1
        return total / cnt if cnt else 0.0

class FeatureUnion:
    def __init__(self, extractors: List[FeatureExtractor]):
        self.extractors = extractors
        self._names = []
        for e in extractors:
            self._names.extend(e.feature_names)

    def extract(self, sequence: List[int]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for e in self.extractors:
            out.update(e.extract(sequence))
        return out

    @property
    def feature_names(self) -> List[str]:
        return self._names

# ============================================================
# Optional PyTorch Regressor (not used by default, but kept)
# ============================================================
class PyTorchModel(nn.Module):
    def __init__(self, input_size: int, hidden: List[int] = [64, 32]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_size
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.2)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CollatzDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class TorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_size: int, hidden: List[int] = [64, 32], lr: float = 1e-3, epochs: int = 100, batch_size: int = 32):
        self.input_size = input_size
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = PyTorchModel(input_size, hidden)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        ds = CollatzDataset(Xs, y)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
        self.model.to(self.device)
        crit = nn.MSELoss()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=5, factor=0.5)
        best = float("inf")
        patience = 0
        for _ in range(self.epochs):
            self.model.train()
            ep = 0.0
            for bx, by in dl:
                bx, by = bx.to(self.device), by.to(self.device)
                opt.zero_grad()
                out = self.model(bx)
                loss = crit(out, by)
                loss.backward()
                opt.step()
                ep += loss.item()
            ep /= max(1, len(dl))
            sched.step(ep)
            if ep < best:
                best, patience = ep, 0
            else:
                patience += 1
                if patience >= config.EARLY_STOPPING_PATIENCE:
                    break
        return self

    def predict(self, X):
        self.model.eval()
        Xs = self.scaler.transform(X)
        xt = torch.tensor(Xs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            y = self.model(xt).cpu().numpy().flatten()
        return y

# ============================================================
# Parametric & ML Models
# ============================================================
@register_model(ModelType.EXPONENTIAL)
class ExponentialModel:
    @staticmethod
    def f(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * (b ** n) + c

    def fit(self, X, y):
        try:
            self.params_, self.pcov_ = curve_fit(self.f, X.flatten(), y, p0=[1, 1.1, 0], maxfev=5000)
        except Exception as e:
            logger.error(f"Exponential fit failed: {e}")
            self.params_, self.pcov_ = np.array([1, 1, 0]), np.eye(3)
        return self

    def predict(self, X) -> np.ndarray:
        return self.f(X.flatten(), *self.params_)

@register_model(ModelType.POWER_LAW)
class PowerLawModel:
    @staticmethod
    def f(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * np.power(n, b) + c

    def fit(self, X, y):
        try:
            self.params_, self.pcov_ = curve_fit(self.f, X.flatten(), y, p0=[1, 1, 0], maxfev=5000)
        except Exception as e:
            logger.error(f"Power law fit failed: {e}")
            self.params_, self.pcov_ = np.array([1, 1, 0]), np.eye(3)
        return self

    def predict(self, X) -> np.ndarray:
        return self.f(X.flatten(), *self.params_)

@register_model(ModelType.LOGARITHMIC)
class LogarithmicModel:
    @staticmethod
    def f(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * np.log(n + 1e-12) + b * n + c

    def fit(self, X, y):
        try:
            self.params_, self.pcov_ = curve_fit(self.f, X.flatten(), y, p0=[1, 1, 0], maxfev=5000)
        except Exception as e:
            logger.error(f"Logarithmic fit failed: {e}")
            self.params_, self.pcov_ = np.array([1, 1, 0]), np.eye(3)
        return self

    def predict(self, X) -> np.ndarray:
        return self.f(X.flatten(), *self.params_)

@register_model(ModelType.LINEAR)
class LinearModel:
    @staticmethod
    def f(n: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * n + b

    def fit(self, X, y):
        try:
            self.params_, self.pcov_ = curve_fit(self.f, X.flatten(), y, p0=[1, 0], maxfev=5000)
        except Exception as e:
            logger.error(f"Linear fit failed: {e}")
            self.params_, self.pcov_ = np.array([1, 0]), np.eye(2)
        return self

    def predict(self, X) -> np.ndarray:
        return self.f(X.flatten(), *self.params_)

@register_model(ModelType.RANDOM_FOREST)
class RandomForestModel:
    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("rf", RandomForestRegressor(n_estimators=200, random_state=config.RANDOM_SEED, n_jobs=-1)),
            ]
        )
        self.best_params_ = None

    def fit(self, X, y):
        grid = {
            "rf__n_estimators": config.RF_N_EST,
            "rf__max_depth": config.RF_MAX_DEPTH,
            "rf__min_samples_split": config.RF_MIN_SAMPLES_SPLIT,
        }
        gs = GridSearchCV(self.pipeline, grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=0)
        gs.fit(X, y)
        self.pipeline = gs.best_estimator_
        self.best_params_ = gs.best_params_
        return self

    def predict(self, X) -> np.ndarray:
        return self.pipeline.predict(X)

@register_model(ModelType.NEURAL_NET)
class NeuralNetworkModel:
    def __init__(self, hidden_layer_sizes=(100, 50)):
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("nn", MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000, random_state=config.RANDOM_SEED, early_stopping=True)),
            ]
        )

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.pipeline.predict(X)

# ============================================================
# Parallel Worker
# ============================================================
def _compute_sequence_worker(base: int, exponent: int, variant_name: str, variant_params: Dict[str, Any]) -> Tuple[int, Optional[int], CollatzSequence]:
    try:
        variant = CollatzVariant[variant_name]
        f = CollatzVariant.get_function(variant, **(variant_params or {}))
        start_value = pow(base, exponent)

        seq: List[int] = []
        cur = start_value
        steps = 0
        while cur != 1 and steps < MAX_ITERATIONS:
            if cur > MAX_VALUE:
                raise OverflowError(f"value {cur} exceeds MAX_VALUE")
            seq.append(cur)
            cur = f(cur)
            steps += 1
        if steps >= MAX_ITERATIONS:
            warnings.warn(f"Max iterations for exponent {exponent}")
        seq.append(1)

        cs = CollatzSequence(starting_value=start_value, sequence=seq)

        extractor = FeatureUnion([StatisticalFeatureExtractor(), AlgebraicFeatureExtractor()])
        cs.features = extractor.extract(seq)
        return exponent, cs.stopping_time, cs
    except Exception as e:
        logger.error(f"Worker error (exp={exponent}): {e}")
        v = pow(base, exponent)
        return exponent, None, CollatzSequence(starting_value=v, sequence=[v, 1])

# ============================================================
# Analyzer
# ============================================================
class CollatzAnalyzer:
    def __init__(self, base: int = 2, output_dir: Optional[str] = None, variant: CollatzVariant = CollatzVariant.CLASSIC, variant_params: Optional[Dict[str, Any]] = None):
        self.base = base
        self.variant = variant
        self.variant_params = variant_params or {}
        self.collatz_func = CollatzVariant.get_function(variant, **self.variant_params)

        self.n_values: Optional[np.ndarray] = None
        self.T_values: Optional[List[Optional[int]]] = None
        self.sequences: Dict[int, CollatzSequence] = {}
        self.fit_results: Dict[ModelType, FitResult] = {}
        self.computation_time = 0.0
        self.results_df: Optional[pd.DataFrame] = None

        self.feature_extractor = FeatureUnion([StatisticalFeatureExtractor(), AlgebraicFeatureExtractor()])

        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True, parents=True)

        plt.style.use(PLOT_STYLE)
        self._configure_plotting()
        logger.info(f"Initialized CollatzAnalyzer base={base} variant={variant.name}")

    @staticmethod
    def _configure_plotting():
        plt.rcParams.update(
            {
                "figure.figsize": (12, 8),
                "font.size": 12,
                "axes.labelsize": 14,
                "axes.titlesize": 16,
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.1,
            }
        )

    def _cache_key(self, n: int) -> str:
        key = json.dumps({"base": self.base, "n": n, "variant": self.variant.name, "variant_params": self.variant_params}, sort_keys=True)
        import hashlib

        return hashlib.md5(key.encode()).hexdigest()

    def _load_cache(self, n: int) -> Optional[CollatzSequence]:
        if not config.ENABLE_CACHE:
            return None
        f = CACHE_DIR / f"{self._cache_key(n)}.pkl"
        if f.exists():
            try:
                with open(f, "rb") as fh:
                    return pickle.load(fh)
            except Exception as e:
                logger.warning(f"Cache load failed (n={n}): {e}")
        return None

    def _save_cache(self, n: int, cs: CollatzSequence) -> None:
        if not config.ENABLE_CACHE:
            return
        f = CACHE_DIR / f"{self._cache_key(n)}.pkl"
        try:
            with open(f, "wb") as fh:
                pickle.dump(cs, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning(f"Cache save failed (n={n}): {e}")

    def collatz_sequence(self, n: int) -> CollatzSequence:
        cached = self._load_cache(n)
        if cached:
            return cached
        seq: List[int] = []
        cur = n
        steps = 0
        while cur != 1 and steps < MAX_ITERATIONS:
            if cur > MAX_VALUE:
                raise OverflowError(f"{cur} exceeds MAX_VALUE")
            seq.append(cur)
            cur = self.collatz_func(cur)
            steps += 1
        if steps >= MAX_ITERATIONS:
            warnings.warn(f"Max iterations reached for n={n}")
        seq.append(1)
        cs = CollatzSequence(starting_value=n, sequence=seq)
        cs.features = self.feature_extractor.extract(seq)
        self._save_cache(n, cs)
        return cs

    def collatz_stopping_time(self, exponent: int) -> Optional[int]:
        try:
            value = self.base**exponent
            cs = self.collatz_sequence(value)
            self.sequences[exponent] = cs
            return cs.stopping_time
        except Exception as e:
            logger.error(f"Stopping time error base^{exponent}: {e}")
            return None

    def parallel_compute(self, exponents: np.ndarray) -> List[Optional[int]]:
        logger.info(f"Parallel compute for {len(exponents)} exponents")
        start = time.perf_counter()
        pbar = tqdm(total=len(exponents), disable=not config.ENABLE_PROGRESS_BAR, desc="Compute", unit="exp")

        results: List[Optional[int]] = [None] * len(exponents)
        variant_name = self.variant.name
        variant_params = self.variant_params

        with ProcessPoolExecutor(max_workers=config.MAX_PROCESSES) as pool:
            fut2idx = {
                pool.submit(_compute_sequence_worker, self.base, int(exp), variant_name, variant_params): i
                for i, exp in enumerate(exponents)
            }
            for fut in as_completed(fut2idx):
                idx = fut2idx[fut]
                exp = int(exponents[idx])
                try:
                    _exp, stop_time, cs = fut.result()
                    results[idx] = stop_time
                    self.sequences[exp] = cs
                    self._save_cache(self.base**exp, cs)
                except Exception as e:
                    logger.error(f"Worker failed for exp={exp}: {e}")
                    results[idx] = None
                pbar.update(1)
        pbar.close()
        self.computation_time = time.perf_counter() - start
        logger.info(f"Parallel compute done in {self.computation_time:.2f}s")
        return results

    def _prepare_ml(self) -> Tuple[np.ndarray, np.ndarray]:
        feats, targs = [], []
        names = self.feature_extractor.feature_names
        for n in self.n_values:
            ni = int(n)
            if ni in self.sequences:
                s = self.sequences[ni]
                feats.append([s.features.get(k, np.nan) for k in names])
                targs.append(s.stopping_time)
        return np.array(feats, dtype=np.float64), np.array(targs, dtype=np.float64)

    @staticmethod
    def _metrics(model, Xtr, Xte, ytr, yte) -> Dict[str, float]:
        pred_tr = model.predict(Xtr)
        pred_te = model.predict(Xte)
        return {
            "train_mse": float(mean_squared_error(ytr, pred_tr)),
            "train_mae": float(mean_absolute_error(ytr, pred_tr)),
            "train_r2": float(r2_score(ytr, pred_tr)),
            "train_exp_var": float(explained_variance_score(ytr, pred_tr)),
            "test_mse": float(mean_squared_error(yte, pred_te)),
            "test_mae": float(mean_absolute_error(yte, pred_te)),
            "test_r2": float(r2_score(yte, pred_te)),
            "test_exp_var": float(explained_variance_score(yte, pred_te)),
        }

    def _fit_ml(self, model_type: ModelType) -> Optional[FitResult]:
        X, y = self._prepare_ml()
        if len(X) < 20:
            logger.warning(f"Insufficient samples ({len(X)}) for ML")
            return None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=config.DEFAULT_TEST_SIZE, random_state=config.RANDOM_SEED)
        t0 = time.perf_counter()
        try:
            cls = MODEL_REGISTRY.get(model_type)
            if cls is None:
                return None
            model = cls()
            model.fit(Xtr, ytr)
            metr = self._metrics(model, Xtr, Xte, ytr, yte)
            cv = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
            importances = None
            # Try to infer feature importances
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "pipeline"):
                # RandomForest inside pipeline
                step = dict(model.pipeline.named_steps).get("rf")
                if hasattr(step, "feature_importances_"):
                    importances = step.feature_importances_
            return FitResult(
                parameters=np.array([]),
                errors=np.array([metr["test_mse"]]),
                r_squared=metr["test_r2"],
                adjusted_r_squared=1 - (1 - metr["test_r2"]) * (len(y) - 1) / (len(y) - X.shape[1] - 1),
                aic=len(y) * np.log(max(metr["test_mse"], 1e-12)) + 2 * X.shape[1],
                bic=len(y) * np.log(max(metr["test_mse"], 1e-12)) + np.log(len(y)) * X.shape[1],
                model_type=model_type,
                fit_time=time.perf_counter() - t0,
                cross_val_scores=cv,
                feature_importances=importances,
                model=model,
                training_metrics={k: v for k, v in metr.items() if k.startswith("train_")},
                validation_metrics={k: v for k, v in metr.items() if k.startswith("test_")},
            )
        except Exception as e:
            logger.error(f"ML fit failed for {model_type.name}: {e}")
            return None

    def _fit_param(self, model_type: ModelType) -> Optional[FitResult]:
        valid_idx = [i for i, t in enumerate(self.T_values) if t is not None]
        if len(valid_idx) < 3:
            logger.warning("Not enough valid points for parametric fit")
            return None
        x = self.n_values[valid_idx]
        y = np.array([self.T_values[i] for i in valid_idx], dtype=float)
        logger.info(f"Fitting {model_type.name} on {len(x)} points")
        t0 = time.perf_counter()
        try:
            cls = MODEL_REGISTRY.get(model_type)
            if cls is None:
                return None
            model = cls()
            model.fit(x.reshape(-1, 1), y)
            pred = model.predict(x.reshape(-1, 1))
            r2 = r2_score(y, pred)
            mse = mean_squared_error(y, pred)
            n = len(y)
            k = len(getattr(model, "params_", [])) or 1
            return FitResult(
                parameters=getattr(model, "params_", np.array([])),
                errors=np.sqrt(np.diag(getattr(model, "pcov_", np.eye(k)))) if hasattr(model, "pcov_") else np.array([mse]),
                r_squared=r2,
                adjusted_r_squared=1 - (1 - r2) * (n - 1) / (n - k - 1),
                aic=n * np.log(max(mse, 1e-12)) + 2 * k,
                bic=n * np.log(max(mse, 1e-12)) + np.log(n) * k,
                model_type=model_type,
                fit_time=time.perf_counter() - t0,
                model=model,
            )
        except Exception as e:
            logger.error(f"Parametric fit failed for {model_type.name}: {e}")
            return None

    def fit_models(self, model_types: Optional[List[ModelType]] = None) -> None:
        if model_types is None:
            model_types = list(MODEL_REGISTRY.keys())
        self.fit_results.clear()
        for mt in model_types:
            try:
                res = self._fit_ml(mt) if mt in (ModelType.RANDOM_FOREST, ModelType.NEURAL_NET) else self._fit_param(mt)
                if res:
                    self.fit_results[mt] = res
                    logger.info(f"Fitted {mt.name}: R²={res.r_squared:.4f}, AIC={res.aic:.1f}, t={res.fit_time:.2f}s")
            except Exception as e:
                logger.error(f"Fit error for {mt.name}: {e}")

    def create_results_dataframe(self) -> None:
        if self.n_values is None or self.T_values is None:
            logger.error("No computation results to build dataframe")
            return

        df = pd.DataFrame(
            {
                "exponent": self.n_values,
                "value": [self.base ** int(n) if t is not None else np.nan for n, t in zip(self.n_values, self.T_values)],
                "stopping_time": self.T_values,
            }
        )

        # Features (aligned)
        feat_rows: List[Dict[str, float]] = []
        for n in self.n_values:
            ni = int(n)
            if ni in self.sequences:
                feat_rows.append(self.sequences[ni].features)
            else:
                feat_rows.append({name: np.nan for name in self.feature_extractor.feature_names})
        fdf = pd.DataFrame(feat_rows)
        df = pd.concat([df, fdf], axis=1)

        # Add model predictions / residuals
        for mt, fr in self.fit_results.items():
            col = f"fitted_{mt.name.lower()}"
            try:
                if mt in (ModelType.RANDOM_FOREST, ModelType.NEURAL_NET):
                    if fr.model is not None:
                        feats = []
                        for n in self.n_values:
                            ni = int(n)
                            if ni in self.sequences:
                                feats.append([self.sequences[ni].features.get(k, np.nan) for k in self.feature_extractor.feature_names])
                            else:
                                feats.append([np.nan] * len(self.feature_extractor.feature_names))
                        pred = fr.model.predict(np.array(feats, dtype=np.float64))
                        df[col] = pred
                else:
                    if hasattr(fr.model, "predict"):
                        df[col] = fr.model.predict(self.n_values.reshape(-1, 1))
                if col in df.columns:
                    df[f"residual_{mt.name.lower()}"] = df["stopping_time"] - df[col]
            except Exception as e:
                logger.error(f"Add model column failed for {mt.name}: {e}")

        self.results_df = df
        self._add_derived_columns()

    def _add_derived_columns(self) -> None:
        if self.results_df is None:
            return
        df = self.results_df
        for col in ["stopping_time", "max_value", "value"]:
            if col in df.columns:
                df[f"log_{col}"] = np.log1p(df[col])
        if "max_value" in df.columns and "value" in df.columns:
            df["max_to_start_ratio"] = df["max_value"] / df["value"]
        if "stopping_time" in df.columns:
            df["stopping_time_diff"] = df["stopping_time"].diff()
        for col in ["stopping_time", "max_value"]:
            if col in df.columns:
                for w in (3, 5, 10):
                    df[f"{col}_rolling_mean_{w}"] = df[col].rolling(window=w).mean()

    # -------------------- Plotting --------------------
    def plot_main_results(self) -> Optional[plt.Figure]:
        if self.n_values is None or self.T_values is None:
            return None
        valid = [(n, t) for n, t in zip(self.n_values, self.T_values) if t is not None]
        if not valid:
            return None
        x = np.array([v[0] for v in valid], dtype=float)
        y = np.array([v[1] for v in valid], dtype=float)

        fig, ax = plt.subplots(figsize=(14, 8))
        sc = ax.scatter(x, y, c=y, cmap="viridis", s=80, alpha=0.85, edgecolors="w", linewidth=0.5, label="Data")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Stopping Time", rotation=270, labelpad=15)

        n_fit = np.linspace(float(min(self.n_values)), float(max(self.n_values)), 500)
        palette = cm.get_cmap("tab10")
        for i, (mt, fr) in enumerate(self.fit_results.items()):
            color = palette(i % 10)
            try:
                if mt in (ModelType.RANDOM_FOREST, ModelType.NEURAL_NET):
                    if fr.model and self.results_df is not None:
                        med = {name: float(self.results_df[name].median(skipna=True)) if name in self.results_df else 0.0 for name in self.feature_extractor.feature_names}
                        feats = [[med[k] for k in self.feature_extractor.feature_names] for _ in n_fit]
                        y_fit = fr.model.predict(np.array(feats, dtype=np.float64))
                else:
                    y_fit = fr.model.predict(n_fit.reshape(-1, 1))
                ax.plot(n_fit, y_fit, "-", color=color, linewidth=2, label=f"{mt.name} (R²={fr.r_squared:.3f})")
            except Exception as e:
                logger.error(f"Plot curve failed for {mt.name}: {e}")

        ax.set_xlabel("Exponent (n)")
        ax.set_ylabel(f"T({self.base}^n)")
        ax.set_title(f"Collatz Stopping Times for {self.base}^n • Variant: {self.variant.name} • Time: {self.computation_time:.2f}s")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        plt.tight_layout()
        return fig

    def plot_feature_importance(self) -> Optional[plt.Figure]:
        # Aggregate feature importances if available
        rows = []
        for mt, fr in self.fit_results.items():
            if fr.feature_importances is not None:
                names = self.feature_extractor.feature_names
                for i, imp in enumerate(fr.feature_importances):
                    if i < len(names):
                        rows.append({"model": mt.name, "feature": names[i], "importance": float(imp)})
        if not rows:
            return None
        df = pd.DataFrame(rows)
        # Normalize per model
        df["importance"] = df.groupby("model")["importance"].transform(lambda s: s / (s.sum() if s.sum() > 0 else 1.0))
        # Average importances and sort
        order = df.groupby("feature")["importance"].mean().sort_values().index.tolist()
        df["feature"] = pd.Categorical(df["feature"], categories=order, ordered=True)
        fig, ax = plt.subplots(figsize=(12, 10))
        # Stacked bars per model grouping
        models = sorted(df["model"].unique())
        ypos = np.arange(len(order))
        width = 0.8 / max(1, len(models))
        for i, m in enumerate(models):
            sub = df[df["model"] == m].sort_values("feature")
            ax.barh(ypos + i * width, sub["importance"].values, height=width, label=m)
        ax.set_yticks(ypos + width * (len(models) - 1) / 2)
        ax.set_yticklabels(order)
        ax.set_xlabel("Normalized Importance")
        ax.set_title("Feature Importances (by model)")
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        return fig

    def plot_model_comparison(self) -> Optional[plt.Figure]:
        if not self.fit_results:
            return None
        metrics = ["r_squared", "aic", "bic"]
        names = [mt.name for mt in self.fit_results]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for i, met in enumerate(metrics):
            vals = [getattr(fr, met) for fr in self.fit_results.values()]
            axes[i].barh(names, vals, color="#88c", alpha=0.85)
            axes[i].set_title(met.upper() if met in ("aic", "bic") else met.replace("_", " ").title())
            axes[i].grid(True, alpha=0.3)
        fig.suptitle("Model Comparison")
        plt.tight_layout()
        return fig

    def plot_3d_visualization(self) -> Optional[plt.Figure]:
        if self.results_df is None:
            return None
        need = {"exponent", "max_value", "stopping_time"}
        if not need.issubset(self.results_df.columns):
            return None
        dat = self.results_df.dropna(subset=list(need))
        if len(dat) < 10:
            return None
        x = dat["exponent"].astype(float).values
        y = np.log1p(dat["max_value"].astype(float).values)
        z = dat["stopping_time"].astype(float).values
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(x, y, z, c=z, cmap="viridis", s=40, alpha=0.85, edgecolors="k", linewidths=0.2)
        cb = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.9)
        cb.set_label("Stopping Time", rotation=270, labelpad=15)
        ax.set_xlabel("Exponent (n)")
        ax.set_ylabel("log(1 + max_value)")
        ax.set_zlabel(f"T({self.base}^n)")
        ax.set_title("3D View: n vs log(max_value) vs stopping time")
        ax.view_init(elev=22, azim=35)
        plt.tight_layout()
        return fig

    # -------------------- Interactive (Plotly) --------------------
    def create_interactive_plot(self):
        if self.results_df is None or "stopping_time" not in self.results_df.columns:
            return None
        valid = self.results_df.dropna(subset=["stopping_time"])
        if len(valid) < 3:
            return None
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Stopping Times", "Max Values", "Stopping vs Exponent", "Model R²"),
            specs=[[{"type": "scatter"}, {"type": "scatter"}], [{"type": "scatter"}, {"type": "bar"}]],
        )
        # Data scatter
        fig.add_scatter(
            x=valid["exponent"],
            y=valid["stopping_time"],
            mode="markers",
            name="Data",
            marker=dict(color=valid["stopping_time"], colorscale="Viridis", size=7, showscale=True, colorbar=dict(title="T")),
            text=[f"{self.base}^{int(n)} = {val:.1e}<br>T={t}" for n, val, t in zip(valid["exponent"], valid["value"], valid["stopping_time"])],
            hoverinfo="text",
            row=1,
            col=1,
        )
        if "max_value" in valid.columns:
            fig.add_scatter(
                x=valid["exponent"], y=valid["max_value"], mode="markers", name="Max Value", marker=dict(size=6), row=1, col=2
            )
        # Re-plot scatter
        fig.add_scatter(x=valid["exponent"], y=valid["stopping_time"], mode="markers", name="Stopping", marker=dict(size=6), row=2, col=1)
        # Model R²
        if self.fit_results:
            names = [m.name for m in self.fit_results.keys()]
            r2s = [r.r_squared for r in self.fit_results.values()]
            fig.add_bar(x=names, y=r2s, name="R²", row=2, col=2)
        fig.update_layout(
            title=f"Collatz Analysis for {self.base}^n • n={valid['exponent'].min()}..{valid['exponent'].max()}",
            height=800,
            showlegend=True,
            hovermode="closest",
        )
        fig.update_xaxes(title_text="Exponent (n)", row=1, col=1)
        fig.update_yaxes(title_text="Stopping Time", row=1, col=1)
        fig.update_xaxes(title_text="Exponent (n)", row=1, col=2)
        fig.update_yaxes(title_text="Max Value", row=1, col=2)
        fig.update_xaxes(title_text="Exponent (n)", row=2, col=1)
        fig.update_yaxes(title_text="Stopping Time", row=2, col=1)
        fig.update_xaxes(title_text="Model", row=2, col=2)
        fig.update_yaxes(title_text="R²", row=2, col=2)
        return fig

    # -------------------- Save Outputs --------------------
    def save_results(self) -> None:
        out = self.output_dir
        plots_dir = out / "plots"
        data_dir = out / "data"
        models_dir = out / "models"
        for d in (plots_dir, data_dir, models_dir):
            d.mkdir(exist_ok=True, parents=True)

        if self.results_df is not None:
            csv_path = data_dir / "collatz_results.csv"
            self.results_df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV: {csv_path}")
            json_path = data_dir / "collatz_results.json"
            self.results_df.dropna(subset=["stopping_time"]).to_json(json_path, orient="records")
            logger.info(f"Saved JSON: {json_path}")

        if self.fit_results:
            models_data = []
            for mt, fr in self.fit_results.items():
                md = fr.to_dict()
                if mt in (ModelType.RANDOM_FOREST, ModelType.NEURAL_NET) and fr.model:
                    import joblib

                    p = models_dir / f"{mt.name.lower()}_model.joblib"
                    joblib.dump(fr.model, p)
                    md["model_path"] = str(p)
                models_data.append(md)
            with open(models_dir / "fitted_models.json", "w") as fh:
                json.dump(models_data, fh, indent=2)
            logger.info(f"Saved model metadata: {models_dir / 'fitted_models.json'}")

        plots = {
            "main_results": self.plot_main_results(),
            "feature_importance": self.plot_feature_importance(),
            "model_comparison": self.plot_model_comparison(),
            "3d_visualization": self.plot_3d_visualization(),
        }
        for name, fig in plots.items():
            if fig is not None:
                p = plots_dir / f"{name}.png"
                fig.savefig(p, dpi=300, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"Saved plot: {p}")

        inter = self.create_interactive_plot()
        if inter:
            html_path = plots_dir / "interactive_plot.html"
            inter.write_html(html_path)
            logger.info(f"Saved interactive plot: {html_path}")

        self.save_analysis_notebook()
        exporter = NarrativeExporter(self)
        summary = exporter.generate_summary()
        exporter.save_summary()
        logger.info("Results saved successfully")

    def save_analysis_notebook(self) -> None:
        import nbformat
        from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

        nb = new_notebook()
        nb.cells.append(new_markdown_cell("# Collatz Conjecture Analysis\nAutomated notebook generated by the analyzer."))
        nb.cells.append(
            new_code_cell(
                """import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
results = pd.read_csv('data/collatz_results.csv')
results.head()"""
            )
        )
        nb.cells.append(
            new_code_cell(
                """fig, ax = plt.subplots(figsize=(12,6))
ax.scatter(results['exponent'], results['stopping_time'], c=results['stopping_time'], cmap='viridis')
ax.set_xlabel('Exponent (n)'); ax.set_ylabel('Stopping Time'); ax.set_title('Collatz Stopping Times')
plt.colorbar(label='T'); plt.show()"""
            )
        )
        path = self.output_dir / "analysis_notebook.ipynb"
        with open(path, "w") as fh:
            nbformat.write(nb, fh)
        logger.info(f"Saved notebook: {path}")

    # -------------------- Pipeline --------------------
    def run_analysis(self, start: int, end: int, model_types: Optional[List[ModelType]] = None) -> None:
        if start > end or start <= 0:
            raise ValueError("Invalid range: start ≤ end and start > 0 required")
        logger.info(f"Run analysis: {self.base}^{start} .. {self.base}^{end}")
        self.n_values = np.arange(start, end + 1)
        self.T_values = self.parallel_compute(self.n_values)
        self.fit_models(model_types)
        self.create_results_dataframe()
        self.save_results()

# ============================================================
# Narrative Exporter (no LLM)
# ============================================================
class NarrativeExporter:
    def __init__(self, analyzer: CollatzAnalyzer):
        self.analyzer = analyzer
        self.summary = ""
        self.sections: List[Tuple[str, str]] = []

    def add_section(self, title: str, content: str):
        self.sections.append((title, content))

    def generate_summary(self) -> str:
        self._basic_summary()
        self._model_comparison()
        self._feature_analysis()
        self.summary = "# Collatz Conjecture Analysis Report\n\n"
        for t, c in self.sections:
            self.summary += f"## {t}\n\n{c}\n\n"
        return self.summary

    def _basic_summary(self):
        df = self.analyzer.results_df
        if df is None:
            self.add_section("Error", "No results available.")
            return
        valid = df.dropna(subset=["stopping_time"])
        content = (
            f"**Base:** {self.analyzer.base}\n\n"
            f"**Variant:** {self.analyzer.variant.name}\n\n"
            f"**Range:** {self.analyzer.base}^{int(valid['exponent'].min())} to {self.analyzer.base}^{int(valid['exponent'].max())}\n\n"
            f"**Computation time:** {self.analyzer.computation_time:.2f} seconds\n\n"
            "### Stopping Time Statistics\n\n"
            f"- Min: {int(valid['stopping_time'].min())}\n"
            f"- Max: {int(valid['stopping_time'].max())}\n"
            f"- Mean: {valid['stopping_time'].mean():.2f}\n"
            f"- Median: {valid['stopping_time'].median():.2f}\n"
            f"- Std Dev: {valid['stopping_time'].std():.2f}\n"
        )
        if "max_value" in valid.columns:
            content += (
                "\n### Maximum Value Statistics\n\n"
                f"- Min: {valid['max_value'].min():.2e}\n"
                f"- Max: {valid['max_value'].max():.2e}\n"
                f"- Mean: {valid['max_value'].mean():.2e}\n"
            )
        self.add_section("Summary Statistics", content)

    def _model_comparison(self):
        if not self.analyzer.fit_results:
            return
        rows = "| Model | R² | Adjusted R² | AIC | BIC | Time (s) |\n|---|---:|---:|---:|---:|---:|\n"
        for mt, fr in self.analyzer.fit_results.items():
            rows += f"| {mt.name} | {fr.r_squared:.4f} | {fr.adjusted_r_squared:.4f} | {fr.aic:.1f} | {fr.bic:.1f} | {fr.fit_time:.2f} |\n"
        best = max(self.analyzer.fit_results.items(), key=lambda kv: kv[1].r_squared)
        rows += f"\n**Best model:** {best[0].name} (R²={best[1].r_squared:.4f})\n"
        self.add_section("Model Comparison", rows)

    def _feature_analysis(self):
        if not any(fr.feature_importances is not None for fr in self.analyzer.fit_results.values()):
            return
        best = max((kv for kv in self.analyzer.fit_results.items() if kv[1].feature_importances is not None), key=lambda kv: kv[1].r_squared)
        names = self.analyzer.feature_extractor.feature_names
        imps = best[1].feature_importances
        order = np.argsort(imps)[::-1]
        top = "| Feature | Importance |\n|---|---:|\n" + "\n".join(
            f"| {names[i]} | {imps[i]:.4f} |" for i in order[:10]
        )
        self.add_section("Feature Importance (Top 10)", top)

    def save_summary(self, filename: str = "collatz_report.md") -> None:
        p = self.analyzer.output_dir / filename
        with open(p, "w") as fh:
            fh.write(self.summary)
        logger.info(f"Saved summary: {p}")

# ============================================================
# CLI
# ============================================================
def main():
    print("\nCollatz Conjecture Advanced Analyzer (Matplotlib + Plotly)")
    print("----------------------------------------------------------\n")
    try:
        base = int(input("Enter base value (default 2): ") or 2)
        start = int(input("Enter start exponent: "))
        end = int(input("Enter end exponent: "))

        print("\nVariants:")
        for i, v in enumerate(CollatzVariant):
            print(f"{i+1}. {v.name}")
        v_choice = input("Select variant (default 1): ") or "1"
        try:
            v_idx = int(v_choice) - 1
            variant = list(CollatzVariant)[v_idx]
        except Exception:
            variant = CollatzVariant.CLASSIC

        v_params: Dict[str, Any] = {}
        if variant == CollatzVariant.GENERALIZED:
            print("\nEnter generalized parameters:")
            v_params["p"] = int(input("Odd multiplier p (default 3): ") or 3)
            v_params["q"] = int(input("Odd adder q (default 1): ") or 1)
            v_params["d"] = int(input("Even divider d (default 2): ") or 2)

        print("\nAvailable models:")
        for i, m in enumerate(MODEL_REGISTRY.keys()):
            print(f"{i+1}. {m.name}")
        msel = input("Select models to fit (comma-separated, default=all): ") or "all"
        if msel.strip().lower() == "all":
            model_types = None
        else:
            model_types = []
            for ch in msel.split(","):
                try:
                    idx = int(ch.strip()) - 1
                    model_types.append(list(MODEL_REGISTRY.keys())[idx])
                except Exception:
                    pass

        out_dir = input("Enter output directory (default=collatz_results): ") or None

        analyzer = CollatzAnalyzer(base=base, output_dir=out_dir, variant=variant, variant_params=v_params)
        analyzer.run_analysis(start, end, model_types)

        print("\n✅ Analysis completed.")
        print(f"Results saved to: {analyzer.output_dir}")
        print("• data/  → CSV & JSON")
        print("• plots/ → PNGs & interactive HTML")
        print("• models/→ model artifacts & metadata")
        print("• analysis_notebook.ipynb, collatz_report.md")

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

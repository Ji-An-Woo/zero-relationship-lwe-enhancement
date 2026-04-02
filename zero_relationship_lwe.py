from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


def clip(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0

    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)

    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x_val, y_val in zip(xs, ys):
        dx = x_val - mean_x
        dy = y_val - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy

    denom = math.sqrt(den_x * den_y)
    if denom == 0.0:
        return 0.0
    return num / denom


def shannon_entropy(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    entropy = 0.0
    for count in counts.values():
        prob = count / total
        entropy -= prob * math.log2(prob)
    return entropy


def roc_auc_binary(labels: Sequence[int], scores: Sequence[float]) -> float:
    if len(labels) != len(scores) or not labels:
        return 0.5

    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    n = len(pairs)
    n_pos = sum(label for _, label in pairs)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    rank = 1
    pos_rank_sum = 0.0
    idx = 0
    while idx < n:
        tie_end = idx
        while tie_end + 1 < n and pairs[tie_end + 1][0] == pairs[idx][0]:
            tie_end += 1

        tie_size = tie_end - idx + 1
        avg_rank = rank + (tie_size - 1) / 2
        pos_in_tie = sum(label for _, label in pairs[idx : tie_end + 1])
        pos_rank_sum += avg_rank * pos_in_tie

        rank += tie_size
        idx = tie_end + 1

    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def sample_discrete_gaussian(sigma: float, bound: int, rng: random.Random) -> int:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if bound < 1:
        raise ValueError("bound must be >= 1")

    while True:
        candidate = int(round(rng.gauss(0.0, sigma)))
        if abs(candidate) <= bound:
            return candidate


def truncated_maclaurin(family: str, x: float, order: int) -> float:
    if order < 0:
        raise ValueError("order must be >= 0")

    if family == "sin":
        term = x
        result = term
        for k in range(1, order + 1):
            term *= -x * x / ((2 * k) * (2 * k + 1))
            result += term
        return result

    if family == "cos":
        term = 1.0
        result = term
        for k in range(1, order + 1):
            term *= -x * x / ((2 * k - 1) * (2 * k))
            result += term
        return result

    if family == "exp":
        term = 1.0
        result = term
        for k in range(1, order + 1):
            term *= x / k
            result += term
        return result

    if family == "atan":
        result = 0.0
        for k in range(order + 1):
            sign = -1.0 if (k % 2 == 1) else 1.0
            result += sign * (x ** (2 * k + 1)) / (2 * k + 1)
        return result

    if family == "sinh":
        term = x
        result = term
        for k in range(1, order + 1):
            term *= x * x / ((2 * k) * (2 * k + 1))
            result += term
        return result

    raise ValueError(f"unsupported family: {family}")


@dataclass
class BaselineGaussianSecretSampler:
    sigma: float = 2.0
    secret_bound: int = 8
    rng: random.Random = field(default_factory=random.Random)

    def sample_secret_vector(self, dimension: int) -> List[int]:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        return [
            sample_discrete_gaussian(self.sigma, self.secret_bound, self.rng)
            for _ in range(dimension)
        ]


@dataclass
class RandomMaclaurinFunction:
    family: str
    order: int
    alpha: float
    beta: float
    amplitude: float
    bias: float

    def evaluate(self, x: float) -> float:
        transformed = self.alpha * x + self.beta

        # Keep transformed input in stable ranges for truncated series.
        if self.family == "atan":
            transformed = max(-0.95, min(0.95, transformed))
        elif self.family == "exp":
            transformed = max(-3.0, min(3.0, transformed))
        else:
            transformed = max(-2.5, min(2.5, transformed))

        base = truncated_maclaurin(self.family, transformed, self.order)
        base_at_zero = truncated_maclaurin(self.family, self.beta, self.order)
        return self.amplitude * (base - base_at_zero) + self.bias


@dataclass
class Segment:
    y0: float
    y1: float
    weight: float


@dataclass
class ZeroRelationshipSecretSampler:
    num_functions: int = 7
    points_per_function: int = 8
    order_min: int = 3
    order_max: int = 8
    x_min: float = -1.25
    x_max: float = 1.25
    output_gain: float = 0.52
    warp_strength: float = 0.16
    jitter_sigma: float = 0.42
    zero_to_nonzero_prob: float = 0.16
    unit_to_zero_prob: float = 0.24
    boundary_mix_prob: float = 0.42
    magnitude_scale: float = 0.72
    high_mag_shrink_prob: float = 0.58
    intra_vector_refresh_prob: float = 0.40
    secret_bound: int = 8
    rng: random.Random = field(default_factory=random.Random)

    def _new_random_function(self) -> RandomMaclaurinFunction:
        family = self.rng.choice(["sin", "cos", "exp", "atan", "sinh"])
        order = self.rng.randint(self.order_min, self.order_max)

        alpha = self.rng.uniform(0.7, 2.1)
        if self.rng.random() < 0.5:
            alpha *= -1.0

        beta = self.rng.uniform(-0.7, 0.7)
        amplitude = self.rng.uniform(0.45, 1.30)
        if self.rng.random() < 0.5:
            amplitude *= -1.0
        bias = self.rng.uniform(-0.15, 0.15)
        return RandomMaclaurinFunction(
            family=family,
            order=order,
            alpha=alpha,
            beta=beta,
            amplitude=amplitude,
            bias=bias,
        )

    def _build_space(self) -> Tuple[List[RandomMaclaurinFunction], List[Segment]]:
        if self.num_functions < 1:
            raise ValueError("num_functions must be >= 1")
        if self.points_per_function < 2:
            raise ValueError("points_per_function must be >= 2")

        functions: List[RandomMaclaurinFunction] = []
        segments: List[Segment] = []

        for _ in range(self.num_functions):
            func = self._new_random_function()
            xs = sorted(
                self.rng.uniform(self.x_min, self.x_max)
                for _ in range(self.points_per_function)
            )
            ys = [func.evaluate(x) + self.rng.gauss(0.0, 0.05) for x in xs]
            center = statistics.fmean(ys)
            ys = [y - center for y in ys]

            functions.append(func)

            for idx in range(len(ys) - 1):
                slope = ys[idx + 1] - ys[idx]
                midpoint = 0.5 * (ys[idx + 1] + ys[idx])
                weight = (1.0 / (1.0 + abs(midpoint))) + 0.25 * abs(slope) + 0.05
                segments.append(Segment(ys[idx], ys[idx + 1], weight))

        if not segments:
            raise RuntimeError("failed to create piecewise segments")

        return functions, segments

    def _pick_segment(self, segments: Sequence[Segment]) -> Segment:
        total_weight = sum(seg.weight for seg in segments)
        if total_weight <= 0.0:
            return self.rng.choice(list(segments))

        threshold = self.rng.uniform(0.0, total_weight)
        acc = 0.0
        for segment in segments:
            acc += segment.weight
            if acc >= threshold:
                return segment
        return segments[-1]

    def _sample_one_coefficient(
        self,
        functions: Sequence[RandomMaclaurinFunction],
        segments: Sequence[Segment],
    ) -> int:
        segment_a = self._pick_segment(segments)
        segment_b = self._pick_segment(segments)
        t_a = self.rng.random()
        t_b = self.rng.random()
        base_a = segment_a.y0 + t_a * (segment_a.y1 - segment_a.y0)
        base_b = segment_b.y0 + t_b * (segment_b.y1 - segment_b.y0)

        # Difference mixing keeps the center near zero and reduces repeated structure.
        base = 0.5 * (base_a - base_b)

        warp_func_1 = self.rng.choice(functions)
        warp_func_2 = self.rng.choice(functions)
        probe_1 = self.rng.uniform(self.x_min, self.x_max)
        probe_2 = self.rng.uniform(self.x_min, self.x_max)
        warp = math.tanh(warp_func_1.evaluate(probe_1)) - math.tanh(
            warp_func_2.evaluate(probe_2)
        )

        local_mix = self.rng.uniform(0.7, 1.3)
        raw = (
            local_mix * base
            + self.warp_strength * warp
            + self.rng.gauss(0.0, self.jitter_sigma)
        )
        squashed = math.tanh(self.output_gain * raw)

        scaled = squashed * self.secret_bound * self.magnitude_scale
        value = int(round(scaled))
        value = clip(value, -self.secret_bound, self.secret_bound)

        # Symmetric boundary mixing blurs zero/non-zero classification.
        if value == 0 and self.rng.random() < self.zero_to_nonzero_prob:
            value = -1 if self.rng.random() < 0.5 else 1
        elif abs(value) == 1 and self.rng.random() < self.unit_to_zero_prob:
            value = 0

        if abs(value) <= 1 and self.rng.random() < self.boundary_mix_prob:
            value = self.rng.choice([-1, 0, 1])

        if abs(value) >= 3 and self.rng.random() < self.high_mag_shrink_prob:
            value = int(math.copysign(abs(value) - 2, value))

        if self.rng.random() < 0.5:
            value = -value

        return value

    def sample_secret_vector(self, dimension: int) -> List[int]:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")

        # Reset all functions and piecewise intervals per secret generation.
        functions, segments = self._build_space()
        vector: List[int] = []
        for idx in range(dimension):
            if idx > 0 and self.rng.random() < self.intra_vector_refresh_prob:
                functions, segments = self._build_space()
            vector.append(
                self._sample_one_coefficient(functions=functions, segments=segments)
            )
        return vector


def sample_lwe_instance(
    n: int,
    m: int,
    q: int,
    secret_sampler,
    error_sigma: float,
    rng: random.Random,
) -> Tuple[List[List[int]], List[int], List[int], List[int]]:
    if n < 1 or m < 1:
        raise ValueError("n and m must be >= 1")
    if q < 3:
        raise ValueError("q must be >= 3")

    secret = secret_sampler.sample_secret_vector(n)
    matrix_a = [[rng.randrange(q) for _ in range(n)] for _ in range(m)]

    error_bound = max(1, int(math.ceil(8 * error_sigma)))
    error = [sample_discrete_gaussian(error_sigma, error_bound, rng) for _ in range(m)]

    vector_b = []
    for row, err in zip(matrix_a, error):
        value = sum(a_ij * s_j for a_ij, s_j in zip(row, secret)) + err
        vector_b.append(value % q)

    return matrix_a, secret, error, vector_b


def evaluate_secret_sampler(
    sampler_name: str,
    sampler,
    dimension: int,
    num_vectors: int,
    observation_noise_sigma: float,
    rng: random.Random,
) -> Dict[str, float]:
    vectors = [sampler.sample_secret_vector(dimension) for _ in range(num_vectors)]
    coeffs = [value for vector in vectors for value in vector]

    if not coeffs:
        raise RuntimeError("no coefficients sampled")

    mean_val = statistics.fmean(coeffs)
    std_val = statistics.pstdev(coeffs)
    zero_ratio = sum(1 for value in coeffs if value == 0) / len(coeffs)

    lag_x: List[float] = []
    lag_y: List[float] = []
    for vector in vectors:
        if len(vector) >= 2:
            lag_x.extend(vector[:-1])
            lag_y.extend(vector[1:])
    lag1_corr = pearson_corr(lag_x, lag_y) if lag_x else 0.0

    dim_probe = min(dimension, 16)
    columns = [[vector[idx] for vector in vectors] for idx in range(dim_probe)]
    pair_corrs: List[float] = []
    for i in range(dim_probe):
        for j in range(i + 1, dim_probe):
            pair_corrs.append(abs(pearson_corr(columns[i], columns[j])))
    avg_abs_pair_corr = statistics.fmean(pair_corrs) if pair_corrs else 0.0

    observed = [value + rng.gauss(0.0, observation_noise_sigma) for value in coeffs]
    labels_nonzero = [1 if value != 0 else 0 for value in coeffs]
    scores = [abs(obs) for obs in observed]
    zero_nonzero_auc = roc_auc_binary(labels_nonzero, scores)

    zero_scores = [score for score, label in zip(scores, labels_nonzero) if label == 0]
    nonzero_scores = [score for score, label in zip(scores, labels_nonzero) if label == 1]
    mean_gap = 0.0
    if zero_scores and nonzero_scores:
        mean_gap = abs(statistics.fmean(nonzero_scores) - statistics.fmean(zero_scores))

    return {
        "name": sampler_name,
        "mean": mean_val,
        "std": std_val,
        "entropy": shannon_entropy(coeffs),
        "zero_ratio": zero_ratio,
        "lag1_corr": lag1_corr,
        "avg_abs_dim_corr": avg_abs_pair_corr,
        "zero_nonzero_auc": zero_nonzero_auc,
        "zero_nonzero_score_gap": mean_gap,
    }


def print_comparison(base_metrics: Dict[str, float], zero_metrics: Dict[str, float]) -> None:
    print("\n=== Secret Sampler Comparison ===")
    print(f"baseline: {base_metrics['name']}")
    print(f"candidate: {zero_metrics['name']}")
    print("")
    print("metric                    baseline        zero_relationship")
    print("-----------------------------------------------------------")

    metric_keys = [
        "mean",
        "std",
        "entropy",
        "zero_ratio",
        "lag1_corr",
        "avg_abs_dim_corr",
        "zero_nonzero_auc",
        "zero_nonzero_score_gap",
    ]

    for key in metric_keys:
        print(f"{key:<24}{base_metrics[key]:>12.6f}{zero_metrics[key]:>20.6f}")

    print("\nInterpretation hints:")
    print("- Lower |lag1_corr| and lower avg_abs_dim_corr generally mean weaker linear correlation.")
    print("- zero_nonzero_auc closer to 0.5 means noisier zero/non-zero separability.")
    print("- entropy is over integer coefficients; higher can indicate richer spread.")


def aggregate_metrics(name: str, metric_rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not metric_rows:
        raise ValueError("metric_rows must not be empty")

    keys = [key for key in metric_rows[0].keys() if key != "name"]
    aggregated: Dict[str, float] = {"name": name}
    for key in keys:
        aggregated[key] = statistics.fmean(row[key] for row in metric_rows)
    return aggregated


def metrics_dispersion(metric_rows: Sequence[Dict[str, float]], key: str) -> float:
    if len(metric_rows) < 2:
        return 0.0
    values = [row[key] for row in metric_rows]
    return statistics.pstdev(values)


def write_metrics_csv(
    file_path: str,
    baseline_metrics: Dict[str, float],
    zero_metrics: Dict[str, float],
    baseline_rows: Sequence[Dict[str, float]],
    zero_rows: Sequence[Dict[str, float]],
    seed_start: int,
) -> None:
    metric_keys = [
        "mean",
        "std",
        "entropy",
        "zero_ratio",
        "lag1_corr",
        "avg_abs_dim_corr",
        "zero_nonzero_auc",
        "zero_nonzero_score_gap",
    ]

    with open(file_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "row_type",
                "seed",
                "sampler",
                *metric_keys,
            ]
        )

        writer.writerow(
            [
                "aggregate",
                "",
                "small_gaussian",
                *[baseline_metrics[key] for key in metric_keys],
            ]
        )
        writer.writerow(
            [
                "aggregate",
                "",
                "zero_relationship",
                *[zero_metrics[key] for key in metric_keys],
            ]
        )

        for idx, row in enumerate(baseline_rows):
            writer.writerow(
                [
                    "seed",
                    seed_start + idx,
                    "small_gaussian",
                    *[row[key] for key in metric_keys],
                ]
            )
        for idx, row in enumerate(zero_rows):
            writer.writerow(
                [
                    "seed",
                    seed_start + idx,
                    "zero_relationship",
                    *[row[key] for key in metric_keys],
                ]
            )

        if len(baseline_rows) > 1:
            writer.writerow(
                [
                    "dispersion",
                    "",
                    "small_gaussian_std",
                    *[metrics_dispersion(baseline_rows, key) for key in metric_keys],
                ]
            )
            writer.writerow(
                [
                    "dispersion",
                    "",
                    "zero_relationship_std",
                    *[metrics_dispersion(zero_rows, key) for key in metric_keys],
                ]
            )


def zero_relationship_profile(profile_name: str) -> Dict[str, float]:
    if profile_name == "default":
        return {}

    if profile_name == "balanced":
        return {
            "output_gain": 0.50,
            "warp_strength": 0.15,
            "jitter_sigma": 0.50,
            "zero_to_nonzero_prob": 0.22,
            "unit_to_zero_prob": 0.32,
            "boundary_mix_prob": 0.56,
            "magnitude_scale": 0.68,
            "high_mag_shrink_prob": 0.68,
            "intra_vector_refresh_prob": 0.55,
        }

    if profile_name == "blur_heavy":
        return {
            "output_gain": 0.45,
            "warp_strength": 0.12,
            "jitter_sigma": 0.62,
            "zero_to_nonzero_prob": 0.28,
            "unit_to_zero_prob": 0.40,
            "boundary_mix_prob": 0.66,
            "magnitude_scale": 0.58,
            "high_mag_shrink_prob": 0.78,
            "intra_vector_refresh_prob": 0.65,
        }

    raise ValueError(f"unsupported profile: {profile_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prototype implementation for Zero_relationship secret sampling in LWE"
    )

    parser.add_argument("--seed", type=int, default=20260403)
    parser.add_argument(
        "--benchmark-seeds",
        type=int,
        default=1,
        help="Number of consecutive seeds to average for comparison",
    )
    parser.add_argument("--n", type=int, default=256, help="LWE secret dimension")
    parser.add_argument("--m", type=int, default=256, help="LWE number of equations")
    parser.add_argument("--q", type=int, default=12289, help="LWE modulus")

    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of secret vectors used for distribution analysis",
    )
    parser.add_argument(
        "--obs-noise-sigma",
        type=float,
        default=1.0,
        help="Noise level used in zero/non-zero separability check",
    )

    parser.add_argument("--baseline-sigma", type=float, default=2.0)
    parser.add_argument("--error-sigma", type=float, default=2.8)
    parser.add_argument("--secret-bound", type=int, default=8)

    parser.add_argument("--zr-functions", type=int, default=7)
    parser.add_argument("--zr-points", type=int, default=8)
    parser.add_argument("--zr-order-min", type=int, default=3)
    parser.add_argument("--zr-order-max", type=int, default=8)
    parser.add_argument(
        "--zr-profile",
        choices=["default", "balanced", "blur_heavy"],
        default="default",
        help="Preset parameter profile for Zero_relationship sampler",
    )
    parser.add_argument("--zr-zero-to-nonzero", type=float, default=0.16)
    parser.add_argument("--zr-unit-to-zero", type=float, default=0.24)
    parser.add_argument("--zr-boundary-mix", type=float, default=0.42)
    parser.add_argument("--zr-refresh-prob", type=float, default=0.40)

    parser.add_argument(
        "--show-lwe-preview",
        action="store_true",
        help="Print a small preview of one LWE sample generated with Zero_relationship",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="",
        help="Optional path to save benchmark metrics as CSV",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.benchmark_seeds < 1:
        raise ValueError("benchmark-seeds must be >= 1")

    profile_overrides = zero_relationship_profile(args.zr_profile)
    baseline_rows: List[Dict[str, float]] = []
    zero_rows: List[Dict[str, float]] = []

    for seed_offset in range(args.benchmark_seeds):
        seed_i = args.seed + seed_offset

        baseline = BaselineGaussianSecretSampler(
            sigma=args.baseline_sigma,
            secret_bound=args.secret_bound,
            rng=random.Random(seed_i + 1),
        )

        zero_kwargs = {
            "num_functions": args.zr_functions,
            "points_per_function": args.zr_points,
            "order_min": args.zr_order_min,
            "order_max": args.zr_order_max,
            "zero_to_nonzero_prob": args.zr_zero_to_nonzero,
            "unit_to_zero_prob": args.zr_unit_to_zero,
            "boundary_mix_prob": args.zr_boundary_mix,
            "intra_vector_refresh_prob": args.zr_refresh_prob,
            "secret_bound": args.secret_bound,
            "rng": random.Random(seed_i + 2),
        }
        zero_kwargs.update(profile_overrides)
        zero_relationship = ZeroRelationshipSecretSampler(**zero_kwargs)

        baseline_rows.append(
            evaluate_secret_sampler(
                sampler_name="small_gaussian",
                sampler=baseline,
                dimension=args.n,
                num_vectors=args.samples,
                observation_noise_sigma=args.obs_noise_sigma,
                rng=random.Random(seed_i + 101),
            )
        )

        zero_rows.append(
            evaluate_secret_sampler(
                sampler_name="zero_relationship",
                sampler=zero_relationship,
                dimension=args.n,
                num_vectors=args.samples,
                observation_noise_sigma=args.obs_noise_sigma,
                rng=random.Random(seed_i + 102),
            )
        )

    if args.benchmark_seeds == 1:
        baseline_metrics = baseline_rows[0]
        zero_metrics = zero_rows[0]
    else:
        baseline_metrics = aggregate_metrics("small_gaussian_avg", baseline_rows)
        zero_metrics = aggregate_metrics("zero_relationship_avg", zero_rows)

    print_comparison(baseline_metrics, zero_metrics)

    if args.benchmark_seeds > 1:
        print("\n=== Multi-seed dispersion (std across seeds) ===")
        for key in [
            "zero_nonzero_auc",
            "zero_nonzero_score_gap",
            "avg_abs_dim_corr",
            "entropy",
        ]:
            b_std = metrics_dispersion(baseline_rows, key)
            z_std = metrics_dispersion(zero_rows, key)
            print(f"{key:<24}{b_std:>12.6f}{z_std:>20.6f}")

    if args.csv_output:
        write_metrics_csv(
            file_path=args.csv_output,
            baseline_metrics=baseline_metrics,
            zero_metrics=zero_metrics,
            baseline_rows=baseline_rows,
            zero_rows=zero_rows,
            seed_start=args.seed,
        )
        print(f"\nCSV saved: {args.csv_output}")

    if args.show_lwe_preview:
        lwe_rng = random.Random(args.seed + 200)
        preview_zero_kwargs = {
            "num_functions": args.zr_functions,
            "points_per_function": args.zr_points,
            "order_min": args.zr_order_min,
            "order_max": args.zr_order_max,
            "zero_to_nonzero_prob": args.zr_zero_to_nonzero,
            "unit_to_zero_prob": args.zr_unit_to_zero,
            "boundary_mix_prob": args.zr_boundary_mix,
            "intra_vector_refresh_prob": args.zr_refresh_prob,
            "secret_bound": args.secret_bound,
            "rng": random.Random(args.seed + 2),
        }
        preview_zero_kwargs.update(profile_overrides)
        zero_preview_sampler = ZeroRelationshipSecretSampler(**preview_zero_kwargs)

        matrix_a, secret, error, vector_b = sample_lwe_instance(
            n=args.n,
            m=args.m,
            q=args.q,
            secret_sampler=zero_preview_sampler,
            error_sigma=args.error_sigma,
            rng=lwe_rng,
        )

        print("\n=== One Zero_relationship LWE sample preview ===")
        print(f"A[0][:12] = {matrix_a[0][:12]}")
        print(f"s[:24]    = {secret[:24]}")
        print(f"e[:24]    = {error[:24]}")
        print(f"b[:24]    = {vector_b[:24]}")


if __name__ == "__main__":
    main()

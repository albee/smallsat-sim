from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from smallsat_sim.envs.perturbation_state import (
    PerturbationStatus,
    _stable_cholesky,
)


class ThrusterFailureSimulator:
    """
    This class is used to the data for the nonlinear perturbations.
    It does the following things (in sequence):
        -
    """

    def __init__(
        self,
        num_points=100,
        upper_bound=0.6,
        subset_size=40,
        valve_min=0.1,
        valve_max=0.5,
        failure_mode_overrides: Optional[dict] = None,
    ):
        self.num_points = num_points
        self.upper_bound = upper_bound
        self.subset_size = subset_size
        self.valve_min = valve_min
        self.valve_max = valve_max
        self.demanded_force = jnp.linspace(
            0, upper_bound, num_points
        )  # Values between 0 and upper_bound
        self.failure_modes = self._apply_overrides(
            self._get_failure_modes(), failure_mode_overrides
        )

    @dataclass
    class _GPPosterior:
        """Lightweight Gaussian distribution wrapper compatible with the old API."""

        loc: jnp.ndarray
        scale_tril: jnp.ndarray

        def mean(self) -> jnp.ndarray:
            return self.loc

        def sample(self, key) -> jnp.ndarray:
            if key is None:
                raise ValueError(
                    "A PRNGKey must be provided when sampling from the GP posterior."
                )
            normal = jax.random.normal(key, shape=self.loc.shape, dtype=self.loc.dtype)
            return self.loc + self.scale_tril @ normal

    class ThrusterFailureGPModel:
        def __init__(
            self,
            train_x,
            train_y,
            kernel_type="RBF",
            lengthscale=0.2,
            outputscale=1.0,
            jitter=1e-5,
        ):
            dtype = jnp.float32
            train_x = jnp.asarray(train_x, dtype=dtype).reshape(-1, 1)
            train_y = jnp.asarray(train_y, dtype=dtype).reshape(-1, 1)

            self._train_x = train_x
            self._kernel = self._choose_kernel(kernel_type, lengthscale, outputscale)
            self._mean_value = float(jnp.mean(train_y))
            self._train_jitter = jnp.asarray(jitter, dtype=dtype)
            self._predictive_jitter = jnp.asarray(max(jitter, 1e-6), dtype=dtype)

            centered_y = train_y - self._mean_value
            k_xx = self._kernel(train_x, train_x)
            noise = 1e-6 * jnp.eye(train_x.shape[0], dtype=k_xx.dtype)
            train_mat = (
                k_xx
                + noise
                + self._train_jitter * jnp.eye(k_xx.shape[0], dtype=k_xx.dtype)
            )
            self._train_chol = _stable_cholesky(train_mat, self._train_jitter)
            self._alpha = jsp.linalg.cho_solve((self._train_chol, True), centered_y)

        def _choose_kernel(self, kernel_type, lengthscale, outputscale):
            dtype = jnp.float32
            lengthscale = jnp.asarray(lengthscale, dtype=dtype)
            variance = jnp.asarray(outputscale, dtype=dtype)

            def _rbf(x, y):
                x = jnp.asarray(x, dtype=dtype)
                y = jnp.asarray(y, dtype=dtype)
                sq_dist = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
                return variance * jnp.exp(-0.5 * sq_dist / (lengthscale**2 + 1e-12))

            def _matern32(x, y):
                x = jnp.asarray(x, dtype=dtype)
                y = jnp.asarray(y, dtype=dtype)
                scaled = jnp.sqrt(
                    jnp.sum(
                        ((x[:, None, :] - y[None, :, :]) / (lengthscale + 1e-12)) ** 2,
                        axis=-1,
                    )
                )
                sqrt3 = jnp.sqrt(jnp.asarray(3.0, dtype=dtype))
                return variance * (1.0 + sqrt3 * scaled) * jnp.exp(-sqrt3 * scaled)

            if kernel_type == "RBF":
                return _rbf
            elif kernel_type == "Matern":
                return _matern32
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

        def forward(self, x):
            test_x = jnp.asarray(x, dtype=jnp.float32).reshape(-1, 1)

            k_xt = self._kernel(self._train_x, test_x)
            predictive_mean = self._mean_value + jnp.matmul(
                k_xt.T, self._alpha
            ).reshape(-1)

            v = jsp.linalg.solve_triangular(self._train_chol, k_xt, lower=True)
            k_tt = self._kernel(test_x, test_x)
            predictive_cov = k_tt - jnp.matmul(v.T, v)
            predictive_cov = (predictive_cov + predictive_cov.T) * 0.5
            predictive_cov = predictive_cov + self._predictive_jitter * jnp.eye(
                predictive_cov.shape[0], dtype=predictive_cov.dtype
            )
            scale_tril = _stable_cholesky(predictive_cov, self._predictive_jitter)

            return ThrusterFailureSimulator._GPPosterior(
                loc=predictive_mean, scale_tril=scale_tril
            )

    def generate_failure_data(self, key, failure_type):
        key_actual, key_subset, key_sample = jax.random.split(key, 3)

        actual_force = self._get_actual_force(key_actual, failure_type)
        if jnp.isnan(actual_force).any():
            print(
                f"[ThrusterFailureSimulator] actual_force has NaNs for failure {failure_type}"
            )
        params = self.failure_modes[failure_type]
        subset_override = params.get("subset_size", self.subset_size)
        subset_indices = self._select_subset_indices(key_subset, subset_override)
        demanded_force_subset = self.demanded_force[subset_indices]
        actual_force_subset = actual_force[subset_indices]

        model = self.ThrusterFailureGPModel(
            demanded_force_subset,
            actual_force_subset,
            kernel_type=params["kernel"],
            lengthscale=params["lengthscale"],
            outputscale=params["outputscale"],
            jitter=params.get("jitter", 1e-4),
        )

        x_test = jnp.linspace(0, self.upper_bound, self.num_points)
        sampled_function = self._sample_gp_function(
            model, x_test, failure_type, key_sample
        )
        if jnp.isnan(sampled_function).any():
            print(
                "[ThrusterFailureSimulator] sampled_function contains NaNs "
                f"(failure={failure_type}, min={jnp.nanmin(sampled_function)}, max={jnp.nanmax(sampled_function)})"
            )

        sampled_function = jax.lax.clamp(
            0.0, jnp.asarray(sampled_function), self.upper_bound
        )

        return x_test, sampled_function

    def _get_actual_force(self, key, failure_type):
        if failure_type == PerturbationStatus.SATURATED_THRUST:
            return jnp.where(
                self.demanded_force < 0.3 * self.upper_bound,
                self.demanded_force,
                0.3 * self.upper_bound
                + 0.0005
                * jax.random.normal(key, shape=self.demanded_force.shape)
                * (self.upper_bound - self.demanded_force),
            )
        elif failure_type == PerturbationStatus.FAULTY_VALVE:
            valve_min, valve_max = self.valve_min, self.valve_max
            lower_threshold = 0.67 * self.upper_bound
            upper_threshold = 0.73 * self.upper_bound
            actual_force = jnp.full_like(self.demanded_force, valve_min)
            linear_region = (self.demanded_force >= lower_threshold) & (
                self.demanded_force <= upper_threshold
            )
            actual_force = actual_force.at[linear_region].set(
                valve_min
                + (
                    (self.demanded_force[linear_region] - lower_threshold)
                    / (upper_threshold - lower_threshold)
                )
                * (valve_max - valve_min)
            )
            actual_force = actual_force.at[self.demanded_force > upper_threshold].set(
                valve_max
            )
            return actual_force
        elif failure_type == PerturbationStatus.THRUST_INSTABILITY:
            return (
                self.demanded_force
                - 0.1 * self.upper_bound * jnp.sin(10 * self.demanded_force)
                + 0.05
                * self.upper_bound
                * jax.random.normal(key, shape=self.demanded_force.shape)
            )
        elif failure_type == "thermal_stress":
            return self.demanded_force * jnp.exp(
                -0.5 * (self.demanded_force / self.upper_bound)
            )
        else:
            raise ValueError(f"Unknown failure type: {failure_type}")

    def _select_subset_indices(self, key, subset_size: Optional[int] = None):
        subset_size = int(subset_size or self.subset_size)
        subset_size = max(2, min(subset_size, self.num_points))
        interior = subset_size - 2
        pool = jnp.arange(1, self.num_points - 1)
        if interior <= 0 or pool.size == 0:
            indices = jnp.array([], dtype=pool.dtype)
        else:
            indices = jax.random.choice(
                key,
                pool,
                shape=(interior,),
                replace=False,
            )
        start = jnp.array([0], dtype=indices.dtype)
        end = jnp.array([self.num_points - 1], dtype=indices.dtype)
        indices = jnp.concatenate([start, indices, end])
        return jnp.sort(indices)

    def _sample_gp_function(self, gp_model, demanded_force, failure_type, sample_key):
        observed_pred = gp_model.forward(demanded_force)

        if failure_type == PerturbationStatus.SATURATED_THRUST:
            return observed_pred.mean()
        else:
            return observed_pred.sample(sample_key)

    def _apply_overrides(self, modes: dict, overrides: Optional[dict]) -> dict:
        if not overrides:
            return modes
        patched = dict(modes)
        for key, params in overrides.items():
            if isinstance(key, PerturbationStatus):
                status = key
            elif isinstance(key, str):
                status = PerturbationStatus[key] if key in PerturbationStatus.__members__ else key
            else:
                status = PerturbationStatus(int(key))
            base = dict(patched.get(status, {}))
            base.update(params)
            patched[status] = base
        return patched

    def _get_failure_modes(self):
        base_jitter = 1e-4
        return {
            PerturbationStatus.SATURATED_THRUST: {
                "kernel": "Matern",
                "lengthscale": 0.1,
                "outputscale": 0.01,
                "jitter": base_jitter,
            },
            PerturbationStatus.FAULTY_VALVE: {
                "kernel": "Matern",
                "lengthscale": 0.3,
                "outputscale": 0.4,
                "jitter": 5e-4,
            },
            PerturbationStatus.THRUST_INSTABILITY: {
                "kernel": "Matern",
                "lengthscale": 0.15,
                "outputscale": 0.5,
                "jitter": 5e-4,
            },
            "thermal_stress": {
                "kernel": "Matern",
                "lengthscale": 0.15,
                "outputscale": 0.5,
                "jitter": base_jitter,
            },
        }

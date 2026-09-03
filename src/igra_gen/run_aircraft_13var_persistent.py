import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import Dataset

WORKDIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(WORKDIR)
sys.path.insert(0, SRC_ROOT)
sys.path.insert(0, WORKDIR)

import sample_lsf_airtemp_common_native as sample_lsf
from igra_gen.generating.factory import sampler_factory
from igra_gen.models.precond import EDMPrecond
from igra_gen.utils import io


ERA5_ROOT = sample_lsf.ERA5_ROOT
HYDRA_CFG = sample_lsf.HYDRA_CFG
DEFAULT_CHECKPOINT = sample_lsf.DEFAULT_CHECKPOINT
GOES_LSF_ROOT = sample_lsf.GOES_LSF_ROOT
HOURLY_AIRTEMP2KM_ROOT = sample_lsf.HOURLY_AIRTEMP2KM_ROOT
IGRA_PKL = os.environ.get("IGRA_PKL", "")
NUM_CHANNELS = 13
AIRCRAFT_AROUND5_ROOT = os.environ.get("AIRCRAFT_AROUND5_ROOT", "")
AIRCRAFT_AROUND25_ROOT = os.environ.get("AIRCRAFT_AROUND25_ROOT", "")
SURFACE_METAR_STRAT24_ROOT = os.environ.get("SURFACE_METAR_ROOT", "")
# Legacy aliases retained for old run manifests and earlier diagnostics.
AIRCRAFT_CLEAN_ROOT = AIRCRAFT_AROUND5_ROOT
AIRCRAFT_MID_ROOT = AIRCRAFT_AROUND25_ROOT

IGRA_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "geopotential_500",
    "geopotential_850",
    "u_component_of_wind_500",
    "u_component_of_wind_850",
    "v_component_of_wind_500",
    "v_component_of_wind_850",
    "temperature_500",
    "temperature_850",
    "specific_humidity_500",
    "specific_humidity_850",
]

AIRCRAFT_VARIABLES = [
    "temperature_500",
    "temperature_850",
    "u_component_of_wind_500",
    "v_component_of_wind_500",
    "u_component_of_wind_850",
    "v_component_of_wind_850",
]

SURFACE_METAR_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

EXPERIMENTS = {
    "igra_only": {
        "use_igra": True,
        "obs_mode": "none",
        "obs_modality": None,
        "description": "IGRA radiosonde all-13 observations only.",
    },
    "igra_all13_aircraft_clean_v2": {
        "use_igra": True,
        "obs_mode": "aircraft_clean",
        "obs_modality": "aircraft",
        "obs_space": "aircraft_weighted_sparse",
        "aircraft_weighting": "era5_cell_balanced",
        "description": "IGRA all-13 plus MADIS aircraft 500=495-505 and 850=845-855 hPa, sparse point H with ERA5-cell-balanced weights.",
    },
    "igra_all13_aircraft_clean_v3": {
        "use_igra": True,
        "obs_mode": "aircraft_clean",
        "obs_modality": "aircraft",
        "obs_space": "aircraft_superob_grid",
        "description": "IGRA all-13 plus MADIS aircraft 500=495-505 and 850=845-855 hPa, ERA5-cell superob grid H.",
    },
    "igra_all13_aircraft_mid_v2": {
        "use_igra": True,
        "obs_mode": "aircraft_mid",
        "obs_modality": "aircraft",
        "obs_space": "aircraft_weighted_sparse",
        "aircraft_weighting": "era5_cell_balanced",
        "description": "IGRA all-13 plus MADIS aircraft 500=475-525 and 850=825-875 hPa, sparse point H with ERA5-cell-balanced weights.",
    },
    "igra_all13_aircraft_mid_v3": {
        "use_igra": True,
        "obs_mode": "aircraft_mid",
        "obs_modality": "aircraft",
        "obs_space": "aircraft_superob_grid",
        "description": "IGRA all-13 plus MADIS aircraft 500=475-525 and 850=825-875 hPa, ERA5-cell superob grid H.",
    },
    "igra_all13_aircraft_mid_v5_acars_superob": {
        "use_igra": True,
        "obs_mode": "aircraft_mid",
        "obs_modality": "aircraft",
        "obs_space": "aircraft_superob_grid",
        "aircraft_source_filter": "acars",
        "aircraft_spatial_support": "strict_conus",
        "description": "IGRA all-13 plus acars-only MADIS aircraft mid-window, ERA5-cell superob grid H.",
    },
    "igra_all13_aircraft_mid_v6_profiles_superob": {
        "use_igra": True,
        "obs_mode": "aircraft_mid",
        "obs_modality": "aircraft",
        "obs_space": "aircraft_superob_grid",
        "aircraft_source_filter": "acarsProfiles",
        "aircraft_spatial_support": "strict_conus",
        "description": "IGRA all-13 plus acarsProfiles-only MADIS aircraft mid-window, ERA5-cell superob grid H.",
    },
}

SPATIAL_SUPPORTS = {
    "strict_conus": {
        "lat": (24.0, 50.0),
        "lon": (-125.0, -66.0),
        "description": "Strict CONUS box used as the primary regional metric target.",
    },
    "conus_buffer": {
        "lat": (20.0, 55.0),
        "lon": (-135.0, -55.0),
        "description": "CONUS plus surrounding buffer for boundary-adjacent aircraft observations.",
    },
    "north_america": {
        "lat": (10.0, 70.0),
        "lon": (-170.0, -50.0),
        "description": "Broad North America diagnostic box.",
    },
    "global": {
        "lat": (-90.0, 90.0),
        "lon": (-180.0, 180.0),
        "description": "All available aircraft observations.",
    },
}


def _add_stage1_aircraft_experiments() -> None:
    """Register first-pass aircraft ablation experiments.

    Stage 1 uses the mid pressure window and varies spatial support plus H
    operator.  V2 and V4 are source-split likelihoods.  V3 is kept as the
    merged-superob reference so we can compare to the initial t0000 experiment.
    """
    for support in SPATIAL_SUPPORTS:
        EXPERIMENTS[f"stage1_{support}_mid_v1_split_simple_sparse"] = {
            "use_igra": True,
            "obs_mode": "aircraft_mid",
            "obs_modality": "aircraft_split",
            "obs_space": "aircraft_simple_sparse_split",
            "aircraft_spatial_support": support,
            "description": f"IGRA all-13 plus source-split MADIS aircraft mid-window V1 simple sparse point H without cell balancing, support={support}.",
        }
        EXPERIMENTS[f"stage1_{support}_mid_v2_split_sparse"] = {
            "use_igra": True,
            "obs_mode": "aircraft_mid",
            "obs_modality": "aircraft_split",
            "obs_space": "aircraft_weighted_sparse_split",
            "aircraft_spatial_support": support,
            "description": f"IGRA all-13 plus source-split MADIS aircraft mid-window V2 weighted sparse H, support={support}.",
        }
        EXPERIMENTS[f"stage1_{support}_mid_v3_merged_superob"] = {
            "use_igra": True,
            "obs_mode": "aircraft_mid",
            "obs_modality": "aircraft",
            "obs_space": "aircraft_superob_grid",
            "aircraft_spatial_support": support,
            "description": f"IGRA all-13 plus merged MADIS aircraft mid-window V3 superob/grid H, support={support}.",
        }
        EXPERIMENTS[f"stage1_{support}_mid_v4_split_dense"] = {
            "use_igra": True,
            "obs_mode": "aircraft_mid",
            "obs_modality": "aircraft_split",
            "obs_space": "aircraft_superob_grid_split",
            "aircraft_spatial_support": support,
            "description": f"IGRA all-13 plus source-split MADIS aircraft mid-window V4 AirTemp-like dense masked-grid H, support={support}.",
        }


_add_stage1_aircraft_experiments()


def _add_acars_h_window_ablation_experiments() -> None:
    """Register acars-only H/window ablation experiments.

    These are the clean first20 experiments requested for the aircraft branch:
    strict-CONUS support, acars/Aircraft Based Reports only, and two pressure
    windows named by half-width around 500/850 hPa.
    """
    windows = {
        "around25": ("aircraft_around25", "500=475-525 hPa and 850=825-875 hPa"),
        "around5": ("aircraft_around5", "500=495-505 hPa and 850=845-855 hPa"),
    }
    variants = {
        "v1_simple_sparse": {
            "obs_space": "aircraft_simple_sparse",
            "description": "V1 simple sparse point H with equal point weights.",
        },
        "v2_cell_balanced_sparse": {
            "obs_space": "aircraft_weighted_sparse",
            "description": "V2 sparse point H with inverse-count ERA5-cell balancing.",
        },
        "v4_equal_cell_mean": {
            "obs_space": "aircraft_superob_grid",
            "aircraft_grid_aggregation": "equal",
            "description": "V4 baseline: equal-weight ERA5-cell mean, then masked grid loss.",
        },
        "v4b_pressure_weighted": {
            "obs_space": "aircraft_superob_grid",
            "aircraft_grid_aggregation": "pressure",
            "description": "V4b: pressure-offset-weighted ERA5-cell mean, then masked grid loss.",
        },
        "v4c_distance_weighted": {
            "obs_space": "aircraft_superob_grid",
            "aircraft_grid_aggregation": "distance",
            "description": "V4c: within-cell distance-weighted ERA5-cell mean, then masked grid loss.",
        },
        "v4d_pressure_distance_weighted": {
            "obs_space": "aircraft_superob_grid",
            "aircraft_grid_aggregation": "pressure_distance",
            "description": "V4d: pressure-offset and within-cell-distance weighted ERA5-cell mean.",
        },
    }
    for window_name, (obs_mode, window_desc) in windows.items():
        for variant_name, variant in variants.items():
            EXPERIMENTS[f"acars_{window_name}_{variant_name}"] = {
                "use_igra": True,
                "obs_mode": obs_mode,
                "obs_modality": "aircraft",
                "aircraft_source_filter": "acars",
                "aircraft_spatial_support": "strict_conus",
                "aircraft_pressure_window_name": window_name,
                "aircraft_pressure_window_description": window_desc,
                "aircraft_pressure_weight_sigma_hpa": 15.0,
                "aircraft_distance_weight_sigma_cell": 1.0,
                **variant,
                "description": (
                    f"IGRA all-13 plus acars-only MADIS Aircraft Based Reports, "
                    f"{window_name} ({window_desc}), strict-CONUS support. {variant['description']}"
                ),
            }


_add_acars_h_window_ablation_experiments()


def _add_surface_metar_stagea_experiments() -> None:
    variants = {
        "v1_simple_sparse": {
            "obs_space": "surface_simple_sparse",
            "description": "V1 simple sparse point H with equal point weights.",
        },
        "v2_cell_balanced_sparse": {
            "obs_space": "surface_weighted_sparse",
            "description": "V2 sparse point H with inverse-count ERA5-cell balancing.",
        },
        "v4_equal_cell_mean": {
            "obs_space": "surface_superob_grid",
            "surface_grid_aggregation": "equal",
            "description": "V4 equal-weight ERA5-cell mean, then masked grid loss.",
        },
    }
    for variant_name, variant in variants.items():
        EXPERIMENTS[f"metar_strict_conus_t2muv10_{variant_name}"] = {
            "use_igra": True,
            "obs_mode": "surface_metar_strat24",
            "obs_modality": "surface",
            "surface_variables": SURFACE_METAR_VARIABLES,
            "surface_spatial_support": "strict_conus",
            **variant,
            "description": (
                "IGRA all-13 plus NOAA MADIS METAR surface observations "
                "(t2m/u10/v10), strict-CONUS support. "
                f"{variant['description']}"
            ),
        }


_add_surface_metar_stagea_experiments()


def _add_three_source_experiments() -> None:
    EXPERIMENTS["igra_abo_keep015_metar_v4_equal_cell_mean_strict_conus"] = {
        "use_igra": True,
        "obs_mode": "aircraft_around25",
        "obs_modality": "aircraft_surface",
        "obs_space": "aircraft_surface_superob_grid",
        "aircraft_source_filter": "combined",
        "aircraft_spatial_support": "strict_conus",
        "aircraft_grid_aggregation": "equal",
        "surface_variables": SURFACE_METAR_VARIABLES,
        "surface_spatial_support": "strict_conus",
        "surface_grid_aggregation": "equal",
        "description": (
            "IGRA all-13 plus NOAA MADIS ABO aircraft observations "
            "(keep dataSource={0,1,5}, around25, V4 equal-cell mean) "
            "and NOAA MADIS METAR surface observations (t2m/u10/v10, "
            "V4 equal-cell mean), strict-CONUS support."
        ),
    }


_add_three_source_experiments()


def _add_global_pilot_experiments() -> None:
    """Register global-support pilots without changing calibrated parameters."""
    EXPERIMENTS["igra_abo_keep015_global_v4_equal_cell_mean"] = {
        "use_igra": True,
        "obs_mode": "aircraft_around25",
        "obs_modality": "aircraft",
        "obs_space": "aircraft_superob_grid",
        "aircraft_source_filter": "combined",
        "aircraft_spatial_support": "global",
        "aircraft_grid_aggregation": "equal",
        "description": (
            "IGRA all-13 plus global NOAA MADIS ABO aircraft observations "
            "(keep dataSource={0,1,5}, around25, V4 equal-cell mean)."
        ),
    }
    EXPERIMENTS["igra_metar_global_v4_equal_cell_mean"] = {
        "use_igra": True,
        "obs_mode": "surface_metar_global",
        "obs_modality": "surface",
        "obs_space": "surface_superob_grid",
        "surface_variables": SURFACE_METAR_VARIABLES,
        "surface_spatial_support": "global",
        "surface_grid_aggregation": "equal",
        "description": (
            "IGRA all-13 plus global NOAA MADIS METAR t2m/u10/v10 observations "
            "using V4 equal-cell mean."
        ),
    }
    EXPERIMENTS["igra_abo_keep015_metar_global_v4_equal_cell_mean"] = {
        "use_igra": True,
        "obs_mode": "aircraft_around25",
        "obs_modality": "aircraft_surface",
        "obs_space": "aircraft_surface_superob_grid",
        "aircraft_source_filter": "combined",
        "aircraft_spatial_support": "global",
        "aircraft_grid_aggregation": "equal",
        "surface_variables": SURFACE_METAR_VARIABLES,
        "surface_spatial_support": "global",
        "surface_grid_aggregation": "equal",
        "description": (
            "IGRA all-13 plus global NOAA MADIS ABO keep015/around25 and global "
            "NOAA MADIS METAR t2m/u10/v10, both with V4 equal-cell mean."
        ),
    }


_add_global_pilot_experiments()


def parse_timesteps(text: str) -> List[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return list(dict.fromkeys(out))


def default_timesteps_16() -> List[int]:
    """The locally available HourlyAirTemp2kmUSA 2020 subset."""
    return (
        list(range(0, 4))
        + list(range(364, 368))
        + list(range(728, 732))
        + list(range(1096, 1100))
    )


def default_timesteps_first20() -> List[int]:
    """First 20 six-hourly 2020 timesteps: t0000 through t0019."""
    return list(range(0, 20))


def default_timesteps_first20_12h() -> List[int]:
    """First 20 matched 12-hourly 2020 timesteps: t0000, t0002, ..., t0038."""
    return [2 * i for i in range(20)]


def empty_channels(n_channels: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    ch_locs = [np.empty((0, 2), dtype=np.float32) for _ in range(n_channels)]
    ch_vals = [np.empty((0,), dtype=np.float32) for _ in range(n_channels)]
    return ch_locs, ch_vals


class PersistentFullPoolRunner:
    def __init__(
        self,
        output_root: str,
        ens: int = 16,
        seed: int = 17,
        num_steps: int = 50,
        sigma_min: float = 0.005,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        S_churn: float = 0.0,
        S_min: float = 0.01,
        S_max: float = 50.0,
        S_noise: float = 1.003,
        airtemp_mask_policy: str = "physical",
        igra_pkl: str = IGRA_PKL,
        aircraft_around5_root: str = AIRCRAFT_AROUND5_ROOT,
        aircraft_around25_root: str = AIRCRAFT_AROUND25_ROOT,
        aircraft_clean_root: str = AIRCRAFT_CLEAN_ROOT,
        aircraft_mid_root: str = AIRCRAFT_MID_ROOT,
        surface_metar_root: str = SURFACE_METAR_STRAT24_ROOT,
        checkpoint: str = DEFAULT_CHECKPOINT,
        era5_root: str = ERA5_ROOT,
        hydra_cfg: str = HYDRA_CFG,
        num_channels: int = NUM_CHANNELS,
        raw_goes_max_points: int = 0,
        likelihood_mode: str = "multimodal",
        std_igra: float = 5e-4,
        gamma_igra: float = 2e-6,
        lambda_igra: float = 1.0,
        std_goes: float = 5e-4,
        gamma_goes: float = 2e-6,
        lambda_goes: float = 1.0,
        std_airtemp: float = 5e-4,
        gamma_airtemp: float = 2e-6,
        lambda_airtemp: float = 1.0,
        std_aircraft: float = 5e-4,
        gamma_aircraft: float = 2e-6,
        lambda_aircraft: float = 1.0,
        std_surface: float = 5e-4,
        gamma_surface: float = 2e-6,
        lambda_surface: float = 1.0,
        std_aircraft_acars: Optional[float] = None,
        gamma_aircraft_acars: Optional[float] = None,
        lambda_aircraft_acars: Optional[float] = None,
        std_aircraft_profiles: Optional[float] = None,
        gamma_aircraft_profiles: Optional[float] = None,
        lambda_aircraft_profiles: Optional[float] = None,
        era5_split: str = "test",
        calendar_year: int = 2020,
    ):
        self.output_root = output_root
        self.samples_root = os.path.join(output_root, "samples")
        self.ens = ens
        self.seed = seed
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.airtemp_mask_policy = airtemp_mask_policy
        self.igra_pkl = igra_pkl
        self.aircraft_roots = {
            "aircraft_around5": aircraft_around5_root,
            "aircraft_around25": aircraft_around25_root,
            "aircraft_clean": aircraft_clean_root,
            "aircraft_mid": aircraft_mid_root,
        }
        self.surface_metar_root = surface_metar_root
        self.checkpoint = checkpoint
        self.era5_root = era5_root
        self.hydra_cfg = hydra_cfg
        self.num_channels = num_channels
        self.era5_split = era5_split
        self.calendar_year = int(calendar_year)
        self.raw_goes_max_points = raw_goes_max_points if raw_goes_max_points > 0 else None
        if likelihood_mode not in {"legacy", "multimodal"}:
            raise ValueError(f"Unknown likelihood_mode={likelihood_mode}")
        self.likelihood_mode = likelihood_mode
        self.likelihood_kwargs = {
            "std_igra": std_igra,
            "gamma_igra": gamma_igra,
            "lambda_igra": lambda_igra,
            "std_goes": std_goes,
            "gamma_goes": gamma_goes,
            "lambda_goes": lambda_goes,
            "std_airtemp": std_airtemp,
            "gamma_airtemp": gamma_airtemp,
            "lambda_airtemp": lambda_airtemp,
            "std_aircraft": std_aircraft,
            "gamma_aircraft": gamma_aircraft,
            "lambda_aircraft": lambda_aircraft,
            "std_surface": std_surface,
            "gamma_surface": gamma_surface,
            "lambda_surface": lambda_surface,
            "std_aircraft_acars": std_aircraft if std_aircraft_acars is None else std_aircraft_acars,
            "gamma_aircraft_acars": gamma_aircraft if gamma_aircraft_acars is None else gamma_aircraft_acars,
            "lambda_aircraft_acars": lambda_aircraft if lambda_aircraft_acars is None else lambda_aircraft_acars,
            "std_aircraft_profiles": std_aircraft if std_aircraft_profiles is None else std_aircraft_profiles,
            "gamma_aircraft_profiles": gamma_aircraft if gamma_aircraft_profiles is None else gamma_aircraft_profiles,
            "lambda_aircraft_profiles": lambda_aircraft if lambda_aircraft_profiles is None else lambda_aircraft_profiles,
        }

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        io.log0(f"Persistent runner using device={self.device}")

        means = np.load(os.path.join(self.era5_root, "normalize_mean.npz"))
        stds = np.load(os.path.join(self.era5_root, "normalize_std.npz"))
        self.mean_2m = float(np.asarray(means["2m_temperature"]).reshape(-1)[0])
        self.std_2m = float(np.asarray(stds["2m_temperature"]).reshape(-1)[0])
        self.era5_lat = np.load(os.path.join(self.era5_root, "lat.npy")).astype(np.float32)
        self.era5_lon = np.load(os.path.join(self.era5_root, "lon.npy")).astype(np.float32)

        conf = OmegaConf.load(self.hydra_cfg)
        conf.data.dataset.root = self.era5_root
        conf.data.dataset.split = self.era5_split
        self.dataset: Dataset = instantiate(conf.data.dataset, _convert_="object")
        self.model_vars = list(self.dataset.variables[:-1])
        if len(self.model_vars) != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} variables, got {len(self.model_vars)}: {self.model_vars}")
        self.temp_idx = self.model_vars.index("2m_temperature")
        self.channel_mean = {
            var: float(np.asarray(means[var]).reshape(-1)[0])
            for var in self.model_vars
        }
        self.channel_std = {
            var: float(np.asarray(stds[var]).reshape(-1)[0])
            for var in self.model_vars
        }

        self.net = EDMPrecond(
            model=conf.model,
            img_resolution=(128, 256),
            img_channels=self.num_channels,
            sigma_data=1,
            sigma_max=80,
            sigma_min=0.005,
            condition_channels=1,
        ).to(self.device).eval()

        io.log0(f"Loading checkpoint once: {checkpoint}")
        chkpt = torch.load(checkpoint, map_location=self.device, weights_only=True)
        state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in chkpt["ema"].items()}
        if any(k.startswith("model.") for k in state_dict):
            self.net.load_state_dict(state_dict)
        else:
            self.net.model.load_state_dict(state_dict)

        self.sample_fn = sampler_factory(
            mode="edm_pos_sample",
            net=self.net,
            conditioning_type="multimodal" if self.likelihood_mode == "multimodal" else "igra",
            in_shape=(16, 32),
            target_shape=(128, 256),
        )
        self.in_shape = (1, self.num_channels, 128, 256)

        io.log0(f"Loading IGRA pkl once: {igra_pkl}")
        with open(igra_pkl, "rb") as f:
            self.igra_data = pickle.load(f)

    def timestep_to_datetime(self, timestep: int) -> datetime:
        return datetime(self.calendar_year, 1, 1) + timedelta(hours=6 * int(timestep))

    def load_igra_channels(
        self,
        timestep: int,
        variables: Optional[Iterable[str]] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if timestep >= len(self.igra_data):
            raise IndexError(f"{self.igra_pkl} has only {len(self.igra_data)} timesteps; got t{timestep:04d}")
        base_query_locations, base_true_values = self.igra_data[timestep]
        base_query_locations = base_query_locations[0]
        base_true_values = base_true_values[0]
        ch_locs, ch_vals = empty_channels(len(self.model_vars))
        allowed = set(variables) if variables is not None else None
        for src_idx, var_name in enumerate(IGRA_VARIABLES):
            if allowed is not None and var_name not in allowed:
                continue
            dst_idx = self.model_vars.index(var_name)
            ch_locs[dst_idx] = np.asarray(base_query_locations[src_idx], dtype=np.float32)
            ch_vals[dst_idx] = np.asarray(base_true_values[src_idx], dtype=np.float32)
        return ch_locs, ch_vals

    def _nearest_era5_flat_cells(self, locs: np.ndarray) -> np.ndarray:
        lat_axis = self.era5_lat
        lon_axis = self.era5_lon
        dlat = float(np.median(np.diff(lat_axis)))
        dlon = float(np.median(np.diff(lon_axis)))
        locs = np.asarray(locs, dtype=np.float64)
        lat_idx = np.rint((locs[:, 0] - float(lat_axis[0])) / dlat).astype(np.int64)
        lon_idx = np.rint((np.mod(locs[:, 1], 360.0) - float(lon_axis[0])) / dlon).astype(np.int64)
        lat_idx = np.clip(lat_idx, 0, lat_axis.size - 1)
        lon_idx = np.mod(lon_idx, lon_axis.size)
        return lat_idx * lon_axis.size + lon_idx

    def _nearest_era5_indices(self, locs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lat_axis = self.era5_lat
        lon_axis = self.era5_lon
        dlat = float(np.median(np.diff(lat_axis)))
        dlon = float(np.median(np.diff(lon_axis)))
        locs = np.asarray(locs, dtype=np.float64)
        lat_idx = np.rint((locs[:, 0] - float(lat_axis[0])) / dlat).astype(np.int64)
        lon_idx = np.rint((np.mod(locs[:, 1], 360.0) - float(lon_axis[0])) / dlon).astype(np.int64)
        lat_idx = np.clip(lat_idx, 0, lat_axis.size - 1)
        lon_idx = np.mod(lon_idx, lon_axis.size)
        return lat_idx, lon_idx

    def _distance_to_nearest_cell_center_in_cell_units(self, locs: np.ndarray) -> np.ndarray:
        if len(locs) == 0:
            return np.empty((0,), dtype=np.float32)
        lat_idx, lon_idx = self._nearest_era5_indices(locs)
        dlat = float(np.median(np.diff(self.era5_lat)))
        dlon = float(np.median(np.diff(self.era5_lon)))
        locs = np.asarray(locs, dtype=np.float64)
        lat_diff = (locs[:, 0] - self.era5_lat[lat_idx]) / dlat
        lon360 = np.mod(locs[:, 1], 360.0)
        lon_diff_deg = ((lon360 - self.era5_lon[lon_idx] + 180.0) % 360.0) - 180.0
        lon_diff = lon_diff_deg / dlon
        return np.sqrt(lat_diff**2 + lon_diff**2).astype(np.float32)

    def _cell_balanced_weights(self, locs: np.ndarray) -> np.ndarray:
        if len(locs) == 0:
            return np.empty((0,), dtype=np.float32)
        flat_cell = self._nearest_era5_flat_cells(locs)
        unique, inverse, counts = np.unique(flat_cell, return_inverse=True, return_counts=True)
        del unique
        weights = 1.0 / counts[inverse].astype(np.float32)
        return weights.astype(np.float32)

    @staticmethod
    def _simple_weights(locs: np.ndarray) -> np.ndarray:
        return np.ones((len(locs),), dtype=np.float32)

    def _spatial_support_mask(self, locs: np.ndarray, support: str) -> np.ndarray:
        if support not in SPATIAL_SUPPORTS:
            raise ValueError(f"Unknown aircraft spatial support={support}; choices={sorted(SPATIAL_SUPPORTS)}")
        locs = np.asarray(locs)
        lat_lo, lat_hi = SPATIAL_SUPPORTS[support]["lat"]
        lon_lo, lon_hi = SPATIAL_SUPPORTS[support]["lon"]
        lon = ((locs[:, 1] + 180.0) % 360.0) - 180.0
        return (
            np.isfinite(locs[:, 0])
            & np.isfinite(lon)
            & (locs[:, 0] >= lat_lo)
            & (locs[:, 0] <= lat_hi)
            & (lon >= lon_lo)
            & (lon <= lon_hi)
        )

    @staticmethod
    def _source_filter_mask(products: np.ndarray, source_filter: Optional[str]) -> np.ndarray:
        if source_filter is None or source_filter == "combined":
            return np.ones(products.shape, dtype=bool)
        if source_filter == "acars":
            return products == "acars"
        if source_filter in {"acarsProfiles", "profiles"}:
            return products == "acarsProfiles"
        raise ValueError("source_filter must be one of None/combined/acars/acarsProfiles/profiles")

    def aggregate_points_to_era5_grid(
        self,
        locs: np.ndarray,
        vals: np.ndarray,
        pressures_hpa: Optional[np.ndarray] = None,
        target_pressure_hpa: Optional[float] = None,
        aggregation: str = "equal",
        sigma_pressure_hpa: float = 15.0,
        sigma_distance_cell: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate point observations to one normalized value per ERA5 cell.

        aggregation:
            equal: simple mean within each ERA5 cell.
            pressure: Gaussian pressure-offset weights.
            distance: Gaussian distance-to-nearest-cell-center weights.
            pressure_distance: product of pressure and distance weights.
        """
        locs = np.asarray(locs, dtype=np.float64)
        vals = np.asarray(vals, dtype=np.float64)
        if pressures_hpa is not None:
            pressures_hpa = np.asarray(pressures_hpa, dtype=np.float64)
        shape = (self.era5_lat.size, self.era5_lon.size)
        if len(vals) == 0:
            return (
                np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=bool),
                np.zeros(shape, dtype=np.int64),
            )
        good = np.isfinite(locs[:, 0]) & np.isfinite(locs[:, 1]) & np.isfinite(vals)
        if aggregation in {"pressure", "pressure_distance"}:
            if pressures_hpa is None or target_pressure_hpa is None:
                raise ValueError("Pressure-weighted aircraft aggregation requires pressures_hpa and target_pressure_hpa")
            good = good & np.isfinite(pressures_hpa)
        locs = locs[good]
        vals = vals[good]
        if pressures_hpa is not None:
            pressures_hpa = pressures_hpa[good]
        if len(vals) == 0:
            return (
                np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=bool),
                np.zeros(shape, dtype=np.int64),
            )
        flat_cell = self._nearest_era5_flat_cells(locs)
        n_cells = self.era5_lat.size * self.era5_lon.size
        if aggregation == "equal":
            weights = np.ones_like(vals, dtype=np.float64)
        elif aggregation == "pressure":
            weights = np.exp(-0.5 * ((pressures_hpa - float(target_pressure_hpa)) / float(sigma_pressure_hpa)) ** 2)
        elif aggregation == "distance":
            dist = self._distance_to_nearest_cell_center_in_cell_units(locs)
            weights = np.exp(-0.5 * (dist / float(sigma_distance_cell)) ** 2)
        elif aggregation == "pressure_distance":
            dist = self._distance_to_nearest_cell_center_in_cell_units(locs)
            w_pressure = np.exp(-0.5 * ((pressures_hpa - float(target_pressure_hpa)) / float(sigma_pressure_hpa)) ** 2)
            w_distance = np.exp(-0.5 * (dist / float(sigma_distance_cell)) ** 2)
            weights = w_pressure * w_distance
        else:
            raise ValueError(f"Unknown aircraft grid aggregation={aggregation}")
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
        sum_vals = np.bincount(flat_cell, weights=vals * weights, minlength=n_cells)
        sum_weights = np.bincount(flat_cell, weights=weights, minlength=n_cells)
        count = np.bincount(flat_cell, minlength=n_cells).astype(np.int64)
        mask = sum_weights > 0
        grid = np.zeros(n_cells, dtype=np.float32)
        grid[mask] = (sum_vals[mask] / sum_weights[mask]).astype(np.float32)
        return grid.reshape(shape), mask.reshape(shape), count.reshape(shape)

    def load_aircraft_measurements(
        self,
        obs_mode: str,
        timestep: int,
        source_filter: Optional[str] = None,
        spatial_support: str = "global",
        weighting: str = "cell_balanced",
        return_pressure: bool = False,
        data_sources: Optional[Iterable[int]] = None,
    ):
        root = self.aircraft_roots[obs_mode]
        path = os.path.join(root, "obs", f"madis_aircraft_13var_t{timestep:04d}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing aircraft obs file for {obs_mode} t{timestep:04d}: {path}")
        data = np.load(path, allow_pickle=False)
        ch_locs, ch_vals = empty_channels(len(self.model_vars))
        ch_weights = [np.empty((0,), dtype=np.float32) for _ in range(len(self.model_vars))]
        ch_pressures = [np.empty((0,), dtype=np.float32) for _ in range(len(self.model_vars))]
        counts: Dict[str, int] = {}
        cells: Dict[str, int] = {}
        source_counts: Dict[str, Dict[str, int]] = {}
        for var in AIRCRAFT_VARIABLES:
            dst_idx = self.model_vars.index(var)
            locs = np.asarray(data[f"{var}_locs"], dtype=np.float32)
            vals_physical = np.asarray(data[f"{var}_vals"], dtype=np.float32)
            pressure = np.asarray(data[f"{var}_pressure_hpa_alt_derived"], dtype=np.float32)
            products = np.asarray(data[f"{var}_source_product"])
            source_codes = np.asarray(data[f"{var}_data_source"], dtype=np.int16)
            keep = self._spatial_support_mask(locs, spatial_support) & self._source_filter_mask(products, source_filter)
            if data_sources is not None:
                allowed_sources = np.asarray(sorted(set(int(value) for value in data_sources)), dtype=np.int16)
                keep = keep & np.isin(source_codes, allowed_sources)
            locs = locs[keep]
            vals_physical = vals_physical[keep]
            pressure = pressure[keep]
            products = products[keep]
            source_codes = source_codes[keep]
            vals = ((vals_physical - self.channel_mean[var]) / self.channel_std[var]).astype(np.float32)
            ch_locs[dst_idx] = locs
            ch_vals[dst_idx] = vals
            ch_pressures[dst_idx] = pressure
            if weighting == "simple":
                ch_weights[dst_idx] = self._simple_weights(locs)
            elif weighting == "cell_balanced":
                ch_weights[dst_idx] = self._cell_balanced_weights(locs)
            else:
                raise ValueError(f"Unknown aircraft sparse weighting={weighting}")
            counts[var] = int(vals.size)
            cells[var] = int(np.unique(self._nearest_era5_flat_cells(locs)).size) if vals.size else 0
            source_counts[var] = {
                "acars": int(np.sum(products == "acars")) if vals.size else 0,
                "acarsProfiles": int(np.sum(products == "acarsProfiles")) if vals.size else 0,
                **{f"dataSource_{int(code)}": int(np.sum(source_codes == code)) for code in np.unique(source_codes)},
            }
        metadata_json = str(data["metadata_json"]) if "metadata_json" in data.files else "{}"
        meta = {
            "obs_mode": obs_mode,
            "aircraft_root": root,
            "aircraft_file": path,
            "aircraft_source_filter": source_filter or "combined",
            "aircraft_data_sources": "all" if data_sources is None else sorted(set(int(value) for value in data_sources)),
            "aircraft_spatial_support": spatial_support,
            "aircraft_spatial_support_json": json.dumps(SPATIAL_SUPPORTS[spatial_support], sort_keys=True),
            "aircraft_counts_json": json.dumps(counts, sort_keys=True),
            "aircraft_source_counts_json": json.dumps(source_counts, sort_keys=True),
            "aircraft_covered_era5_cells_json": json.dumps(cells, sort_keys=True),
            "source_metadata_json": metadata_json,
        }
        if return_pressure:
            return meta, ch_locs, ch_vals, ch_weights, ch_pressures
        return meta, ch_locs, ch_vals, ch_weights

    def load_aircraft_superob_grid(
        self,
        obs_mode: str,
        timestep: int,
        source_filter: Optional[str] = None,
        spatial_support: str = "global",
        aggregation: str = "equal",
        sigma_pressure_hpa: float = 15.0,
        sigma_distance_cell: float = 1.0,
        data_sources: Optional[Iterable[int]] = None,
    ) -> Tuple[Dict[str, str], List[np.ndarray], List[np.ndarray], List[int], np.ndarray]:
        meta, ch_locs, ch_vals, _, ch_pressures = self.load_aircraft_measurements(
            obs_mode,
            timestep,
            source_filter=source_filter,
            spatial_support=spatial_support,
            return_pressure=True,
            data_sources=data_sources,
        )
        grids: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        counts: List[np.ndarray] = []
        channel_indices: List[int] = []
        valid_cells: Dict[str, int] = {}
        max_points: Dict[str, int] = {}
        for var in AIRCRAFT_VARIABLES:
            idx = self.model_vars.index(var)
            target_pressure_hpa = 500.0 if var.endswith("_500") else 850.0
            grid, mask, count = self.aggregate_points_to_era5_grid(
                ch_locs[idx],
                ch_vals[idx],
                pressures_hpa=ch_pressures[idx],
                target_pressure_hpa=target_pressure_hpa,
                aggregation=aggregation,
                sigma_pressure_hpa=sigma_pressure_hpa,
                sigma_distance_cell=sigma_distance_cell,
            )
            grids.append(grid)
            masks.append(mask)
            counts.append(count)
            channel_indices.append(idx)
            valid_cells[var] = int(mask.sum())
            max_points[var] = int(count.max()) if count.size else 0
        count_stack = np.stack(counts, axis=0)
        meta.update({
            "obs_space": "aircraft_superob_grid",
            "aircraft_grid_aggregation": aggregation,
            "aircraft_pressure_weight_sigma_hpa": str(sigma_pressure_hpa),
            "aircraft_distance_weight_sigma_cell": str(sigma_distance_cell),
            "aircraft_valid_era5_cells_json": json.dumps(valid_cells, sort_keys=True),
            "aircraft_max_points_per_cell_json": json.dumps(max_points, sort_keys=True),
        })
        return meta, grids, masks, channel_indices, count_stack

    def load_aircraft_measurements_split(
        self,
        obs_mode: str,
        timestep: int,
        spatial_support: str,
        weighting: str = "cell_balanced",
    ) -> Tuple[Dict[str, str], Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]]]:
        out: Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]] = {}
        metas = {}
        for source_filter, key in [("acars", "aircraft_acars"), ("acarsProfiles", "aircraft_profiles")]:
            meta, locs, vals, weights = self.load_aircraft_measurements(
                obs_mode,
                timestep,
                source_filter=source_filter,
                spatial_support=spatial_support,
            )
            if weighting == "simple":
                weights = [self._simple_weights(x) for x in locs]
            elif weighting != "cell_balanced":
                raise ValueError(f"Unknown aircraft split sparse weighting={weighting}")
            metas[key] = meta
            out[key] = (locs, vals, weights)
        meta = {
            "obs_mode": obs_mode,
            "aircraft_spatial_support": spatial_support,
            "aircraft_split_sparse_weighting": weighting,
            "aircraft_split_meta_json": json.dumps(metas, sort_keys=True),
        }
        return meta, out

    def load_aircraft_superob_grid_split(
        self,
        obs_mode: str,
        timestep: int,
        spatial_support: str,
    ) -> Tuple[Dict[str, str], Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[int], np.ndarray]]]:
        out: Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[int], np.ndarray]] = {}
        metas = {}
        for source_filter, key in [("acars", "aircraft_acars"), ("acarsProfiles", "aircraft_profiles")]:
            meta, grids, masks, channel_indices, count_stack = self.load_aircraft_superob_grid(
                obs_mode,
                timestep,
                source_filter=source_filter,
                spatial_support=spatial_support,
            )
            metas[key] = meta
            out[key] = (grids, masks, channel_indices, count_stack)
        meta = {
            "obs_mode": obs_mode,
            "obs_space": "aircraft_superob_grid_split",
            "aircraft_spatial_support": spatial_support,
            "aircraft_split_meta_json": json.dumps(metas, sort_keys=True),
        }
        return meta, out

    def load_surface_metar_measurements(
        self,
        timestep: int,
        variables: Optional[Iterable[str]] = None,
        weighting: str = "cell_balanced",
        spatial_support: str = "strict_conus",
    ) -> Tuple[Dict[str, str], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        path = os.path.join(self.surface_metar_root, "obs", f"madis_metar_surface_13var_t{timestep:04d}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing METAR surface obs file for t{timestep:04d}: {path}")
        data = np.load(path, allow_pickle=False)
        ch_locs, ch_vals = empty_channels(len(self.model_vars))
        ch_weights = [np.empty((0,), dtype=np.float32) for _ in range(len(self.model_vars))]
        counts: Dict[str, int] = {}
        cells: Dict[str, int] = {}
        allowed = set(variables) if variables is not None else set(SURFACE_METAR_VARIABLES)
        for var in SURFACE_METAR_VARIABLES:
            if var not in allowed:
                continue
            dst_idx = self.model_vars.index(var)
            locs = np.asarray(data[f"{var}_locs"], dtype=np.float32)
            vals_physical = np.asarray(data[f"{var}_vals"], dtype=np.float32)
            keep = self._spatial_support_mask(locs, spatial_support)
            locs = locs[keep]
            vals_physical = vals_physical[keep]
            vals = ((vals_physical - self.channel_mean[var]) / self.channel_std[var]).astype(np.float32)
            ch_locs[dst_idx] = locs
            ch_vals[dst_idx] = vals
            if weighting == "simple":
                ch_weights[dst_idx] = self._simple_weights(locs)
            elif weighting == "cell_balanced":
                ch_weights[dst_idx] = self._cell_balanced_weights(locs)
            else:
                raise ValueError(f"Unknown surface sparse weighting={weighting}")
            counts[var] = int(vals.size)
            cells[var] = int(np.unique(self._nearest_era5_flat_cells(locs)).size) if vals.size else 0
        metadata_json = str(data["metadata_json"]) if "metadata_json" in data.files else "{}"
        meta = {
            "obs_mode": "surface_metar_strat24",
            "surface_metar_root": self.surface_metar_root,
            "surface_metar_file": path,
            "surface_spatial_support": spatial_support,
            "surface_spatial_support_json": json.dumps(SPATIAL_SUPPORTS[spatial_support], sort_keys=True),
            "surface_variables_json": json.dumps(sorted(allowed)),
            "surface_counts_json": json.dumps(counts, sort_keys=True),
            "surface_covered_era5_cells_json": json.dumps(cells, sort_keys=True),
            "source_metadata_json": metadata_json,
        }
        return meta, ch_locs, ch_vals, ch_weights

    def load_surface_metar_superob_grid(
        self,
        timestep: int,
        variables: Optional[Iterable[str]] = None,
        aggregation: str = "equal",
        spatial_support: str = "strict_conus",
    ) -> Tuple[Dict[str, str], List[np.ndarray], List[np.ndarray], List[int], np.ndarray]:
        meta, ch_locs, ch_vals, _ = self.load_surface_metar_measurements(
            timestep,
            variables=variables,
            weighting="cell_balanced",
            spatial_support=spatial_support,
        )
        grids: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        counts: List[np.ndarray] = []
        channel_indices: List[int] = []
        valid_cells: Dict[str, int] = {}
        max_points: Dict[str, int] = {}
        allowed = set(variables) if variables is not None else set(SURFACE_METAR_VARIABLES)
        for var in SURFACE_METAR_VARIABLES:
            if var not in allowed:
                continue
            idx = self.model_vars.index(var)
            grid, mask, count = self.aggregate_points_to_era5_grid(
                ch_locs[idx],
                ch_vals[idx],
                aggregation=aggregation,
            )
            grids.append(grid)
            masks.append(mask)
            counts.append(count)
            channel_indices.append(idx)
            valid_cells[var] = int(mask.sum())
            max_points[var] = int(count.max()) if count.size else 0
        count_stack = np.stack(counts, axis=0) if counts else np.zeros((0,), dtype=np.int64)
        meta.update({
            "obs_space": "surface_superob_grid",
            "surface_grid_aggregation": aggregation,
            "surface_valid_era5_cells_json": json.dumps(valid_cells, sort_keys=True),
            "surface_max_points_per_cell_json": json.dumps(max_points, sort_keys=True),
        })
        return meta, grids, masks, channel_indices, count_stack

    def load_all_airtemp2km_measurements(self, timestep: int) -> Tuple[Dict[str, str], np.ndarray, np.ndarray]:
        target_dt = self.timestep_to_datetime(timestep)
        goes_file = sample_lsf._nearest_lsf_file(self.goes_lsf_root, target_dt)
        airtemp_file, airtemp_k, airtemp_valid = sample_lsf.load_hourly_airtemp2km(
            root=self.hourly_airtemp2km_root,
            target_dt=target_dt,
            mask_policy=self.airtemp_mask_policy,
        )
        ds = xr.open_dataset(goes_file, engine="h5netcdf")
        try:
            lats, lons = sample_lsf._goes_xy_to_latlon(ds)
        finally:
            ds.close()
        if airtemp_k.shape != lats.shape:
            raise ValueError(f"AirTemp grid shape {airtemp_k.shape} does not match GOES grid shape {lats.shape}")
        valid = airtemp_valid & np.isfinite(airtemp_k) & np.isfinite(lats) & np.isfinite(lons)
        idx = np.where(valid.ravel())[0]
        if idx.size == 0:
            raise ValueError(f"No valid all-AirTemp points for t{timestep:04d}")
        locs = np.stack([lats.ravel()[idx], lons.ravel()[idx]], axis=1).astype(np.float32)
        vals = ((airtemp_k.ravel()[idx] - self.mean_2m) / self.std_2m).astype(np.float32)
        meta = {
            "goes_grid_file": goes_file,
            "airtemp_file": airtemp_file,
            "obs_mode": "all_airtemp2km",
            "kept_points": str(idx.size),
        }
        return meta, locs, vals

    def load_raw_goes_common_measurements(self, timestep: int) -> Tuple[Dict[str, str], np.ndarray, np.ndarray]:
        target_dt = self.timestep_to_datetime(timestep)
        pool = sample_lsf.load_lsf_valid_pool(
            goes_lsf_root=self.goes_lsf_root,
            target_dt=target_dt,
            era5_mean_2m=self.mean_2m,
            era5_std_2m=self.std_2m,
            observation_product="hourly_airtemp2kmusa",
            hourly_airtemp2km_root=self.hourly_airtemp2km_root,
            airtemp_mask_policy=self.airtemp_mask_policy,
        )
        idx = np.asarray(pool["idx"], dtype=np.int64)
        lats = pool["lats"]
        lons = pool["lons"]
        lst_k = pool["lst_k"]
        locs = np.stack([lats.ravel()[idx], lons.ravel()[idx]], axis=1).astype(np.float32)
        vals = ((lst_k.ravel()[idx] - self.mean_2m) / self.std_2m).astype(np.float32)
        meta = {
            "lsf_file": pool["lsf_file"],
            "airtemp_file": pool["observation_file"],
            "obs_mode": "raw_goes_common",
            "kept_points": str(idx.size),
            "common_pool_rule": "GOES LST valid and HourlyAirTemp2kmUSA valid on the same ABI grid index; raw GOES LST values are used.",
        }
        return meta, locs, vals

    def aggregate_native_to_era5_grid(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        values_k: np.ndarray,
        valid_idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate dense native pixels to one normalized value per ERA5 cell."""
        if valid_idx.size == 0:
            raise ValueError("Cannot aggregate an empty native-pixel pool")
        lat_axis = self.era5_lat
        lon_axis = self.era5_lon
        dlat = float(np.median(np.diff(lat_axis)))
        dlon = float(np.median(np.diff(lon_axis)))
        lat_flat = lats.ravel()[valid_idx].astype(np.float64)
        lon_flat = np.mod(lons.ravel()[valid_idx].astype(np.float64), 360.0)
        val_flat = values_k.ravel()[valid_idx].astype(np.float64)
        good = np.isfinite(lat_flat) & np.isfinite(lon_flat) & np.isfinite(val_flat)
        lat_flat = lat_flat[good]
        lon_flat = lon_flat[good]
        val_flat = val_flat[good]
        if val_flat.size == 0:
            raise ValueError("No finite native pixels left after aggregate filtering")

        lat_idx = np.rint((lat_flat - float(lat_axis[0])) / dlat).astype(np.int64)
        lon_idx = np.rint((lon_flat - float(lon_axis[0])) / dlon).astype(np.int64)
        lat_idx = np.clip(lat_idx, 0, lat_axis.size - 1)
        lon_idx = np.mod(lon_idx, lon_axis.size)
        flat_cell = lat_idx * lon_axis.size + lon_idx
        n_cells = lat_axis.size * lon_axis.size
        sum_k = np.bincount(flat_cell, weights=val_flat, minlength=n_cells)
        count = np.bincount(flat_cell, minlength=n_cells).astype(np.int64)
        mask = count > 0
        grid_k = np.full(n_cells, np.nan, dtype=np.float32)
        grid_k[mask] = (sum_k[mask] / count[mask]).astype(np.float32)
        grid_norm = np.zeros(n_cells, dtype=np.float32)
        grid_norm[mask] = ((grid_k[mask] - self.mean_2m) / self.std_2m).astype(np.float32)
        shape = (lat_axis.size, lon_axis.size)
        return grid_norm.reshape(shape), mask.reshape(shape), count.reshape(shape)

    def load_grid_common_measurements(self, obs_mode: str, timestep: int) -> Tuple[Dict[str, str], np.ndarray, np.ndarray, np.ndarray]:
        target_dt = self.timestep_to_datetime(timestep)
        pool = sample_lsf.load_lsf_valid_pool(
            goes_lsf_root=self.goes_lsf_root,
            target_dt=target_dt,
            era5_mean_2m=self.mean_2m,
            era5_std_2m=self.std_2m,
            observation_product="hourly_airtemp2kmusa",
            hourly_airtemp2km_root=self.hourly_airtemp2km_root,
            airtemp_mask_policy=self.airtemp_mask_policy,
        )
        idx = np.asarray(pool["idx"], dtype=np.int64)
        if obs_mode == "goes_grid_common":
            values_k = pool["lst_k"]
            product = "raw_goes_lst"
        elif obs_mode == "airtemp_grid_common":
            values_k = pool["observation_k"]
            product = "hourly_airtemp2kmusa"
        else:
            raise ValueError(f"Unknown gridded obs_mode={obs_mode}")
        obs_grid, mask_grid, count_grid = self.aggregate_native_to_era5_grid(
            lats=pool["lats"],
            lons=pool["lons"],
            values_k=values_k,
            valid_idx=idx,
        )
        meta = {
            "lsf_file": pool["lsf_file"],
            "airtemp_file": pool["observation_file"],
            "obs_mode": obs_mode,
            "obs_space": "era5_grid_aggregate",
            "native_common_points": str(idx.size),
            "valid_era5_cells": str(int(mask_grid.sum())),
            "max_native_pixels_per_cell": str(int(count_grid.max())),
            "observation_product": product,
            "common_pool_rule": "Native pixels must have both valid GOES LST and valid HourlyAirTemp2kmUSA; values are aggregated to ERA5 grid cells before likelihood.",
        }
        return meta, obs_grid, mask_grid, count_grid

    def load_observation_measurements(self, obs_mode: str, timestep: int) -> Tuple[Dict[str, str], np.ndarray, np.ndarray]:
        target_dt = self.timestep_to_datetime(timestep)
        if obs_mode == "none":
            locs = np.empty((0, 2), dtype=np.float32)
            vals = np.empty((0,), dtype=np.float32)
            return {"obs_mode": "none", "kept_points": "0"}, locs, vals
        if obs_mode in {"raw_goes_lst", "raw_goes_full"}:
            lsf_file, locs, vals = sample_lsf.load_lsf_measurements(
                goes_lsf_root=self.goes_lsf_root,
                target_dt=target_dt,
                max_points=self.raw_goes_max_points,
                seed=self.seed + timestep,
                era5_mean_2m=self.mean_2m,
                era5_std_2m=self.std_2m,
                selected_indices_path=None,
                observation_product="raw_goes_lst",
                hourly_airtemp2km_root=self.hourly_airtemp2km_root,
                airtemp_mask_policy=self.airtemp_mask_policy,
            )
            return {
                "lsf_file": lsf_file,
                "obs_mode": obs_mode,
                "kept_points": str(len(vals)),
                "raw_goes_max_points": str(self.raw_goes_max_points or ""),
            }, locs, vals
        if obs_mode == "raw_goes_common":
            return self.load_raw_goes_common_measurements(timestep)
        if obs_mode == "airtemp_common":
            lsf_file, locs, vals = sample_lsf.load_lsf_measurements(
                goes_lsf_root=self.goes_lsf_root,
                target_dt=target_dt,
                max_points=None,
                seed=self.seed + timestep,
                era5_mean_2m=self.mean_2m,
                era5_std_2m=self.std_2m,
                selected_indices_path=None,
                observation_product="hourly_airtemp2kmusa",
                hourly_airtemp2km_root=self.hourly_airtemp2km_root,
                airtemp_mask_policy=self.airtemp_mask_policy,
            )
            return {
                "lsf_file": lsf_file,
                "obs_mode": "airtemp_common",
                "kept_points": str(len(vals)),
                "common_pool_rule": "GOES LST valid and HourlyAirTemp2kmUSA valid on the same ABI grid index; HourlyAirTemp values are used.",
            }, locs, vals
        if obs_mode == "all_airtemp2km":
            return self.load_all_airtemp2km_measurements(timestep)
        raise ValueError(f"Unknown obs_mode={obs_mode}")

    def output_path(self, experiment: str, timestep: int) -> str:
        return os.path.join(
            self.samples_root,
            experiment,
            f"{experiment}_t{timestep:04d}_e{self.ens}_s{self.num_steps}.npy",
        )

    def as_ensemble_measurement(self, ch_locs: List[np.ndarray], ch_vals: List[np.ndarray]):
        return [[ch_locs] * self.ens, [ch_vals] * self.ens]

    def as_weighted_ensemble_measurement(
        self,
        ch_locs: List[np.ndarray],
        ch_vals: List[np.ndarray],
        ch_weights: List[np.ndarray],
    ):
        return {
            "kind": "weighted_sparse",
            "query_locations": [ch_locs] * self.ens,
            "true_values": [ch_vals] * self.ens,
            "weights": [ch_weights] * self.ens,
        }

    def temp_only_channels(self, locs: np.ndarray, vals: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        ch_locs, ch_vals = empty_channels(len(self.model_vars))
        ch_locs[self.temp_idx] = np.asarray(locs, dtype=np.float32)
        ch_vals[self.temp_idx] = np.asarray(vals, dtype=np.float32)
        return ch_locs, ch_vals

    def run_one(self, experiment: str, timestep: int, overwrite: bool = False) -> str:
        if experiment not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment={experiment}. Choices: {sorted(EXPERIMENTS)}")
        spec = EXPERIMENTS[experiment]
        output = self.output_path(experiment, timestep)
        if os.path.exists(output) and not overwrite:
            io.log0(f"Skipping existing output: {output}")
            return output

        target_dt = self.timestep_to_datetime(timestep)
        io.log0(f"Running {experiment} t{timestep:04d} target_dt={target_dt.isoformat()}")
        obs_space = spec.get("obs_space", "points")
        obs_grid = obs_mask = obs_count = None
        aircraft_locs = aircraft_vals = aircraft_weights = None
        aircraft_grids = aircraft_masks = aircraft_channel_indices = None
        aircraft_split_sparse = None
        aircraft_split_grid = None
        surface_locs = surface_vals = surface_weights = None
        surface_grids = surface_masks = surface_channel_indices = None
        if obs_space == "grid":
            obs_meta, obs_grid, obs_mask, obs_count = self.load_grid_common_measurements(spec["obs_mode"], timestep)
            obs_locs = np.empty((0, 2), dtype=np.float32)
            obs_vals = np.empty((0,), dtype=np.float32)
        elif obs_space == "aircraft_surface_sparse":
            aircraft_weighting = spec.get("aircraft_weighting", "simple")
            surface_weighting = spec.get("surface_weighting", "simple")
            aircraft_meta, aircraft_locs, aircraft_vals, aircraft_weights = self.load_aircraft_measurements(
                spec["obs_mode"],
                timestep,
                source_filter=spec.get("aircraft_source_filter"),
                spatial_support=spec.get("aircraft_spatial_support", "global"),
                weighting=aircraft_weighting,
                data_sources=spec.get("aircraft_data_sources"),
            )
            surface_meta, surface_locs, surface_vals, surface_weights = self.load_surface_metar_measurements(
                timestep,
                variables=spec.get("surface_variables"),
                weighting=surface_weighting,
                spatial_support=spec.get("surface_spatial_support", "strict_conus"),
            )
            obs_meta = {
                "obs_space": "aircraft_surface_sparse",
                "aircraft_sparse_weighting": aircraft_weighting,
                "surface_sparse_weighting": surface_weighting,
                "aircraft_meta": aircraft_meta,
                "surface_meta": surface_meta,
            }
            obs_locs = np.empty((0, 2), dtype=np.float32)
            obs_vals = np.empty((0,), dtype=np.float32)
        elif obs_space in {"aircraft_weighted_sparse", "aircraft_simple_sparse"}:
            obs_meta, aircraft_locs, aircraft_vals, aircraft_weights = self.load_aircraft_measurements(
                spec["obs_mode"],
                timestep,
                source_filter=spec.get("aircraft_source_filter"),
                spatial_support=spec.get("aircraft_spatial_support", "global"),
                weighting="simple" if obs_space == "aircraft_simple_sparse" else "cell_balanced",
                data_sources=spec.get("aircraft_data_sources"),
            )
            obs_locs = np.empty((0, 2), dtype=np.float32)
            obs_vals = np.empty((0,), dtype=np.float32)
        elif obs_space in {"aircraft_weighted_sparse_split", "aircraft_simple_sparse_split"}:
            obs_meta, aircraft_split_sparse = self.load_aircraft_measurements_split(
                spec["obs_mode"],
                timestep,
                spatial_support=spec.get("aircraft_spatial_support", "global"),
                weighting="simple" if obs_space == "aircraft_simple_sparse_split" else "cell_balanced",
            )
            obs_locs = np.empty((0, 2), dtype=np.float32)
            obs_vals = np.empty((0,), dtype=np.float32)
        elif obs_space == "aircraft_superob_grid":
            obs_meta, aircraft_grids, aircraft_masks, aircraft_channel_indices, obs_count = self.load_aircraft_superob_grid(
                spec["obs_mode"],
                timestep,
                source_filter=spec.get("aircraft_source_filter"),
                spatial_support=spec.get("aircraft_spatial_support", "global"),
                aggregation=spec.get("aircraft_grid_aggregation", "equal"),
                sigma_pressure_hpa=float(spec.get("aircraft_pressure_weight_sigma_hpa", 15.0)),
                sigma_distance_cell=float(spec.get("aircraft_distance_weight_sigma_cell", 1.0)),
                data_sources=spec.get("aircraft_data_sources"),
            )
            obs_locs = np.empty((0, 2), dtype=np.float32)
            obs_vals = np.empty((0,), dtype=np.float32)
        elif obs_space == "aircraft_superob_grid_split":
            obs_meta, aircraft_split_grid = self.load_aircraft_superob_grid_split(
                spec["obs_mode"],
                timestep,
                spatial_support=spec.get("aircraft_spatial_support", "global"),
            )
            obs_locs = np.empty((0, 2), dtype=np.float32)
            obs_vals = np.empty((0,), dtype=np.float32)
        elif obs_space in {"surface_weighted_sparse", "surface_simple_sparse"}:
            obs_meta, surface_locs, surface_vals, surface_weights = self.load_surface_metar_measurements(
                timestep,
                variables=spec.get("surface_variables"),
                weighting="simple" if obs_space == "surface_simple_sparse" else "cell_balanced",
                spatial_support=spec.get("surface_spatial_support", "strict_conus"),
            )
            obs_locs = np.empty((0, 2), dtype=np.float32)
            obs_vals = np.empty((0,), dtype=np.float32)
        elif obs_space == "surface_superob_grid":
            obs_meta, surface_grids, surface_masks, surface_channel_indices, obs_count = self.load_surface_metar_superob_grid(
                timestep,
                variables=spec.get("surface_variables"),
                aggregation=spec.get("surface_grid_aggregation", "equal"),
                spatial_support=spec.get("surface_spatial_support", "strict_conus"),
            )
            obs_locs = np.empty((0, 2), dtype=np.float32)
            obs_vals = np.empty((0,), dtype=np.float32)
        elif obs_space == "aircraft_surface_superob_grid":
            aircraft_meta, aircraft_grids, aircraft_masks, aircraft_channel_indices, aircraft_count = self.load_aircraft_superob_grid(
                spec["obs_mode"],
                timestep,
                source_filter=spec.get("aircraft_source_filter"),
                spatial_support=spec.get("aircraft_spatial_support", "global"),
                aggregation=spec.get("aircraft_grid_aggregation", "equal"),
                sigma_pressure_hpa=float(spec.get("aircraft_pressure_weight_sigma_hpa", 15.0)),
                sigma_distance_cell=float(spec.get("aircraft_distance_weight_sigma_cell", 1.0)),
                data_sources=spec.get("aircraft_data_sources"),
            )
            surface_meta, surface_grids, surface_masks, surface_channel_indices, surface_count = self.load_surface_metar_superob_grid(
                timestep,
                variables=spec.get("surface_variables"),
                aggregation=spec.get("surface_grid_aggregation", "equal"),
                spatial_support=spec.get("surface_spatial_support", "strict_conus"),
            )
            obs_meta = {
                "obs_space": "aircraft_surface_superob_grid",
                "aircraft_meta": aircraft_meta,
                "surface_meta": surface_meta,
            }
            obs_count = np.asarray(
                list(np.ravel(aircraft_count)) + list(np.ravel(surface_count)),
                dtype=np.int64,
            )
            obs_locs = np.empty((0, 2), dtype=np.float32)
            obs_vals = np.empty((0,), dtype=np.float32)
        else:
            obs_meta, obs_locs, obs_vals = self.load_observation_measurements(spec["obs_mode"], timestep)

        if spec["use_igra"]:
            igra_locs, igra_vals = self.load_igra_channels(timestep, variables=spec.get("igra_variables"))
        else:
            igra_locs, igra_vals = empty_channels(len(self.model_vars))

        if self.likelihood_mode == "legacy":
            if obs_space == "grid":
                raise ValueError("Grid-space GOES/AirTemp experiments require --likelihood_mode multimodal")
            ch_locs = [np.asarray(x, dtype=np.float32) for x in igra_locs]
            ch_vals = [np.asarray(x, dtype=np.float32) for x in igra_vals]
            ch_locs[self.temp_idx] = np.concatenate([ch_locs[self.temp_idx], obs_locs], axis=0)
            ch_vals[self.temp_idx] = np.concatenate([ch_vals[self.temp_idx], obs_vals], axis=0)
            measurement = self.as_ensemble_measurement(ch_locs, ch_vals)
        else:
            measurement = {}
            ch_locs, ch_vals = empty_channels(len(self.model_vars))
            if spec["use_igra"]:
                measurement["igra"] = self.as_ensemble_measurement(igra_locs, igra_vals)
                ch_locs = [np.asarray(x, dtype=np.float32) for x in igra_locs]
                ch_vals = [np.asarray(x, dtype=np.float32) for x in igra_vals]
            if len(obs_vals):
                obs_ch_locs, obs_ch_vals = self.temp_only_channels(obs_locs, obs_vals)
                obs_modality = spec["obs_modality"]
                if obs_modality not in {"goes", "airtemp"}:
                    raise ValueError(f"Experiment {experiment} has obs points but invalid obs_modality={obs_modality}")
                measurement[obs_modality] = self.as_ensemble_measurement(obs_ch_locs, obs_ch_vals)
                ch_locs[self.temp_idx] = np.concatenate([ch_locs[self.temp_idx], obs_locs], axis=0)
                ch_vals[self.temp_idx] = np.concatenate([ch_vals[self.temp_idx], obs_vals], axis=0)
            if obs_space == "grid":
                obs_modality = spec["obs_modality"]
                if obs_modality not in {"goes", "airtemp"}:
                    raise ValueError(f"Experiment {experiment} has gridded obs but invalid obs_modality={obs_modality}")
                measurement[obs_modality] = {
                    "kind": "grid",
                    "grid": obs_grid,
                    "mask": obs_mask,
                    "channel_idx": self.temp_idx,
                }
            if obs_space in {"aircraft_weighted_sparse", "aircraft_simple_sparse"}:
                obs_modality = spec["obs_modality"]
                if obs_modality != "aircraft":
                    raise ValueError(f"Experiment {experiment} has aircraft obs but invalid obs_modality={obs_modality}")
                measurement[obs_modality] = self.as_weighted_ensemble_measurement(
                    aircraft_locs,
                    aircraft_vals,
                    aircraft_weights,
                )
                for i in range(len(self.model_vars)):
                    if len(aircraft_vals[i]):
                        ch_locs[i] = np.concatenate([ch_locs[i], aircraft_locs[i]], axis=0)
                        ch_vals[i] = np.concatenate([ch_vals[i], aircraft_vals[i]], axis=0)
            if obs_space == "aircraft_surface_sparse":
                if spec["obs_modality"] != "aircraft_surface":
                    raise ValueError(
                        f"Experiment {experiment} has aircraft+surface sparse obs "
                        f"but invalid obs_modality={spec['obs_modality']}"
                    )
                measurement["aircraft"] = self.as_weighted_ensemble_measurement(
                    aircraft_locs,
                    aircraft_vals,
                    aircraft_weights,
                )
                measurement["surface"] = self.as_weighted_ensemble_measurement(
                    surface_locs,
                    surface_vals,
                    surface_weights,
                )
                for i in range(len(self.model_vars)):
                    if len(aircraft_vals[i]):
                        ch_locs[i] = np.concatenate([ch_locs[i], aircraft_locs[i]], axis=0)
                        ch_vals[i] = np.concatenate([ch_vals[i], aircraft_vals[i]], axis=0)
                    if len(surface_vals[i]):
                        ch_locs[i] = np.concatenate([ch_locs[i], surface_locs[i]], axis=0)
                        ch_vals[i] = np.concatenate([ch_vals[i], surface_vals[i]], axis=0)
            if obs_space in {"aircraft_weighted_sparse_split", "aircraft_simple_sparse_split"}:
                if spec["obs_modality"] != "aircraft_split":
                    raise ValueError(f"Experiment {experiment} has split aircraft obs but invalid obs_modality={spec['obs_modality']}")
                for key, (src_locs, src_vals, src_weights) in aircraft_split_sparse.items():
                    measurement[key] = self.as_weighted_ensemble_measurement(src_locs, src_vals, src_weights)
                    for i in range(len(self.model_vars)):
                        if len(src_vals[i]):
                            ch_locs[i] = np.concatenate([ch_locs[i], src_locs[i]], axis=0)
                            ch_vals[i] = np.concatenate([ch_vals[i], src_vals[i]], axis=0)
            if obs_space == "aircraft_superob_grid":
                obs_modality = spec["obs_modality"]
                if obs_modality != "aircraft":
                    raise ValueError(f"Experiment {experiment} has aircraft grid obs but invalid obs_modality={obs_modality}")
                measurement[obs_modality] = {
                    "kind": "multi_grid",
                    "grids": aircraft_grids,
                    "masks": aircraft_masks,
                    "channel_indices": aircraft_channel_indices,
                }
            if obs_space == "aircraft_surface_superob_grid":
                if spec["obs_modality"] != "aircraft_surface":
                    raise ValueError(
                        f"Experiment {experiment} has aircraft+surface grid obs "
                        f"but invalid obs_modality={spec['obs_modality']}"
                    )
                measurement["aircraft"] = {
                    "kind": "multi_grid",
                    "grids": aircraft_grids,
                    "masks": aircraft_masks,
                    "channel_indices": aircraft_channel_indices,
                }
                measurement["surface"] = {
                    "kind": "multi_grid",
                    "grids": surface_grids,
                    "masks": surface_masks,
                    "channel_indices": surface_channel_indices,
                }
            if obs_space == "aircraft_superob_grid_split":
                if spec["obs_modality"] != "aircraft_split":
                    raise ValueError(f"Experiment {experiment} has split aircraft grid obs but invalid obs_modality={spec['obs_modality']}")
                for key, (src_grids, src_masks, src_channel_indices, _src_count) in aircraft_split_grid.items():
                    measurement[key] = {
                        "kind": "multi_grid",
                        "grids": src_grids,
                        "masks": src_masks,
                        "channel_indices": src_channel_indices,
                    }
            if obs_space in {"surface_weighted_sparse", "surface_simple_sparse"}:
                obs_modality = spec["obs_modality"]
                if obs_modality not in {"surface", "metar"}:
                    raise ValueError(f"Experiment {experiment} has surface obs but invalid obs_modality={obs_modality}")
                measurement[obs_modality] = self.as_weighted_ensemble_measurement(
                    surface_locs,
                    surface_vals,
                    surface_weights,
                )
                for i in range(len(self.model_vars)):
                    if len(surface_vals[i]):
                        ch_locs[i] = np.concatenate([ch_locs[i], surface_locs[i]], axis=0)
                        ch_vals[i] = np.concatenate([ch_vals[i], surface_vals[i]], axis=0)
            if obs_space == "surface_superob_grid":
                obs_modality = spec["obs_modality"]
                if obs_modality not in {"surface", "metar"}:
                    raise ValueError(f"Experiment {experiment} has surface grid obs but invalid obs_modality={obs_modality}")
                measurement[obs_modality] = {
                    "kind": "multi_grid",
                    "grids": surface_grids,
                    "masks": surface_masks,
                    "channel_indices": surface_channel_indices,
                }
            if not measurement:
                measurement = None

        nonempty = [(self.model_vars[i], len(v)) for i, v in enumerate(ch_vals) if len(v)]
        if obs_space == "aircraft_superob_grid" and aircraft_masks is not None:
            grid_valid_cells = int(sum(mask.sum() for mask in aircraft_masks))
        elif obs_space == "aircraft_superob_grid_split" and aircraft_split_grid is not None:
            grid_valid_cells = int(
                sum(int(mask.sum()) for grids, masks, indices, count in aircraft_split_grid.values() for mask in masks)
            )
        elif obs_space == "surface_superob_grid" and surface_masks is not None:
            grid_valid_cells = int(sum(mask.sum() for mask in surface_masks))
        elif obs_space == "aircraft_surface_sparse":
            aircraft_cells = json.loads(aircraft_meta.get("aircraft_covered_era5_cells_json", "{}"))
            surface_cells = json.loads(surface_meta.get("surface_covered_era5_cells_json", "{}"))
            grid_valid_cells = int(sum(int(v) for v in aircraft_cells.values()))
            grid_valid_cells += int(sum(int(v) for v in surface_cells.values()))
        elif obs_space == "aircraft_surface_superob_grid":
            grid_valid_cells = int(sum(mask.sum() for mask in aircraft_masks))
            grid_valid_cells += int(sum(mask.sum() for mask in surface_masks))
        else:
            grid_valid_cells = int(obs_mask.sum()) if obs_mask is not None else 0
        grid_native_points = int(obs_meta.get("native_common_points", "0")) if obs_space == "grid" else 0
        if obs_space in {"aircraft_weighted_sparse", "aircraft_simple_sparse", "aircraft_superob_grid"}:
            aircraft_counts = json.loads(obs_meta.get("aircraft_counts_json", "{}"))
            grid_native_points = int(sum(int(v) for v in aircraft_counts.values()))
        elif obs_space == "aircraft_weighted_sparse_split" and aircraft_split_sparse is not None:
            grid_native_points = int(
                sum(len(vals[i]) for locs, vals, weights in aircraft_split_sparse.values() for i in range(len(vals)))
            )
        elif obs_space == "aircraft_superob_grid_split" and aircraft_split_grid is not None:
            grid_native_points = int(
                sum(int(count.sum()) for grids, masks, indices, count in aircraft_split_grid.values())
            )
        elif obs_space in {"surface_weighted_sparse", "surface_simple_sparse", "surface_superob_grid"}:
            surface_counts = json.loads(obs_meta.get("surface_counts_json", "{}"))
            grid_native_points = int(sum(int(v) for v in surface_counts.values()))
        elif obs_space == "aircraft_surface_sparse":
            aircraft_counts = json.loads(aircraft_meta.get("aircraft_counts_json", "{}"))
            surface_counts = json.loads(surface_meta.get("surface_counts_json", "{}"))
            grid_native_points = int(sum(int(v) for v in aircraft_counts.values()))
            grid_native_points += int(sum(int(v) for v in surface_counts.values()))
        elif obs_space == "aircraft_surface_superob_grid":
            aircraft_counts = json.loads(obs_meta["aircraft_meta"].get("aircraft_counts_json", "{}"))
            surface_counts = json.loads(obs_meta["surface_meta"].get("surface_counts_json", "{}"))
            grid_native_points = int(sum(int(v) for v in aircraft_counts.values()))
            grid_native_points += int(sum(int(v) for v in surface_counts.values()))
        io.log0(
            f"Observation summary | experiment={experiment} obs_points={len(obs_vals)} "
            f"grid_valid_cells={grid_valid_cells} grid_native_points={grid_native_points} "
            f"likelihood_mode={self.likelihood_mode} nonempty_channels={len(nonempty)} "
            f"total_condition_points={sum(n for _, n in nonempty)}"
        )

        condition, _ = self.dataset.__getitem__(timestep)
        condition = condition.float()[None, :].to(self.device)

        outs = []
        for i in range(self.ens):
            generator = torch.Generator(device=self.device).manual_seed(self.seed + i)
            sample = self.sample_fn(
                measurement,
                generator,
                condition=condition,
                in_shape=self.in_shape,
                device=self.device,
                num_steps=self.num_steps,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                rho=self.rho,
                S_churn=self.S_churn,
                S_min=self.S_min,
                S_max=self.S_max,
                S_noise=self.S_noise,
                **self.likelihood_kwargs,
            ).detach().cpu().numpy()
            outs.append(sample.squeeze())
            io.log0(f"Completed {experiment} t{timestep:04d} sample {i + 1}/{self.ens}")

        out_array = np.asarray(outs, dtype=np.float32)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        np.save(output, out_array)

        meta_output = output[:-4] + "_measurement_summary.npz"
        np.savez_compressed(
            meta_output,
            timestep=np.asarray(timestep, dtype=np.int64),
            target_datetime=np.asarray(target_dt.isoformat()),
            experiment=np.asarray(experiment),
            description=np.asarray(spec["description"]),
            use_igra=np.asarray(spec["use_igra"]),
            obs_mode=np.asarray(spec["obs_mode"]),
            ens=np.asarray(self.ens, dtype=np.int64),
            num_steps=np.asarray(self.num_steps, dtype=np.int64),
            seed=np.asarray(self.seed, dtype=np.int64),
            checkpoint=np.asarray(self.checkpoint),
            igra_pkl=np.asarray(self.igra_pkl if spec["use_igra"] else ""),
            channel_names=np.asarray(self.model_vars),
            channel_counts=np.asarray([len(v) for v in ch_vals], dtype=np.int64),
            obs_points=np.asarray(len(obs_vals), dtype=np.int64),
            obs_space=np.asarray(obs_space),
            grid_valid_cells=np.asarray(grid_valid_cells, dtype=np.int64),
            grid_native_points=np.asarray(grid_native_points, dtype=np.int64),
            grid_count=np.asarray(obs_count if obs_count is not None else np.zeros((0,), dtype=np.int64), dtype=np.int64),
            grid_mask=np.asarray(obs_mask if obs_mask is not None else np.zeros((0,), dtype=bool), dtype=bool),
            wind_channel_indices=np.asarray(aircraft_channel_indices if aircraft_channel_indices is not None else np.zeros((0,), dtype=np.int64), dtype=np.int64),
            wind_grid_masks=np.asarray(aircraft_masks if aircraft_masks is not None else np.zeros((0,), dtype=bool), dtype=bool),
            surface_channel_indices=np.asarray(surface_channel_indices if surface_channel_indices is not None else np.zeros((0,), dtype=np.int64), dtype=np.int64),
            surface_grid_masks=np.asarray(surface_masks if surface_masks is not None else np.zeros((0,), dtype=bool), dtype=bool),
            airtemp_mask_policy=np.asarray(self.airtemp_mask_policy),
            obs_meta_json=np.asarray(json.dumps(obs_meta, sort_keys=True)),
            likelihood_mode=np.asarray(self.likelihood_mode),
            likelihood_kwargs_json=np.asarray(json.dumps(self.likelihood_kwargs, sort_keys=True)),
        )
        io.log0(f"Saved samples to {output} shape={out_array.shape}")
        io.log0(f"Saved measurement summary to {meta_output}")
        return output

    def run_many(self, experiments: Iterable[str], timesteps: Iterable[int], overwrite: bool = False) -> None:
        for timestep in timesteps:
            for experiment in experiments:
                self.run_one(experiment=experiment, timestep=int(timestep), overwrite=overwrite)


def write_run_manifest(output_root: str, args: argparse.Namespace, timesteps: List[int], experiments: List[str]) -> None:
    os.makedirs(output_root, exist_ok=True)
    manifest = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "script": os.environ.get("IGRA_RUNNER_ENTRYPOINT", os.path.abspath(__file__)),
        "output_root": output_root,
        "samples_root": os.path.join(output_root, "samples"),
        "experiments": experiments,
        "experiment_definitions": EXPERIMENTS,
        "timesteps": timesteps,
        "ens": args.ens,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "era5_root": args.era5_root,
        "era5_split": args.era5_split,
        "calendar_year": args.calendar_year,
        "hydra_cfg": args.hydra_cfg,
        "num_channels": args.num_channels,
        "raw_goes_max_points": args.raw_goes_max_points,
        "aircraft_around5_root": args.aircraft_around5_root,
        "aircraft_around25_root": args.aircraft_around25_root,
        "surface_metar_root": args.surface_metar_root,
        "igra_pkl": args.igra_pkl,
        "aircraft_clean_root": args.aircraft_clean_root,
        "aircraft_mid_root": args.aircraft_mid_root,
        "airtemp_mask_policy": args.airtemp_mask_policy,
        "likelihood_mode": args.likelihood_mode,
        "likelihood_params": {
            "std_igra": args.std_igra,
            "gamma_igra": args.gamma_igra,
            "lambda_igra": args.lambda_igra,
            "std_goes": args.std_goes,
            "gamma_goes": args.gamma_goes,
            "lambda_goes": args.lambda_goes,
            "std_airtemp": args.std_airtemp,
            "gamma_airtemp": args.gamma_airtemp,
            "lambda_airtemp": args.lambda_airtemp,
            "std_aircraft": args.std_aircraft,
            "gamma_aircraft": args.gamma_aircraft,
            "lambda_aircraft": args.lambda_aircraft,
            "std_surface": args.std_surface,
            "gamma_surface": args.gamma_surface,
            "lambda_surface": args.lambda_surface,
            "std_aircraft_acars": args.std_aircraft_acars,
            "gamma_aircraft_acars": args.gamma_aircraft_acars,
            "lambda_aircraft_acars": args.lambda_aircraft_acars,
            "std_aircraft_profiles": args.std_aircraft_profiles,
            "gamma_aircraft_profiles": args.gamma_aircraft_profiles,
            "lambda_aircraft_profiles": args.lambda_aircraft_profiles,
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "notes": "Persistent runner for MADIS aircraft observations. In multimodal mode, IGRA and aircraft are separate likelihood terms.",
    }
    with open(os.path.join(output_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description="Persistent MADIS aircraft posterior runner")
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--experiments",
        default="igra_only,igra_all13_aircraft_clean_v2,igra_all13_aircraft_clean_v3,igra_all13_aircraft_mid_v2,igra_all13_aircraft_mid_v3",
        help=f"Comma-separated experiment names. Choices: {','.join(EXPERIMENTS)}",
    )
    parser.add_argument(
        "--timesteps",
        default="default16",
        help="Comma/range list, e.g. 0-3,364-367, or 'default16'/'first20'/'first20_12h'.",
    )
    parser.add_argument("--ens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--sigma_min", type=float, default=0.005)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--S_churn", type=float, default=0.0)
    parser.add_argument("--S_min", type=float, default=0.01)
    parser.add_argument("--S_max", type=float, default=50.0)
    parser.add_argument("--S_noise", type=float, default=1.003)
    parser.add_argument("--airtemp_mask_policy", choices=["official", "physical"], default="physical")
    parser.add_argument("--aircraft_around5_root", default=AIRCRAFT_AROUND5_ROOT)
    parser.add_argument("--aircraft_around25_root", default=AIRCRAFT_AROUND25_ROOT)
    parser.add_argument("--surface_metar_root", default=SURFACE_METAR_STRAT24_ROOT)
    parser.add_argument("--aircraft_clean_root", default=AIRCRAFT_CLEAN_ROOT)
    parser.add_argument("--aircraft_mid_root", default=AIRCRAFT_MID_ROOT)
    parser.add_argument("--igra_pkl", default=IGRA_PKL)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--era5_root", default=ERA5_ROOT)
    parser.add_argument("--era5_split", default="test")
    parser.add_argument("--calendar_year", type=int, default=2020)
    parser.add_argument("--hydra_cfg", default=HYDRA_CFG)
    parser.add_argument("--num_channels", type=int, default=NUM_CHANNELS)
    parser.add_argument(
        "--raw_goes_max_points",
        type=int,
        default=0,
        help="Optional random-without-replacement cap for raw GOES LST observations; 0 keeps all valid points.",
    )
    parser.add_argument("--likelihood_mode", choices=["legacy", "multimodal"], default="multimodal")
    parser.add_argument("--std_igra", type=float, default=5e-4)
    parser.add_argument("--gamma_igra", type=float, default=2e-6)
    parser.add_argument("--lambda_igra", type=float, default=1.0)
    parser.add_argument("--std_goes", type=float, default=5e-4)
    parser.add_argument("--gamma_goes", type=float, default=2e-6)
    parser.add_argument("--lambda_goes", type=float, default=1.0)
    parser.add_argument("--std_airtemp", type=float, default=5e-4)
    parser.add_argument("--gamma_airtemp", type=float, default=2e-6)
    parser.add_argument("--lambda_airtemp", type=float, default=1.0)
    parser.add_argument("--std_aircraft", type=float, default=5e-4)
    parser.add_argument("--gamma_aircraft", type=float, default=2e-6)
    parser.add_argument("--lambda_aircraft", type=float, default=1.0)
    parser.add_argument("--std_surface", type=float, default=5e-4)
    parser.add_argument("--gamma_surface", type=float, default=2e-6)
    parser.add_argument("--lambda_surface", type=float, default=1.0)
    parser.add_argument("--std_aircraft_acars", type=float, default=None)
    parser.add_argument("--gamma_aircraft_acars", type=float, default=None)
    parser.add_argument("--lambda_aircraft_acars", type=float, default=None)
    parser.add_argument("--std_aircraft_profiles", type=float, default=None)
    parser.add_argument("--gamma_aircraft_profiles", type=float, default=None)
    parser.add_argument("--lambda_aircraft_profiles", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    experiments = [x.strip() for x in args.experiments.split(",") if x.strip()]
    unknown = sorted(set(experiments) - set(EXPERIMENTS))
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}; choices={sorted(EXPERIMENTS)}")
    if args.timesteps == "default16":
        timesteps = default_timesteps_16()
    elif args.timesteps == "first20":
        timesteps = default_timesteps_first20()
    elif args.timesteps == "first20_12h":
        timesteps = default_timesteps_first20_12h()
    else:
        timesteps = parse_timesteps(args.timesteps)
    write_run_manifest(args.output_root, args, timesteps, experiments)

    runner = PersistentFullPoolRunner(
        output_root=args.output_root,
        ens=args.ens,
        seed=args.seed,
        num_steps=args.num_steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=args.rho,
        S_churn=args.S_churn,
        S_min=args.S_min,
        S_max=args.S_max,
        S_noise=args.S_noise,
        airtemp_mask_policy=args.airtemp_mask_policy,
        igra_pkl=args.igra_pkl,
        aircraft_around5_root=args.aircraft_around5_root,
        aircraft_around25_root=args.aircraft_around25_root,
        aircraft_clean_root=args.aircraft_clean_root,
        aircraft_mid_root=args.aircraft_mid_root,
        surface_metar_root=args.surface_metar_root,
        checkpoint=args.checkpoint,
        era5_root=args.era5_root,
        hydra_cfg=args.hydra_cfg,
        num_channels=args.num_channels,
        raw_goes_max_points=args.raw_goes_max_points,
        likelihood_mode=args.likelihood_mode,
        std_igra=args.std_igra,
        gamma_igra=args.gamma_igra,
        lambda_igra=args.lambda_igra,
        std_goes=args.std_goes,
        gamma_goes=args.gamma_goes,
        lambda_goes=args.lambda_goes,
        std_airtemp=args.std_airtemp,
        gamma_airtemp=args.gamma_airtemp,
        lambda_airtemp=args.lambda_airtemp,
        std_aircraft=args.std_aircraft,
        gamma_aircraft=args.gamma_aircraft,
        lambda_aircraft=args.lambda_aircraft,
        std_surface=args.std_surface,
        gamma_surface=args.gamma_surface,
        lambda_surface=args.lambda_surface,
        std_aircraft_acars=args.std_aircraft_acars,
        gamma_aircraft_acars=args.gamma_aircraft_acars,
        lambda_aircraft_acars=args.lambda_aircraft_acars,
        std_aircraft_profiles=args.std_aircraft_profiles,
        gamma_aircraft_profiles=args.gamma_aircraft_profiles,
        lambda_aircraft_profiles=args.lambda_aircraft_profiles,
        era5_split=args.era5_split,
        calendar_year=args.calendar_year,
    )
    runner.run_many(experiments=experiments, timesteps=timesteps, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

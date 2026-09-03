import os
import torch
from functools import partial
from torch.nn import functional as F
import numpy as np


# Per-step DPS diagnostics collected when env var DPS_TRACE=1.  Each entry is a
# dict {sigma, err, grad_abs_mean, grad_sat_frac, score_norm, n_obs}.  Scripts
# can read/clear this via:
#
#     from igra_gen.generating.conditioning_methods import DPS_TRACE
#     DPS_TRACE.clear()
#     ... run sampler ...
#     print(DPS_TRACE)
DPS_TRACE: list = []


def _dps_trace_enabled() -> bool:
    return os.environ.get("DPS_TRACE", "0") not in ("0", "", "false", "False")


class ConditioningMethod:
    def __init__(self, operator , num_sampling = 1, scale=1, **kwargs):
        super().__init__()
        self.operator = operator
        self.num_sampling = num_sampling
        self.scale = scale

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, sigma,
                     std_sr=1e-1, gamma_sr=5e-3, std_igra=5e-4, gamma_igra=2e-6,
                     std_goes=5e-4, gamma_goes=2e-6,
                     std_airtemp=5e-4, gamma_airtemp=2e-6,
                     std_goes_wind=5e-4, gamma_goes_wind=2e-6,
                     std_aircraft=5e-4, gamma_aircraft=2e-6,
                     std_surface=5e-4, gamma_surface=2e-6,
                     std_aircraft_acars=None, gamma_aircraft_acars=None,
                     std_aircraft_profiles=None, gamma_aircraft_profiles=None,
                     lambda_igra=1.0, lambda_goes=1.0, lambda_airtemp=1.0,
                     lambda_goes_wind=1.0, lambda_aircraft=1.0,
                     lambda_surface=1.0,
                     lambda_aircraft_acars=None, lambda_aircraft_profiles=None,
                     mu=1, beta=1, lda=0.25, retain_graph=False, **kwargs):
        """
        Unified conditioning method that handles SR, IGRA, and SR+IGRA based on measurement type.

        measurement can be:
        - None: unconditional (no conditioning)
        - tensor: SR conditioning
        - [query_locations, true_values]: IGRA conditioning
        - [low_res_tensor, [query_locations, true_values]]: SR+IGRA conditioning
        """
        if measurement is None:
            # Unconditional - return zero score
            return torch.zeros_like(x_prev), 0.0

        # Ensure measurement tensors are on the same device as x_0_hat
        device = x_0_hat.device
        if torch.is_tensor(measurement):
            measurement = measurement.to(device)
        elif isinstance(measurement, list) and len(measurement) == 2:
            # For SR+IGRA case, convert tensor part to device
            if torch.is_tensor(measurement[0]):
                measurement[0] = measurement[0].to(device)
            # IGRA part stays as lists (converted to tensors internally in error_function)

        # Modality-specific sparse observations:
        # {
        #   "igra": [query_locations, true_values],
        #   "goes": [query_locations, true_values],
        #   "airtemp": [query_locations, true_values],
        # }
        # Each modality gets its own variance and lambda weight.  Empty
        # channels are skipped in sparse_error_function, so dense products do
        # not inherit the old IGRA empty-channel denominator.
        if isinstance(measurement, dict):
            std_aircraft_acars = std_aircraft if std_aircraft_acars is None else std_aircraft_acars
            gamma_aircraft_acars = gamma_aircraft if gamma_aircraft_acars is None else gamma_aircraft_acars
            lambda_aircraft_acars = lambda_aircraft if lambda_aircraft_acars is None else lambda_aircraft_acars
            std_aircraft_profiles = std_aircraft if std_aircraft_profiles is None else std_aircraft_profiles
            gamma_aircraft_profiles = gamma_aircraft if gamma_aircraft_profiles is None else gamma_aircraft_profiles
            lambda_aircraft_profiles = lambda_aircraft if lambda_aircraft_profiles is None else lambda_aircraft_profiles
            specs = {
                "igra": (std_igra, gamma_igra, lambda_igra),
                "goes": (std_goes, gamma_goes, lambda_goes),
                "airtemp": (std_airtemp, gamma_airtemp, lambda_airtemp),
                "goes_wind": (std_goes_wind, gamma_goes_wind, lambda_goes_wind),
                "aircraft": (std_aircraft, gamma_aircraft, lambda_aircraft),
                "surface": (std_surface, gamma_surface, lambda_surface),
                "metar": (std_surface, gamma_surface, lambda_surface),
                "aircraft_acars": (std_aircraft_acars, gamma_aircraft_acars, lambda_aircraft_acars),
                "aircraft_profiles": (std_aircraft_profiles, gamma_aircraft_profiles, lambda_aircraft_profiles),
            }
            log_p = torch.zeros((), device=device, dtype=x_0_hat.dtype)
            err_items = {}
            active = 0
            for name, (std_i, gamma_i, lambda_i) in specs.items():
                payload = measurement.get(name)
                if payload is None:
                    continue
                if isinstance(payload, dict) and payload.get("kind") == "grid":
                    err_i = self.operator.gridded_error_function(
                        x_0_hat,
                        payload["grid"],
                        payload["mask"],
                        channel_idx=int(payload["channel_idx"]),
                    )
                elif isinstance(payload, dict) and payload.get("kind") == "multi_grid":
                    err_i = self.operator.multi_gridded_error_function(
                        x_0_hat,
                        payload["grids"],
                        payload["masks"],
                        payload["channel_indices"],
                    )
                elif isinstance(payload, dict) and payload.get("kind") == "weighted_sparse":
                    err_i = self.operator.weighted_sparse_error_function(
                        x_0_hat,
                        payload["query_locations"],
                        payload["true_values"],
                        payload["weights"],
                    )
                else:
                    query_locations, true_values = payload
                    err_i = self.operator.sparse_error_function(x_0_hat, query_locations, true_values)
                var_i = std_i**2 + gamma_i * (sigma / mu)**2
                log_p = log_p - lambda_i * (err_i / var_i).sum() / 2
                err_items[name] = float(err_i.detach().mean().item())
                active += 1
            if active == 0:
                return torch.zeros_like(x_prev), {}
            grad = torch.autograd.grad(outputs=log_p, inputs=x_prev, retain_graph=retain_graph)[0]
            grad_clipped = grad.clamp(min=-1.0, max=1.0)
            scaled_score = grad_clipped * self.scale * sigma
            if _dps_trace_enabled():
                with torch.no_grad():
                    g_abs = grad.abs()
                    DPS_TRACE.append({
                        "sigma": float(sigma) if not torch.is_tensor(sigma) else float(sigma.item()),
                        "err": err_items,
                        "grad_abs_mean": float(g_abs.mean().item()),
                        "grad_abs_max": float(g_abs.max().item()),
                        "grad_sat_frac": float((g_abs >= 1.0).float().mean().item()),
                        "score_abs_mean": float(scaled_score.abs().mean().item()),
                        "score_abs_max": float(scaled_score.abs().max().item()),
                        "scale": float(self.scale),
                        "lambda_igra": float(lambda_igra),
                        "lambda_goes": float(lambda_goes),
                        "lambda_airtemp": float(lambda_airtemp),
                        "lambda_goes_wind": float(lambda_goes_wind),
                        "lambda_aircraft": float(lambda_aircraft),
                        "lambda_surface": float(lambda_surface),
                    })
            return scaled_score, err_items

        # Detect conditioning type based on measurement structure
        if isinstance(measurement, list):
            if len(measurement) == 3:
                # GOES: [locations, values, target_indices]
                # locations: (N, 2), values: (N, 2), target_indices: (u_idx, v_idx)
                # Ensure tensors are on correct device
                locations, values, target_indices = measurement
                if torch.is_tensor(locations): locations = locations.to(device)
                if torch.is_tensor(values): values = values.to(device)
                
                # We need a list of tensors for the operator interface (batch processing)
                # Assuming batch size 1 for simplicity here as sample_goes.py does manual loop
                # If batch size > 1, the input should already be a list of tensors.
                # Here we wrap single tensors in a list to match the batch-loop in error_function
                if not isinstance(locations, list) and locations.ndim == 2:
                    locations = [locations]
                    values = [values]
                    
                err_goes = self.operator.goes_error_function(x_0_hat, locations, values, target_indices)
                var = std_igra**2 + gamma_igra * (sigma/mu)**2 # Reusing IGRA variance params
                log_p = -(err_goes/var).sum()/2
                grad = torch.autograd.grad(outputs=log_p, inputs=x_prev, retain_graph=retain_graph)[0]
                grad_clipped = grad.clamp(min=-1.0, max=1.0)
                scaled_score = grad_clipped * self.scale * sigma
                return scaled_score, err_goes.mean().item()
                
            elif len(measurement) == 2:
                if torch.is_tensor(measurement[0]) and (isinstance(measurement[1], (list, tuple)) or hasattr(measurement[1], '__len__')):
                    # SR+IGRA: [low_res_tensor, [query_locations, true_values]]
                    lr, igra_data = measurement
                    err_sr = (lr[:, :, 1:, :] - self.operator.forward(x_0_hat)[:, :, 1:, :])**2
                    err_igra = self.operator.error_function(x_0_hat, igra_data[0], igra_data[1])
                    var_sr = std_sr**2 + gamma_sr * (sigma/mu)**2
                    var_igra = std_igra**2 + gamma_igra * (sigma/mu)**2
                    log_p = -beta * (err_sr/var_sr).sum()/2 - lda * (err_igra/var_igra).sum()/2
                    grad = torch.autograd.grad(outputs=log_p, inputs=x_prev, retain_graph=retain_graph)[0]
                    grad_clipped = grad.clamp(min=-1.0, max=1.0)
                    scaled_score = grad_clipped * self.scale * sigma
                    return scaled_score, [err_sr.mean().item(), err_igra.mean().item()]
                elif (isinstance(measurement[0], (list, tuple)) or hasattr(measurement[0], '__len__')) and \
                      (isinstance(measurement[1], (list, tuple)) or hasattr(measurement[1], '__len__')):
                    # IGRA: [query_locations, true_values]
                    query_locations, true_values = measurement
                    err_igra = self.operator.error_function(x_0_hat, query_locations, true_values)
                    var = std_igra**2 + gamma_igra * (sigma/mu)**2
                    log_p = -(err_igra/var).sum()/2
                    grad = torch.autograd.grad(outputs=log_p, inputs=x_prev, retain_graph=retain_graph)[0]
                    grad_clipped = grad.clamp(min=-1.0, max=1.0)
                    scaled_score = grad_clipped * self.scale * sigma
                    if _dps_trace_enabled():
                        with torch.no_grad():
                            g_abs = grad.abs()
                            sat_frac = (g_abs >= 1.0).float().mean().item()
                            DPS_TRACE.append({
                                "sigma": float(sigma) if not torch.is_tensor(sigma) else float(sigma.item()),
                                "err": float(err_igra.mean().item()),
                                "grad_abs_mean": float(g_abs.mean().item()),
                                "grad_abs_max": float(g_abs.max().item()),
                                "grad_sat_frac": float(sat_frac),
                                "score_abs_mean": float(scaled_score.abs().mean().item()),
                                "score_abs_max": float(scaled_score.abs().max().item()),
                                "scale": float(self.scale),
                            })
                    return scaled_score, err_igra.mean().item()
                else:
                    # Debug: unexpected list structure
                    raise ValueError(f"Unexpected measurement structure for list of length 2: types are {type(measurement[0])}, {type(measurement[1])}")
        else:
            # SR: low_res_tensor
            if not torch.is_tensor(measurement):
                raise ValueError(f"Expected tensor for SR conditioning, got {type(measurement)}")
            Ax_hat = self.operator.forward(x_0_hat)
            err1 = (measurement[:, :, 1:, :] - Ax_hat[:, :, 1:, :])**2
            var = std_sr**2 + gamma_sr * (sigma/mu)**2
            log_p = -(err1/var).sum()/2
            grad = torch.autograd.grad(outputs=log_p, inputs=x_prev, retain_graph=retain_graph)[0]
            scaled_score = grad * self.scale * sigma
            return scaled_score, err1.mean().item()
    
    

class SuperResolutionOperator:
    def __init__(self, in_shape=(16,32), target_shape=(128, 256), mode="bilinear"):
        super().__init__()
        self.scale = target_shape[0]//in_shape[0]
        self.up_sample = partial(F.interpolate, size=target_shape, mode=mode, align_corners=False)
        self.down_sample = partial(F.interpolate, size=in_shape, mode=mode, align_corners=False)

    def forward(self, data, **kwargs):
        return self.down_sample(data)  

    def transpose(self, data, **kwargs):
        return self.up_sample(data) 

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)
    
_ERA5_GRID_ROOT = os.environ.get(
    "IGRA_ERA5_GRID_ROOT",
    os.environ.get("ERA5_ROOT", ""),
)
_ERA5_LAT_PATH = os.path.join(_ERA5_GRID_ROOT, "lat.npy")
_ERA5_LON_PATH = os.path.join(_ERA5_GRID_ROOT, "lon.npy")


def _load_era5_grid(lat_path=None, lon_path=None):
    lat_path = _ERA5_LAT_PATH if lat_path is None else lat_path
    lon_path = _ERA5_LON_PATH if lon_path is None else lon_path
    return np.load(lat_path), np.load(lon_path)


class IGRAOperator:
    def __init__(self, lat=None, lon=None, mode="bilinear", lat_path=None, lon_path=None):
        super().__init__()
        if lat is None or lon is None:
            loaded_lat, loaded_lon = _load_era5_grid(lat_path=lat_path, lon_path=lon_path)
            if lat is None:
                lat = loaded_lat
            if lon is None:
                lon = loaded_lon
        self.lat = torch.tensor(lat).float()
        self.lon = torch.tensor(lon).float() 
        self.lon = self.lon - self.lon.max()/2 
        self.mode = mode
    
    def error_function(self, data,  query_locations,true_values, **kwargs):
        """
        Computes the mean error between interpolated and true values for a batch of images.
        
        Parameters:
        - data: torch tensor of shape (B, C, L, W), representing batch image values.
        - true_values: list of torch tensors, each of shape (B_i, C_i, N_i), true values at query points.
        - query_locations: list of torch tensors, each of shape (B_i, C_i, N_i, 2), query points ([lat, lon]).
        
        Returns:
        - mean_error: torch scalar tensor, representing the mean error.
        """
        batch_size = len(data)
        error = 0.
        counter=1.
        for b in range(batch_size):
            channel_size = len(true_values[b])
            for c in range(channel_size):
                x = data[b, c]  # Extracting specific channel for interpolation
                queries = torch.as_tensor(query_locations[b][c], device=x.device, dtype=torch.float32)
                true_vals = torch.as_tensor(true_values[b][c], device=x.device, dtype=torch.float32)
                if true_vals.numel() == 0: 
                    true_vals = torch.zeros(1, device=x.device, dtype=torch.float32)
                # Compute interpolated values using function H
                interpolated_vals = self.H(x, queries)
                #TODO: some channels have unexpected large differences from IGRA
                error += torch.mean((interpolated_vals - true_vals)**2)
                counter += 1
        return error / counter

    def sparse_error_function(self, data, query_locations, true_values, **kwargs):
        """
        Sparse point-observation MSE that skips empty channels.

        This is intended for modality-specific posterior likelihoods.  It keeps
        the same point interpolation H as error_function, but avoids the legacy
        empty-channel denominator behavior.
        """
        error = torch.zeros((), device=data.device, dtype=torch.float32)
        counter = 0
        for b in range(len(data)):
            channel_size = len(true_values[b])
            for c in range(channel_size):
                x = data[b, c]
                queries = torch.as_tensor(query_locations[b][c], device=x.device, dtype=torch.float32)
                true_vals = torch.as_tensor(true_values[b][c], device=x.device, dtype=torch.float32)
                if true_vals.numel() == 0:
                    continue
                interpolated_vals = self.H(x, queries)
                error = error + torch.mean((interpolated_vals - true_vals)**2)
                counter += 1
        if counter == 0:
            return torch.zeros((), device=data.device, dtype=torch.float32)
        return error / counter

    def weighted_sparse_error_function(self, data, query_locations, true_values, weights, **kwargs):
        """
        Sparse point-observation MSE with per-point weights.

        This is intended for clustered dense-ish products such as GOES DMW.
        Empty channels are skipped, and each active channel contributes
        sum(w * residual^2) / sum(w), so clusters can be balanced by assigning
        w_i = 1 / n_points_in_same_ERA5_cell.
        """
        error = torch.zeros((), device=data.device, dtype=torch.float32)
        counter = 0
        for b in range(len(data)):
            channel_size = len(true_values[b])
            for c in range(channel_size):
                x = data[b, c]
                queries = torch.as_tensor(query_locations[b][c], device=x.device, dtype=torch.float32)
                true_vals = torch.as_tensor(true_values[b][c], device=x.device, dtype=torch.float32)
                w = torch.as_tensor(weights[b][c], device=x.device, dtype=torch.float32)
                if true_vals.numel() == 0:
                    continue
                if w.numel() != true_vals.numel():
                    raise RuntimeError(
                        f"Weight length {w.numel()} does not match true value length {true_vals.numel()} "
                        f"for channel {c}"
                    )
                interpolated_vals = self.H(x, queries)
                w_sum = torch.clamp(w.sum(), min=1e-12)
                error = error + (w * (interpolated_vals - true_vals) ** 2).sum() / w_sum
                counter += 1
        if counter == 0:
            return torch.zeros((), device=data.device, dtype=torch.float32)
        return error / counter

    def gridded_error_function(self, data, obs_grid, mask_grid, channel_idx=0, **kwargs):
        """
        Masked grid-cell MSE for dense products that have already been
        aggregated onto the ERA5/posterior 128 x 256 grid.

        This avoids treating hundreds of thousands of native GOES/AirTemp
        pixels as independent point observations.  Each valid ERA5 grid cell
        contributes at most one residual.
        """
        obs = torch.as_tensor(obs_grid, device=data.device, dtype=torch.float32)
        mask = torch.as_tensor(mask_grid, device=data.device, dtype=torch.bool)
        if obs.ndim == 2:
            obs = obs.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        pred = data[:, int(channel_idx)].float()
        if obs.shape[0] == 1 and pred.shape[0] > 1:
            obs = obs.expand(pred.shape[0], -1, -1)
            mask = mask.expand(pred.shape[0], -1, -1)
        if obs.shape != pred.shape:
            raise RuntimeError(f"gridded obs shape {tuple(obs.shape)} does not match prediction {tuple(pred.shape)}")
        if int(mask.sum().item()) == 0:
            return torch.zeros((), device=data.device, dtype=torch.float32)
        return ((pred - obs) ** 2)[mask].mean()

    def multi_gridded_error_function(self, data, obs_grids, mask_grids, channel_indices, **kwargs):
        """
        Multi-channel masked grid-cell MSE.

        Used for GOES wind superobs: each wind component has its own ERA5-grid
        observation and mask, and active channels are averaged without empty
        channel dilution.
        """
        error = torch.zeros((), device=data.device, dtype=torch.float32)
        counter = 0
        for obs_grid, mask_grid, channel_idx in zip(obs_grids, mask_grids, channel_indices):
            err = self.gridded_error_function(data, obs_grid, mask_grid, channel_idx=int(channel_idx))
            mask = torch.as_tensor(mask_grid, device=data.device, dtype=torch.bool)
            if int(mask.sum().item()) == 0:
                continue
            error = error + err
            counter += 1
        if counter == 0:
            return torch.zeros((), device=data.device, dtype=torch.float32)
        return error / counter
    
    def H(self,x, query_points):
        """
        Interpolates values from an image given latitude and longitude grids using PyTorch.
        Parameters:
        - x: 2D torch tensor of shape (128, 256), representing image values.
        - lat_values: 1D torch tensor of shape (128,), latitude values corresponding to image rows.
        - lon_values: 1D torch tensor of shape (256,), longitude values corresponding to image columns.
        - query_points: 2D torch tensor of shape (N, 2), where each row is [lat, lon].
        Returns:
        - interpolated_values: 1D torch tensor of shape (N,), interpolated image values at query points.
        """
        x = torch.roll(x,shifts=128,dims=1)
        if query_points.shape[0]==0:
            interpolated_values = torch.tensor(0.).float()
        else: 
            # Normalize lat/lon to range [-1, 1] for grid_sample
            lat_min, lat_max = -90,90#self.lat.min(), self.lat.max()
            lon_min, lon_max = 0,360#self.lon.min(), self.lon.max()
            norm_lat = 2 * (query_points[:, 0] - lat_min) / (lat_max - lat_min) - 1
            norm_lon = 2 * (query_points[:, 1]+180 - lon_min) / (lon_max - lon_min) - 1
            # Create grid for interpolation with shape (1, N, 1, 2)
            grid = torch.stack((norm_lon, norm_lat), dim=-1).to(dtype=torch.float32).view(1, -1, 1, 2)
            x = x.unsqueeze(0).unsqueeze(0)
            interpolated_values = F.grid_sample(x.float(), grid, mode=self.mode, align_corners=True, padding_mode='reflection')

        # Reshape output from (1, 1, N, 1) to (N,)
        return interpolated_values.squeeze(0).squeeze(0).squeeze(-1)


class GOESOperator:
    def __init__(self, lat=None, lon=None, mode="bilinear"):
        super().__init__()
        # GOES operator uses the same interpolation logic as IGRA
        # If lat/lon are not provided, we assume the caller handles paths or defaults
        self.mode = mode

    def error_function(self, data, query_locations, true_values, target_indices, **kwargs):
        """
        Computes the mean error between interpolated and true values for GOES wind data.
        
        Parameters:
        - data: torch tensor of shape (B, C, L, W), representing batch image values.
          C=69 for ERA5. We need to select the correct U and V channels.
        - true_values: list of torch tensors (length B), each shape (N_points, 2). 
          The last dim 2 is [u_true, v_true].
        - query_locations: list of torch tensors (length B), each shape (N_points, 2). 
          The last dim 2 is [lat, lon].
        - target_indices: list or tuple of (u_channel_idx, v_channel_idx) corresponding to the pressure level.
          Example: for 500hPa, indices for 'u_component_of_wind_500' and 'v_component_of_wind_500'.
        
        Returns:
        - mean_error: torch scalar tensor
        """
        batch_size = len(data)
        error = 0.
        counter = 0.
        
        u_idx, v_idx = target_indices

        for b in range(batch_size):
            # Get U and V fields from the model prediction
            # shape (H, W)
            u_field = data[b, u_idx]
            v_field = data[b, v_idx]
            
            # Get queries and ground truth for this batch item
            # queries: (N, 2) [lat, lon]
            # truths: (N, 2) [u_true, v_true]
            queries = query_locations[b]
            truths = true_values[b]
            
            if queries.numel() == 0:
                continue
                
            # Ensure tensors are on correct device
            queries = queries.to(u_field.device).float()
            truths = truths.to(u_field.device).float()
            
            # Interpolate U and V at query locations
            # self.H returns shape (N,)
            u_interp = self.H(u_field, queries)
            v_interp = self.H(v_field, queries)
            
            # Compute MSE for U and V components separately and sum
            mse_u = torch.mean((u_interp - truths[:, 0])**2)
            mse_v = torch.mean((v_interp - truths[:, 1])**2)
            
            error += (mse_u + mse_v)
            counter += 1

        if counter == 0:
            return torch.tensor(0., device=data.device, requires_grad=True)
            
        return error / counter

    def H(self, x, query_points):
        """
        Interpolates values from an image given latitude and longitude coordinates.
        Uses the same logic as IGRAOperator.H
        """
        x = torch.roll(x, shifts=128, dims=1)
        
        if query_points.shape[0] == 0:
            return torch.tensor(0.).to(x.device)
            
        # Normalize lat/lon to range [-1, 1] for grid_sample
        # Assuming global coverage: lat [-90, 90], lon [0, 360]
        lat_min, lat_max = -90, 90
        lon_min, lon_max = 0, 360
        
        norm_lat = 2 * (query_points[:, 0] - lat_min) / (lat_max - lat_min) - 1
        # Shift lon by +180 (converting -180/180 or similar to 0/360 frame if needed)
        # Note: The original IGRA code adds 180. We stick to that convention.
        norm_lon = 2 * (query_points[:, 1] + 180 - lon_min) / (lon_max - lon_min) - 1
        
        # Create grid for interpolation with shape (1, N, 1, 2)
        grid = torch.stack((norm_lon, norm_lat), dim=-1).to(dtype=torch.float32).view(1, -1, 1, 2)
        
        x = x.unsqueeze(0).unsqueeze(0)
        interpolated_values = F.grid_sample(x.float(), grid, mode=self.mode, align_corners=True, padding_mode='reflection')

        # Reshape output from (1, 1, N, 1) to (N,)
        return interpolated_values.squeeze(0).squeeze(0).squeeze(-1)


class UnifiedOperator:
    """Unified operator that can handle SR, IGRA, and SR+IGRA conditioning"""
    def __init__(self, conditioning_type="sr", in_shape=(48,96), target_shape=(128, 256), mode="bilinear",
                 lat_path=_ERA5_LAT_PATH,
                 lon_path=_ERA5_LON_PATH):
        self.conditioning_type = conditioning_type
        if conditioning_type in ["sr", "sr_igra"]:
            self.sr_op = SuperResolutionOperator(in_shape=in_shape, target_shape=target_shape, mode=mode)
        if conditioning_type in ["igra", "sr_igra", "multimodal"]:
            try:
                self.igra_op = IGRAOperator(lat_path=lat_path, lon_path=lon_path, mode=mode)
            except FileNotFoundError:
                print(f"Warning: Could not load lat/lon files from {lat_path}, {lon_path}")
                # Try the configured local ERA5 grid path as a fallback.
                alt_lat_path = _ERA5_LAT_PATH
                alt_lon_path = _ERA5_LON_PATH
                try:
                    self.igra_op = IGRAOperator(lat_path=alt_lat_path, lon_path=alt_lon_path, mode=mode)
                    print(f"Loaded lat/lon from alternative paths: {alt_lat_path}, {alt_lon_path}")
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Could not load IGRA lat/lon grid from {lat_path}, {lon_path} "
                        f"or fallback {alt_lat_path}, {alt_lon_path}. "
                        "Set IGRA_ERA5_GRID_ROOT or pass lat_path/lon_path."
                    )
        if conditioning_type in ["goes", "sr_goes"]:
            self.goes_op = GOESOperator(mode=mode)

    def forward(self, data):
        """SR forward operation"""
        if hasattr(self, 'sr_op'):
            return self.sr_op.forward(data)
        else:
            raise ValueError(f"SR operation not supported for conditioning_type: {self.conditioning_type}")

    def error_function(self, data, query_locations, true_values):
        """IGRA error function"""
        if hasattr(self, 'igra_op'):
            return self.igra_op.error_function(data, query_locations, true_values)
        else:
            raise ValueError(f"IGRA operation not supported for conditioning_type: {self.conditioning_type}")

    def sparse_error_function(self, data, query_locations, true_values):
        """Sparse point-observation error with empty channels skipped."""
        if hasattr(self, 'igra_op'):
            return self.igra_op.sparse_error_function(data, query_locations, true_values)
        else:
            raise ValueError(f"Sparse point operation not supported for conditioning_type: {self.conditioning_type}")

    def gridded_error_function(self, data, obs_grid, mask_grid, channel_idx=0):
        """Masked grid-cell error for dense GOES/AirTemp products on the ERA5 grid."""
        if hasattr(self, 'igra_op'):
            return self.igra_op.gridded_error_function(data, obs_grid, mask_grid, channel_idx=channel_idx)
        else:
            raise ValueError(f"Gridded operation not supported for conditioning_type: {self.conditioning_type}")

    def weighted_sparse_error_function(self, data, query_locations, true_values, weights):
        """Weighted sparse point-observation error with empty channels skipped."""
        if hasattr(self, 'igra_op'):
            return self.igra_op.weighted_sparse_error_function(data, query_locations, true_values, weights)
        else:
            raise ValueError(f"Weighted sparse operation not supported for conditioning_type: {self.conditioning_type}")

    def multi_gridded_error_function(self, data, obs_grids, mask_grids, channel_indices):
        """Multi-channel masked grid-cell error."""
        if hasattr(self, 'igra_op'):
            return self.igra_op.multi_gridded_error_function(data, obs_grids, mask_grids, channel_indices)
        else:
            raise ValueError(f"Multi-grid operation not supported for conditioning_type: {self.conditioning_type}")

    def goes_error_function(self, data, query_locations, true_values, target_indices):
        """GOES error function"""
        if hasattr(self, 'goes_op'):
            return self.goes_op.error_function(data, query_locations, true_values, target_indices)
        else:
            raise ValueError(f"GOES operation not supported for conditioning_type: {self.conditioning_type}")


        

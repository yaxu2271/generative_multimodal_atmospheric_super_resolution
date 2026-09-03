import os
from glob import glob
from typing import Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from einops import rearrange

class ERA5SR(Dataset):
    def __init__(
        self,
        root: str,
        variables: list[str],
        conditions: int = 1, # last ones are the conditional variables
        split: str = "train",
        rescale = 1,
    ):
        super().__init__()
        self.root = root
        self.files = sorted(glob(os.path.join(root, split, "*.h5")))
        self.variables = variables
        self.conditions = conditions
        means = np.load(os.path.join(root, f"normalize_mean.npz"))
        self.means = np.stack([means[v] for v in variables], axis=0).reshape(-1, 1, 1)
        stds = np.load(os.path.join(root, f"normalize_std.npz"))
        self.stds = np.stack([stds[v] for v in variables], axis=0).reshape(-1, 1, 1)

        self.shape = self._load_file(
            self.files[np.random.randint(0, len(self.files))], variables
        ).shape
        self.scale = rescale

    @property
    def n_channels(self):
        assert len(self.shape) == 3
        return self.shape[0]*self.scale-self.conditions
    
    @property
    def cond_channels(self):
        return self.conditions

    @property
    def img_resolution(self):
        return self.shape[1], self.shape[2]

    def get_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        lat = np.load(os.path.join(self.root, "lat.npy")).astype(np.float32)
        lon = np.load(os.path.join(self.root, "lon.npy")).astype(np.float32)
        return lat, lon

    def get_time(self, idx: int) -> np.datetime64:
        with h5py.File(self.files[idx], "r") as f:
            timestamp = f["input"]["time"][()]
        return np.datetime64(timestamp.decode("utf-8"))

    def _load_file(self, path: str, variables: list[str]) -> np.ndarray:
        with h5py.File(path, "r") as f:
            data = {
                main_key: {
                    sub_key: np.array(value)
                    for sub_key, value in group.items()
                    if sub_key in variables + ["time"]
                }
                for main_key, group in f.items()
                if main_key in ["input"]
            }
        x = np.stack([data["input"][v] for v in variables], axis=0)
        return x

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        x =  (x - self.means) / self.stds
        return x

    def _unstandardize(self, x: np.ndarray) -> np.ndarray:
        # No surface geopotential in output
        x =  x * self.stds[:-1] + self.means[:-1]
        return x

    def __len__(self) -> int:
        return len(self.files)  # last files dont have a target

    def __getitem__(self, idx: int):
        t = torch.from_numpy(
            self._standardize(self._load_file(self.files[idx], self.variables)[:,:720,:1440])
        )
        x = t[-1:] 
        return x,t[:-1]  # C x H x W


if __name__ == "__main__":
    import time
    dataset = ERA5SR(
        "/depot/rmaulik/data/yangxu/data_from_DJ_original_NERSC/1.40625deg_from_full_res_1_step_6hr_h5df",
        split="train",
        variables = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "geopotential_50",
    "geopotential_100",
    "geopotential_150",
    "geopotential_200",
    "geopotential_250",
    "geopotential_300",
    "geopotential_400",
    "geopotential_500",
    "geopotential_600",
    "geopotential_700",
    "geopotential_850",
    "geopotential_925",
    "geopotential_1000",
    "u_component_of_wind_50",
    "u_component_of_wind_100",
    "u_component_of_wind_150",
    "u_component_of_wind_200",
    "u_component_of_wind_250",
    "u_component_of_wind_300",
    "u_component_of_wind_400",
    "u_component_of_wind_500",
    "u_component_of_wind_600",
    "u_component_of_wind_700",
    "u_component_of_wind_850",
    "u_component_of_wind_925",
    "u_component_of_wind_1000",
    "v_component_of_wind_50",
    "v_component_of_wind_100",
    "v_component_of_wind_150",
    "v_component_of_wind_200",
    "v_component_of_wind_250",
    "v_component_of_wind_300",
    "v_component_of_wind_400",
    "v_component_of_wind_500",
    "v_component_of_wind_600",
    "v_component_of_wind_700",
    "v_component_of_wind_850",
    "v_component_of_wind_925",
    "v_component_of_wind_1000",
    "temperature_50",
    "temperature_100",
    "temperature_150",
    "temperature_200",
    "temperature_250",
    "temperature_300",
    "temperature_400",
    "temperature_500",
    "temperature_600",
    "temperature_700",
    "temperature_850",
    "temperature_925",
    "temperature_1000",
    "specific_humidity_50",
    "specific_humidity_100",
    "specific_humidity_150",
    "specific_humidity_200",
    "specific_humidity_250",
    "specific_humidity_300",
    "specific_humidity_400",
    "specific_humidity_500",
    "specific_humidity_600",
    "specific_humidity_700",
    "specific_humidity_850",
    "specific_humidity_925",
    "specific_humidity_1000"
    ],

    )

    batch_size = 32  # Adjust as needed
    num_workers = 1  # Use >0 for multi-threaded loading, 0 for debugging

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Helps with GPU transfer speed
        prefetch_factor=1
    )

    num_batches = 1
    times = []

    # Loop to measure time over multiple batches
    for i, t in enumerate(dataloader):
        if i >= num_batches:
            break
        start_time = time.time()
        
        # Simulate a computation step (optional)
        # x = x.to(torch.float32)  # Ensuring batch is accessed
        t = t.to(torch.float32)
        print(t.shape,t.amax(dim=(0, 2,3)),t.amin(dim=(0, 2,3)),t.mean(dim=(0, 2,3)),t.std(dim=(0, 2,3)))
        # print(x.shape,x.amax(dim=(0, 2,3)),x.amin(dim=(0, 2,3)),x.mean(dim=(0, 2,3)),x.std(dim=(0, 2,3)))
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Batch {i+1}: {times[-1]:.4f} seconds")
        print(t.shape)
    avg_time = sum(times) / len(times)
    print(f"\nAverage time per batch ({num_batches} batches): {avg_time:.4f} seconds")


    # print(f"{x.mean()=}, {x.std()=}")
    # print(dataset.n_channels, dataset.img_resolution, len(dataset))

    # fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    # ax[0].imshow(x[0], cmap="coolwarm")
    # ax[1].imshow(t[0], cmap="coolwarm")
    # ax[2].imshow(t[0] - x[0], cmap="coolwarm")
    # for a in ax:
    #     a.axis("off")

    # ax[0].set_title("x")
    # ax[1].set_title("t")
    # ax[2].set_title("t - x")

    # fig.tight_layout()
    # fig.savefig("../media/era5.png", bbox_inches="tight", dpi=300)

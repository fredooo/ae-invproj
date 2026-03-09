import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm

from models import Decoder


@dataclass
class GradientMapData:
    grid: NDArray
    min: float
    mean: float
    max: float


class GradientMap:
    gradient_map_data: Optional[GradientMapData] = None

    def __init__(
        self,
        grid_size: int = 250,
        device: Optional[torch.device] = None,
        point_size: float = 2.0,
        point_color: str = "#BBBBBB",
        scale_max: float = -1.0,
        show_scale: bool = True,
        show_metrics: bool = True,
        tile_size: int = 128,
        flip_vertical: bool = False,
    ):
        self.grid_size = grid_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.point_size = point_size
        self.point_color = point_color
        self.scale_max = scale_max
        self.show_scale = show_scale
        self.show_metrics = show_metrics
        self.tile_size = min(tile_size, grid_size)
        self.flip_vertical = flip_vertical

    def gradients(self, model: Decoder, points: np.ndarray) -> GradientMapData:
        print("Preparing grid ...")
        x_max, x_min = np.max(points[:, 0]), np.min(points[:, 0])
        y_max, y_min = np.max(points[:, 1]), np.min(points[:, 1])
        pixel_width = (x_max - x_min) / self.grid_size
        pixel_height = (y_max - y_min) / self.grid_size

        grid_padding = self.grid_size + 2
        xx, yy = np.meshgrid(
            np.linspace(x_min - pixel_width, x_max + pixel_width, grid_padding),
            np.linspace(y_min - pixel_height, y_max + pixel_height, grid_padding),
        )
        xy = np.stack((xx, yy), axis=-1)  # shape: (grid_padding, grid_padding, 2)
        xy_tensor = torch.tensor(xy, dtype=torch.float32, device=self.device)

        grad_map = np.zeros((grid_padding - 2, grid_padding - 2), dtype=np.float32)

        tiles = [
            (i, j)
            for i in range(0, grid_padding - 2, self.tile_size)
            for j in range(0, grid_padding - 2, self.tile_size)
        ]

        for i, j in tqdm(tiles, desc="Computing gradients on tiles"):
            # define region of grad_map this tile will fill
            i_end = min(i + self.tile_size, grid_padding - 2)
            j_end = min(j + self.tile_size, grid_padding - 2)

            # corresponding region in the padded xy grid (add 2 for padding)
            i_slice = slice(i, i_end + 2)
            j_slice = slice(j, j_end + 2)

            tile_xy = xy_tensor[i_slice, j_slice].reshape(-1, 2)

            with torch.no_grad():
                tile_decoded = model.decode(tile_xy).cpu()
            tile_decoded = tile_decoded.view(i_slice.stop - i_slice.start, j_slice.stop - j_slice.start, -1).numpy()

            # compute gradients inside this tile (skip 1-pixel padding)
            dx = (tile_decoded[2:, 1:-1] - tile_decoded[:-2, 1:-1]) / (2 * pixel_width)
            dy = (tile_decoded[1:-1, 2:] - tile_decoded[1:-1, :-2]) / (2 * pixel_height)
            grad_magnitude = np.sqrt(np.sum(dx**2 + dy**2, axis=-1))

            # place the valid gradients into grad_map
            grad_map[i:i_end, j:j_end] = grad_magnitude

        print("Calculating min, mean and max ...")

        grad_min = grad_map.min().item()
        grad_mean = grad_map.mean().item()
        grad_max = grad_map.max().item()

        print(f"Gradients: min = {grad_min:.4f}, mean = {grad_mean:.4f}, max = {grad_max:.4f}")

        self.gradient_map_data = GradientMapData(grad_map, grad_min, grad_mean, grad_max)

        return self.gradient_map_data

    def plot(self, inv_model: Decoder, points: np.ndarray, file_path: Optional[str] = None) -> None:
        if self.gradient_map_data is None:
            self.gradients(inv_model, points)

        assert self.gradient_map_data is not None, "gradient_map_data was not set by gradients()"

        # Scale 2D coords for scatter overlay
        X2d_scaled = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))

        if self.flip_vertical:
            X2d_scaled[:, 1] = 1.0 - X2d_scaled[:, 1]

        grad_map = self.gradient_map_data.grid
        if not self.flip_vertical:
            grad_map = np.flip(grad_map, axis=0)
        scale_max = self.gradient_map_data.max if self.scale_max == -1 else self.scale_max

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(grad_map, cmap="magma", extent=(0, 1, 0, 1), vmin=0, vmax=scale_max)

        if self.point_size > 0.0:
            ax.scatter(X2d_scaled[:, 0], X2d_scaled[:, 1], c=self.point_color, s=self.point_size, edgecolors="none")

        ax.axis("off")
        plt.gca().set_aspect("equal")

        if self.show_metrics:
            ax.text(
                0.005,
                0.005,
                f"avg: {self.gradient_map_data.mean:.2f}",
                ha="left",
                va="bottom",
                fontsize=12,
                color="white",
                transform=ax.transAxes,
                bbox=dict(facecolor="black", edgecolor="none", pad=2),
            )
            ax.text(
                0.993,
                0.005,
                f"max: {self.gradient_map_data.max:.2f}",
                ha="right",
                va="bottom",
                fontsize=12,
                color="white",
                transform=ax.transAxes,
                bbox=dict(facecolor="black", edgecolor="none", pad=2),
            )

        if self.show_scale:
            fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.0462, pad=0.02, shrink=0.9, aspect=30)

        if file_path is not None:
            print("Saving to", file_path, "...")
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            # Draw canvas to compute tight bbox size
            fig.canvas.draw()

            # Get tight bounding box in inches
            tight_bbox_inches = fig.get_tightbbox(fig.canvas.get_renderer())  # type: ignore
            bbox_height_in = tight_bbox_inches.height

            # Calculate dpi to match desired height (in pixels)
            target_pixel_height = self.grid_size
            dpi = target_pixel_height / bbox_height_in

            # Save with calculated dpi and tight bounding box
            fig.savefig(file_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
        else:
            plt.show()

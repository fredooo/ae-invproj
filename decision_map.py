import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm
from tqdm import tqdm

from models import Decoder


class DecisionMap:
    def __init__(
        self,
        grid_size: int = 250,
        device: Optional[torch.device] = None,
        point_size: float = 6.0,
        point_color: str = "black",
        tile_size: int = 128,
        flip_vertical: bool = False,
    ):
        self.grid_size = grid_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.point_size = point_size
        self.point_color = point_color
        self.tile_size = min(tile_size, grid_size)
        self.flip_vertical = flip_vertical

    def remap_labels(self, labels: np.ndarray):
        unique = np.unique(labels)
        label_map = {label: i for i, label in enumerate(unique)}
        remapped = np.vectorize(label_map.get)(labels)
        return remapped, label_map

    def create_label_grid(self, model: Decoder, classifier):
        print("Preparing grid ...")
        pixel_width = self.x_range / self.grid_size
        pixel_height = self.y_range / self.grid_size

        xx, yy = np.meshgrid(
            np.linspace(self.min_x - pixel_width, self.max_x + pixel_width, self.grid_size),
            np.linspace(self.min_y - pixel_height, self.max_y + pixel_height, self.grid_size),
        )
        xy = np.stack((xx, yy), axis=-1)  # shape: (grid_padding, grid_padding, 2)
        xy_tensor = torch.tensor(xy, dtype=torch.float32, device=self.device)

        label_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        tiles = [
            (i, j) for i in range(0, self.grid_size, self.tile_size) for j in range(0, self.grid_size, self.tile_size)
        ]

        for i, j in tqdm(tiles, desc="Computing decisions on tiles"):
            # define region of nd_grid this tile will fill
            i_end = min(i + self.tile_size, self.grid_size)
            j_end = min(j + self.tile_size, self.grid_size)

            # corresponding region in the padded xy grid
            i_slice = slice(i, i_end)
            j_slice = slice(j, j_end)

            tile_xy = xy_tensor[i_slice, j_slice].reshape(-1, 2)

            with torch.no_grad():
                tile_decoded = model.decode(tile_xy).cpu()
            tile_decoded = tile_decoded.view(i_slice.stop - i_slice.start, j_slice.stop - j_slice.start, -1).numpy()

            # Flatten tile for prediction
            tile_flat = tile_decoded.reshape(-1, tile_decoded.shape[-1])
            tile_pred = classifier.predict(tile_flat).reshape(
                i_slice.stop - i_slice.start, j_slice.stop - j_slice.start
            )

            label_grid[i:i_end, j:j_end] = tile_pred

        return label_grid

    def plot(self, inv_model: Decoder, points, labels, classifier, file_path=None):
        self.min_x, self.min_y = points.min(axis=0)
        self.max_x, self.max_y = points.max(axis=0)

        self.x_range = self.max_x - self.min_x
        self.y_range = self.max_y - self.min_y

        if self.x_range > self.y_range:
            pad = (self.x_range - self.y_range) / 2
            self.min_y -= pad
            self.max_y += pad
        else:
            pad = (self.y_range - self.x_range) / 2
            self.min_x -= pad
            self.max_x += pad

        self.range_x = self.max_x - self.min_x
        self.range_y = self.max_y - self.min_y

        label_grid = self.create_label_grid(inv_model, classifier)
        label_grid, self.label_map = self.remap_labels(label_grid.astype(int))

        # Plot decision maps
        grid_labels = label_grid.reshape(self.grid_size, self.grid_size)
        if not self.flip_vertical:
            grid_labels = np.flip(grid_labels, axis=0)

        scaled = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))

        if self.flip_vertical:
            scaled[:, 1] = 1.0 - scaled[:, 1]

        fig = plt.figure(figsize=(6, 6))

        num_classes = len(self.label_map)

        norm = BoundaryNorm(
            boundaries=np.arange(num_classes + 1) - 0.5,
            ncolors=num_classes,
        )

        plt.imshow(grid_labels, cmap="tab10", norm=norm, extent=(0, 1, 0, 1), alpha=0.93)

        # remapped_labels = np.vectorize(self.label_map.get)(labels.astype(int))

        labels_int = labels.astype(int)
        missing = set(np.unique(labels_int)) - set(self.label_map.keys())

        if missing:
            print(f"[WARN] Missing labels {missing}, using default 0")

        remapped_labels = np.vectorize(lambda x: self.label_map.get(x, 0))(labels_int)

        if self.point_size > 0.0:
            plt.scatter(
                scaled[:, 0],
                scaled[:, 1],
                c=remapped_labels,
                cmap="tab10",
                norm=norm,
                s=self.point_size,
                edgecolor="black",
                lw=0.6,
            )

        plt.axis("off")
        plt.gca().set_aspect("equal")

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

import os
from typing import List, Tuple
import torchvision.transforms.functional as F

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
import torch


class CropZoomTransform(BasicTransform):
    def __init__(
        self,
        crop_distances: RandomScalar = (25, 50),
        zoom_distances: RandomScalar = (75, 150),
        zoom_prob: float = 0.8,
        synchronize_xy: bool = True,  # 同步xy轴的缩放
        xy_ratio: float = 2,  # xy轴的缩放比例
    ):
        self.crop_distances = crop_distances
        self.zoom_distances = zoom_distances
        self.zoom_prob = zoom_prob
        self.synchronize_xy = synchronize_xy
        self.xy_ratio = xy_ratio
        super().__init__()

    def get_parameters(self, **data_dict) -> dict:
        shape = data_dict["image"].shape

        zoom = torch.rand(1) < self.zoom_prob
        distance_range = self.crop_distances if not zoom else self.zoom_distances
        distance_y = int(sample_scalar(distance_range))

        if self.synchronize_xy:
            distance_x = distance_y * self.xy_ratio
        else:
            distance_x = int(sample_scalar(distance_range))
        distances = [distance_y, distance_y, distance_x, distance_x]

        dct = {}
        dct["shape"] = shape[1:]
        dct["zoom"] = zoom
        dct["distances"] = distances

        return dct

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return self._crop_zoom(img, **params)

    def _apply_to_segmentation(self, seg: torch.Tensor, **params) -> torch.Tensor:
        return self._crop_zoom(seg, **params, interpolation=0)

    def _apply_to_regr_target(
        self, regression_target: torch.Tensor, **params
    ) -> torch.Tensor:
        return self._crop_zoom(regression_target, **params, interpolation=0)

    def _crop_zoom(
        self, tensor: torch.Tensor, shape, zoom, distances, interpolation=2
    ) -> torch.Tensor:
        print("original:", tensor.shape)
        if zoom:
            # 缩放
            new_tensor = torch.zeros_like(tensor)
            new_shape = (
                shape[0] - distances[0] - distances[1],
                shape[1] - distances[2] - distances[3],
            )
            # print(new_shape)
            print("zoom:", new_shape)
            tensor = F.resize(tensor, new_shape, interpolation=interpolation)  # type: ignore
            new_tensor[
                :, distances[0] : -distances[1], distances[2] : -distances[2]
            ] = tensor
        else:
            # 裁剪
            new_tensor = tensor[
                :, distances[0] : -distances[1], distances[2] : -distances[2]
            ]
            print("crop", new_tensor.shape)
            tensor = F.resize(new_tensor, shape, interpolation=interpolation)  # type: ignore
        return tensor

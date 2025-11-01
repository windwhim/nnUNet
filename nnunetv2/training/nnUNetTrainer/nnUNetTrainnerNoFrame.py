import torch
from .nnUNetTrainnerGlasses import nnUNetTrainerGlasses


class nnUNetTrainerNoFrame(nnUNetTrainerGlasses):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.num_iterations_per_epoch = 800
        self.num_epochs = 20

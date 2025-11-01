import torch
from .nnUNetTrainnerGlasses import nnUNetTrainerGlasses


class nnUNetTrainerFrame(nnUNetTrainerGlasses):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.num_iterations_per_epoch = 3500
        self.num_epochs = 30

from attr import define

@define
class TrainingConfig:
    lr: int
    num_epochs: int
    weight_decay: float
    fine_tuning: bool=False
    generation_loss: bool=False
    regression_loss: bool=False
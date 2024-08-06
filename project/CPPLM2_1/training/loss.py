from attr import define
import torch
import torch.nn as nn


@define
class LossValues():
    intensity: float
    sequence: float


    def total_loss(self, alpha=0, beta=1):
        return alpha * self.intensity + beta * self.sequence
    

    def __add__(self, other):
        if isinstance(other, LossValues):
            return LossValues(self.intensity + other.intensity, self.sequence + other.sequence)
        raise TypeError("Operands must be of type LossValues")
    
    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            return LossValues(self.intensity / scalar, self.sequence / scalar)
        raise TypeError("Divisor must be a number")
    
    def __str__(self):
        return f'Intensity Prediction Loss: {self.intensity:.3f}, Sequence Prediction Loss: {self.sequence:.3f}'
    

    @staticmethod
    def calc_loss(
        intensity_logits: torch.Tensor|None =None,
        intensity_target: torch.Tensor|None =None,
        intensity_mask: torch.Tensor|None =None,
        sequence_logits: torch.Tensor|None =None,
        sequence_target: torch.Tensor|None =None,
        sequence_mask: torch.Tensor|None =None
        ):
        intensity_loss = 0
        sequence_loss = 0
        if intensity_logits != None:
            intensity_loss = LossValues.calc_intensity_loss(intensity_logits, intensity_target, intensity_mask)
        if sequence_logits != None:
            sequence_loss = LossValues.calc_sequence_loss(sequence_logits, sequence_target, sequence_mask)
        return LossValues(intensity_loss, sequence_loss)
    
    @staticmethod
    def calc_sequence_loss(logits, ids, masked_positions):
        criterion = nn.CrossEntropyLoss()
        masked_indices = torch.nonzero(masked_positions).squeeze()
        masked_logits = logits[masked_indices]
        masked_labels = ids[masked_indices]
        loss = criterion(masked_logits, masked_labels)
        if torch.isnan(loss).any().item():
                print("WARNING (GENERATION): there are no prediction to calculate the loss!")
        return loss
    
    @staticmethod
    def calc_intensity_loss(logits, ids, masked_intensities):
        criterion = nn.CrossEntropyLoss()
        masked_indices = torch.nonzero(masked_intensities).squeeze()
        masked_logits = logits[masked_indices]
        masked_labels = ids[masked_indices]
        loss = criterion(masked_logits, masked_labels)
        if torch.isnan(loss).any().item():
            print("WARNING (REGRESSION): there is no prediction to calcualte the loss!")
        return loss


if __name__ == "__main__":
    pass
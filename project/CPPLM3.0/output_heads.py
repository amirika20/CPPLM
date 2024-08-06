import torch.nn as nn
import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from layers.regression_head import RegressionHead
from tokenization.constants import *
from attr import dataclass

@dataclass
class CPPLMOutput:
    sequence_logits: torch.Tensor
    predicted_intensity: torch.Tensor
    embeddings: torch.Tensor

class OutputHeads(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.sequence_head = RegressionHead(d_model, 30)
        self.intensity_head = RegressionHead(d_model, NUM_INTENSITY_BIN)


    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> CPPLMOutput:
        sequence_logits = self.sequence_head(x[:,1:,:])
        predicted_intensity = self.intensity_head(torch.mean(x,1))

        return CPPLMOutput(
            sequence_logits=sequence_logits,
            predicted_intensity=predicted_intensity.squeeze(),
            embeddings=embed,
        )
    
if __name__ == "__main__":
    model = OutputHeads(128)
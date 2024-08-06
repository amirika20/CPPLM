# Directory configuration
import os
import sys
sys.path.append("/home/amirka/CPP/CPPLM")
sys.path.append("/home/amirka/CPP/CPPLM/project/CPPLM1.0")

from layers.regression_head import RegressionHead
from tokenization.constants import *

import torch.nn as nn
import torch
import os
import sys

from attr import dataclass

@dataclass
class CPPLMOutput:
    predicted_bin: torch.Tensor
    embeddings: torch.Tensor

class OutputHeads(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.classification_head = RegressionHead(d_model, NUM_INTENSITY_BIN)


    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> CPPLMOutput:
        predicted_intensity = self.classification_head(torch.mean(x,1))
        return CPPLMOutput(
            predicted_bin=predicted_intensity.squeeze(),
            embeddings=embed,
        )
    
if __name__ == "__main__":
    model = OutputHeads(128)
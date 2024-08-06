# Directory configuration
import os
import sys
sys.path.append("/home/amirka/CPP/CPPLM")
sys.path.append("/home/amirka/CPP/CPPLM/project/CPPLM1.0")

import torch.nn as nn
import torch
import os
import sys
import esm

from tokenization.constants import *

class EncodeInputs(nn.Module):
    """
    Module for encoding input features in the ESM-3 model.

    Args:
        d_model (int): The dimensionality of the model's hidden states.
    """

    def __init__(
            self,
            d_model: int,
            ):
        super().__init__()

        # Sequence
        self.sequence_embed = nn.Embedding(30, d_model, padding_idx=SEQUENCE_PAD_TOKEN)

    def forward(
        self,
        sequence_tokens: torch.Tensor,
    ) -> torch.Tensor:
        sequence_embed = self.sequence_embed(sequence_tokens)
        return sequence_embed

    

if __name__ == "__main__":
    sequence_tokens = torch.LongTensor([[ 0, 19,  8,  9, 15,  9, 27,  7,  8, 17, 19,  2,  1,  1,  1,  1,  1,  1,  1],
                                        [ 0, 19,  8,  9, 15,  9, 26, 19, 22,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1],])
    intensity_ids = torch.LongTensor([5,4])
    encoder = EncodeInputs(64)
    print(encoder(sequence_tokens, intensity_ids))
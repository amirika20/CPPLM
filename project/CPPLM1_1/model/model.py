# Directory configuration
import os
import sys
sys.path.append("/home/amirka/CPP/CPPLM")
sys.path.append("/home/amirka/CPP/CPPLM/project/CPPLM1.0")

from model.encoding import EncodeInputs
from model.output_heads import OutputHeads
import torch
import torch.nn as nn
from layers.transformer_stack import TransformerStack
from tokenization.constants import *


class CPPLM(nn.Module):
    def __init__(
            self,
            d_model: int|None,
            n_heads: int|None, 
            n_layers: int|None,
            ):
        super(CPPLM, self).__init__()
        self.encoder = EncodeInputs(d_model)
        self.transformer = TransformerStack(
            d_model=d_model,
            n_heads=n_heads,
            v_heads=0,
            n_layers=n_layers,
            n_layers_geom=0,
            mask_and_zero_frameless=False,
        )
        self.output_heads = OutputHeads(d_model)

        self.vocab_size = VOCAB_SIZE

        self.name = f'{d_model}d_{n_heads}h_{n_layers}l'
        

    @classmethod
    def from_pretrained(
        d_model: int|None,
        n_heads: int|None, 
        n_layers: int|None,
        model_name: str,
        device: torch.device | str ="cpu"
    ):
        
        model = CPPLM(d_model, n_heads, n_layers)
        model.load_state_dict(torch.load(f"/home/amirka/CPP/CPPLM/parameters/{model_name}.pt"))
        return model.to(device)
        
    
    def forward(
            self,
            cpp_ids: torch.Tensor,
            padding_mask: torch.Tensor | None = None,
            ):
        embedding = self.encoder(cpp_ids)
        x, embedding = self.transformer(embedding, sequence_id=padding_mask)
        return self.output_heads(x, embedding)


    def generate(self, src, src_mask=None):
        # TODO
        self.eval()
        with torch.no_grad():
            output = self(src, src_mask)
        return output
    


if __name__=="__main__":
    model = CPPLM(512,4,8).to("cuda")
    

    
import attr
import torch
import torch.nn.functional as F

from esm.models.esm3 import ESM3
from esm.sdk.api import (
    ESMProtein,
    ESMProteinTensor,
    SamplingConfig,
    SamplingTrackConfig,
)
from esm.tokenization import get_model_tokenizers
from esm.utils.constants.models import ESM3_OPEN_SMALL


def add_padding(protein_tensor: ESMProteinTensor, max_length: int) -> ESMProteinTensor:
    tokenizers = get_model_tokenizers(ESM3_OPEN_SMALL)

    current_length = len(protein_tensor)

    if current_length >= max_length:
        raise ValueError(
            f"Protein length is {current_length} which is greater than the maximum length of {max_length}"
        )

    left_pad = 0
    right_pad = max_length - current_length

    empty_protein_tensor = ESMProteinTensor.empty(
        current_length - 2,  # Account for BOS/EOS that our input already has
        tokenizers=tokenizers,
        device=protein_tensor.device,
    )

    for track in attr.fields(ESMProteinTensor):
        track_tensor = getattr(protein_tensor, track.name)

        if track_tensor is None:
            if track.name == "coordinates":
                continue
            else:
                # Initialize from empty tensor
                track_tensor = getattr(empty_protein_tensor, track.name)

        if track.name == "coordinates":
            pad_token = torch.inf
            new_tensor = F.pad(
                track_tensor,
                (0, 0, 0, 0, left_pad, right_pad),
                value=pad_token,
            )
        elif track.name in ["function", "residue_annotations"]:
            pad_token = getattr(tokenizers, track.name).pad_token_id
            new_tensor = F.pad(
                track_tensor,
                (0, 0, left_pad, right_pad),
                value=pad_token,
            )
        else:
            pad_token = getattr(tokenizers, track.name).pad_token_id
            new_tensor = F.pad(
                track_tensor,
                (
                    left_pad,
                    right_pad,
                ),
                value=pad_token,
            )
        protein_tensor = attr.evolve(protein_tensor, **{track.name: new_tensor})

    return protein_tensor


client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device="cuda")
protein = ESMProtein(
    sequence=(
        "AGYLLGKINLKALAALAKKILKRVK2KRVKAGYLLGKINLKALAALAKKIL3RGGRLSYSRRRFSTSTGR"
    )
)
protein_tensor = client.encode(protein)
protein_tensor_padded = add_padding(protein_tensor, 1024)
output = client.forward_and_sample(
    protein_tensor_padded,
    SamplingConfig(sequence=SamplingTrackConfig(), return_per_residue_embeddings=True),
)
print(protein_tensor.sequence.shape)
print(protein_tensor_padded.sequence.shape)
print(output.per_residue_embedding.shape)
print(protein_tensor.sequence)
seq = "AGYLLGKINLKALAALAKKILKRVK2KRVKAGYLLGKINLKALAALAKKIL3RGGRLSYSRRRFSTSTGR"
print(len(seq))
print(output)
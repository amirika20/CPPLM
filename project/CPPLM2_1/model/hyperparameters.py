from attr import define


@define
class ModelConfiguration():
    d_model: int
    n_heads: int
    n_layers: int
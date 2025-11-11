import logging

import torch

from ..transformer.model import Transformer
from ...register import tables


@tables.register("model_classes", "Conformer")
class Conformer(Transformer):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

import logging

from ..transformer.model import Transformer
from ...register import tables


@tables.register("model_classes", "Branchformer")
class Branchformer(Transformer):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

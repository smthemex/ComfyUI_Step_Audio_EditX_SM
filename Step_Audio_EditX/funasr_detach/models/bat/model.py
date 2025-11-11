#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import time
import torch
import logging
from contextlib import contextmanager
from typing import Dict, Optional, Tuple
from distutils.version import LooseVersion

from ...register import tables
from ...utils import postprocess_utils
from ...utils.datadir_writer import DatadirWriter
from ..transducer.model import Transducer
from ...train_utils.device_funcs import force_gatherable
from ..transformer.scorers.ctc import CTCPrefixScorer
from ...losses.label_smoothing_loss import LabelSmoothingLoss
from ..transformer.scorers.length_bonus import LengthBonus
from ..transformer.utils.nets_utils import get_transducer_task_io
from ...utils.load_utils import load_audio_text_image_video, extract_fbank
from ..transducer.beam_search_transducer import BeamSearchTransducer


if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


@tables.register("model_classes", "BAT")  # TODO: BAT training
class BAT(Transducer):
    pass

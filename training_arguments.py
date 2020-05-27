import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import json
import torch


@dataclass
class TrainingArguments:
    output_dir = "model"
    overwrite_output_dir = True
    do_train = True
    do_eval = False
    evaluate_during_training = True
    gradient_accumulation_steps = 1
    learning_rate = 5e-5
    weight_decay = 0.0
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    num_train_epochs = 3.0
    max_steps = -1
    warmup_steps = 0
    logging_dir = None
    logging_first_step = None
    logging_steps = 500
    save_steps = 5000
    save_total_limit = None
    no_cuda = False
    seed = 42
    fp16 = False
    fp16_opt_level = "O1"
    local_rank = -1
    tpu_num_cores = None
    tpu_metrics_debug = False
    is_tpu_available = False
    device = "cuda"
    n_gpu = 1

    def __init__(self, per_gpu_train_batch_size=2, per_gpu_eval_batch_size=4):
        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size

    @property
    def train_batch_size(self) -> int:
        return self.per_gpu_train_batch_size * max(1, self.n_gpu)

    @property
    def eval_batch_size(self) -> int:
        return self.per_gpu_eval_batch_size * max(1, self.n_gpu)

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(dataclasses.asdict(self), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = dataclasses.asdict(self)
        valid_types = [bool, int, float, str, torch.Tensor]
        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}
import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..utils import create_custom_optimzer, create_custom_scheduler
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

import wandb
import torch.distributed as dist

if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(self, finetuning_args: "FinetuningArguments", **kwargs) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None  # backup labels
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate(
                    (preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1
                )  # move pad token to last

        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for label, pred in zip(decoded_labels, decoded_preds):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))


class ValueTrainer_herarchical(Seq2SeqTrainer):
    r"""
    Custom trainer for handling value-based training tasks.
    """

    def __init__(self, finetuning_args: "FinetuningArguments", weight_alpha: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.weight_alpha = weight_alpha  # Weight for the value loss
        self.metric_hook = {
            "total_loss": [],
            "value_loss": [],
            "sft_loss": [],
            "value_loss_search": [],
            "value_loss_thought": [],
        }

        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

    def log(self, log_dict):  # overwrite Trainer.log [hacked]
        if len(self.metric_hook["total_loss"]) != 0:
            log_dict["total_loss"] = sum(self.metric_hook["total_loss"]) / len(self.metric_hook["total_loss"])
            log_dict["value_loss"] = sum(self.metric_hook["value_loss"]) / len(self.metric_hook["value_loss"])
            log_dict["sft_loss"] = sum(self.metric_hook["sft_loss"]) / len(self.metric_hook["sft_loss"])
            log_dict["value_loss_search"] = sum(self.metric_hook["value_loss_search"]) / len(self.metric_hook["value_loss_search"])
            log_dict["value_loss_thought"] = sum(self.metric_hook["value_loss_thought"]) / len(self.metric_hook["value_loss_thought"])

            self.metric_hook["total_loss"] = []
            self.metric_hook["value_loss"] = []
            self.metric_hook["sft_loss"] = []
            self.metric_hook["value_loss_search"] = []
            self.metric_hook["value_loss_thought"] = []


        # when logging
        if "total_loss" in log_dict:
            log_dict["total_loss"] = log_dict["total_loss"] / self.world_size
            log_dict["value_loss"] = log_dict["value_loss"] / self.world_size
            log_dict["sft_loss"] = log_dict["sft_loss"] / self.world_size

        super().log(log_dict)

    def compute_loss(
        self, model: "PreTrainedModel", inputs: Dict[str, torch.Tensor], return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        # Compute rewards
        Q_thought = inputs.get("Q_thought", None)
        Q_search = inputs.get("Q_search", None)
        if Q_thought is not None:
            del inputs["Q_thought"]
            del inputs["Q_search"]

        labels = inputs.get("labels", None)
        if labels is not None:
            del inputs["labels"]

        mask_thought = Q_thought.ne(IGNORE_INDEX)
        mask_search = Q_search.ne(IGNORE_INDEX)

        lm_logits, loss, values_thought, values_search = model(**inputs, output_hidden_states=True, return_dict=True)
        values_thought = torch.tanh(values_thought)
        values_search = torch.tanh(values_search)
        # values = torch.sigmoid(values)  # 改为使用 sigmoid

        if loss is None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            if torch.all(shift_labels == IGNORE_INDEX):
                loss_fct = CrossEntropyLoss(reduction='sum')
            else:
                loss_fct = CrossEntropyLoss()

            if isinstance(model, torch.nn.DataParallel):
                vocab_size = model.module.pretrained_model.config.vocab_size
            else:
                vocab_size = model.pretrained_model.config.vocab_size

            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        assert not torch.isnan(loss) and Q_thought is not None

        Q_thought = Q_thought.type_as(values_thought)
        masked_values = torch.where(mask_thought, values_thought, Q_thought)
        value_loss_thought= F.mse_loss(masked_values, Q_thought, reduction='sum') / (mask_thought.sum() + 1e-3)

        Q_search = Q_search.type_as(values_search)
        masked_values = torch.where(mask_search, values_search, Q_search)
        value_loss_search = F.mse_loss(masked_values, Q_search, reduction='sum') / (mask_search.sum() + 1e-3)

        value_loss = value_loss_thought + value_loss_search
        all_losses = loss + self.weight_alpha * value_loss

        self.metric_hook["sft_loss"].append(loss.item())
        self.metric_hook["value_loss_thought"].append(value_loss_thought.item())
        self.metric_hook["value_loss_search"].append(value_loss_search.item())
        self.metric_hook["value_loss"].append(value_loss.item())
        self.metric_hook["total_loss"].append(all_losses.item())

        return all_losses


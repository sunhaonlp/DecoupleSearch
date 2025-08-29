
# from typing import TYPE_CHECKING, List, Optional
from typing import Any, TYPE_CHECKING, List, Optional, Union

from transformers import DataCollatorForSeq2Seq

from ...data import get_dataset, split_dataset
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import get_logits_processor, fix_valuehead_checkpoint
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..utils import create_modelcard_and_push
from .metric import ComputeMetrics
from .trainer import CustomSeq2SeqTrainer, ValueTrainer_herarchical
from dataclasses import dataclass
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase


from transformers.tokenization_utils_base import PaddingStrategy
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

from ...extras.callbacks import FixValueHeadModelCallback

@dataclass
class VMDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

            # Padding Q to the longest sequence in the batch
            if "Q" in features[0]:  # float32
                # max_length_Q = max(len(feature["Q"]) for feature in features)
                for feature in features:
                    remainder = [IGNORE_INDEX] * (
                                max_label_length - len(feature["Q"]))  # Assuming IGNORE_INDEX as padding value for Q

                    if isinstance(feature["Q"], list):
                        feature["Q"] = (
                            feature["Q"] + remainder if padding_side == "right" else remainder + feature["Q"]
                        )
                    elif padding_side == "right":
                        feature["Q"] = np.concatenate([feature["Q"], remainder]).astype(np.float32)
                    else:
                        feature["Q"] = np.concatenate([remainder, feature["Q"]]).astype(np.float32)

        features = pad_without_fast_tokenizer_warning(  # only padding input_ids and attention_mask
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features



@dataclass
class VMDataCollatorForSeq2Seq_hierarchical(DataCollatorForSeq2Seq):
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

            # Padding Q to the longest sequence in the batch
            if "Q_thought" in features[0]:  # float32
                # max_length_Q = max(len(feature["Q"]) for feature in features)
                for feature in features:
                    remainder = [IGNORE_INDEX] * (
                                max_label_length - len(feature["Q_thought"]))  # Assuming IGNORE_INDEX as padding value for Q

                    if isinstance(feature["Q_thought"], list):
                        feature["Q_thought"] = (
                            feature["Q_thought"] + remainder if padding_side == "right" else remainder + feature["Q_thought"]
                        )
                    elif padding_side == "right":
                        feature["Q_thought"] = np.concatenate([feature["Q_thought"], remainder]).astype(np.float32)
                    else:
                        feature["Q_thought"] = np.concatenate([remainder, feature["Q_thought"]]).astype(np.float32)

            # Padding Q to the longest sequence in the batch
            if "Q_search" in features[0]:  # float32
                # max_length_Q = max(len(feature["Q"]) for feature in features)
                for feature in features:
                    remainder = [IGNORE_INDEX] * (
                                max_label_length - len(feature["Q_search"]))  # Assuming IGNORE_INDEX as padding value for Q

                    if isinstance(feature["Q_search"], list):
                        feature["Q_search"] = (
                            feature["Q_search"] + remainder if padding_side == "right" else remainder + feature["Q_search"]
                        )
                    elif padding_side == "right":
                        feature["Q_search"] = np.concatenate([feature["Q_search"], remainder]).astype(np.float32)
                    else:
                        feature["Q_search"] = np.concatenate([remainder, feature["Q_search"]]).astype(np.float32)

        features = pad_without_fast_tokenizer_warning(  # only padding input_ids and attention_mask
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features



def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer = load_tokenizer(model_args)
    tokenizer.add_tokens(['<thought>', '<search>'])

    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="sft")
    model = load_model(data_args, tokenizer, model_args, finetuning_args, training_args.do_train)
    model.resize_token_embeddings(len(tokenizer))

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = VMDataCollatorForSeq2Seq_hierarchical(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )
    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams

    trainer = ValueTrainer_herarchical(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        weight_alpha=data_args.weight_alpha,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks + [FixValueHeadModelCallback()],  # add `+ [FixValueHeadModelCallback()]`
        **split_dataset(dataset, data_args, training_args),
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()
    tokenizer.padding_side = "left"
    # Prevent discarding Q in the batch.
    training_args.remove_unused_columns = False

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if training_args.should_save:
            fix_valuehead_checkpoint(model, training_args.output_dir, training_args.save_safetensors)
        trainer.save_state()
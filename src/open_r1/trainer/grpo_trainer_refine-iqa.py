# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import sys
import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from processing_qwen2_5_vl import Qwen2_5_VLProcessor
import copy
from qwen_vl_utils import process_vision_info

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


SYSTEM_PROMPT = "You are a visual quality analysis expert."


# QUESTION_TEMPLATE = """Answer the question: "[QUESTION]" according to the content of the video. Select the answer from :[OPTION].

# Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

# Then, provide your answer within the <answer> </answer> tags, output the corresponding letter of the option.
# """
# QUESTION_TEMPLATE = """Answer the question: "[QUESTION]" according to the content of the video. Select the answer from :[OPTION]. Firstly, please directly provide the probabilities for each option after the <Answer without thinking> tag (formatted as A: a\%, B: b\%, C: c\%, D: d\% where a + b +c +d = 100. Please note that there is only one correct option. Consequently, the probability assigned to this option must be the highest among all provided probabilities.
#  Ensure that there is only one option with the highest probability.) without performing any preliminary analysis. Next, output your thought and analysis process within the <think> </think> tags and provide the probability of each option being selected within the <answer> </answer> tags (formatted as A: a\%, B: b\%, C: c \%, D: d\% where a + b +c +d = 100. Ensure that there is only one option with the highest probability.). The final output should be with the following format:

# <Answer without thinking>A: a%, B: b%,C: c %, D: d%

# <Answer with thinking><think></think><answer>A: a%, B: b%,C: c %, D: d%</answer> (If the number of options in the question is fewer than the four (A, B, C, D), then the probabilities for the extra options in the answer must be assigned a value of 0.)"""

# QUESTION_TEMPLATE = """Answer the question: "[QUESTION]" according to the content of the video (A value of 0–1 indicates very poor quality, 1–2 indicates poor quality, 2–3 indicates fair quality, 3–4 indicates good quality, and 4–5 indicates excellent quality.). Firstly, please directly rate the quality after the <Answer without thinking> tag (formatted as Score: a). Next, give a through thinking on this  and then output your thought and analysis process within the <think> </think> tags and provide your score within the <answer> </answer> tags (formatted as Score: b). The scores predicted after careful thinking  must differ from those direct output result without thinking, thereby demonstrating the tangible impact of the thinking and reasoning process.Please note that there is no audio and the audio quality of the videos should not be considered. The final output should be with the following format:

# <Answer without thinking>Score: a
# QUESTION_TEMPLATE = """Answer the question: "[QUESTION]" according to the content of the video (A value of 0–1 indicates very poor quality, 1–2 indicates poor quality, 2–3 indicates fair quality, 3–4 indicates good quality, and 4–5 indicates excellent quality.).  Directly provide your score within the <answer> </answer> tags (formatted as Score: b). Please note that there is no audio and the audio quality of the videos should not be considered. The final output should be with the following format:

# # <answer>Score: b</answer>"""
# <Answer with thinking><think></think><answer>Score: b</answer>"""
# QUESTION_TEMPLATE = """Answer the question: "[QUESTION]" according to the content of this image. Firstly, give a through thinking on this  and then output your thought and analysis process within the <think> </think> tags and then provide your score within the <answer> </answer> tags (formatted as Score: b). The final output should be strictly with the following format:

# <think>Your think</think> <answer>Score:b</answer>"""、
QUESTION_TEMPLATE = """Answer the question: "[QUESTION]" according to the content of the image. Firstly, give a through thinking on this and then output your thought and analysis process within the <think> </think> tags and provide your score within the <answer> </answer> tags (formatted as Score: Your score).  The final output should be with the following format:

<think>Your think</think><answer>Score:Your score</answer>"""
QUESTION_TEMPLATE1 = """Answer the question: "[QUESTION]" according to the content of this video. Please note that there is no audio and the audio quality of the videos should not be considered. Directly give your score within the <answer> </answer> tags (formatted as Score: Your score).  The final output should be with the following format:

<answer>Score:Your score</answer>"""
# QUESTION_TEMPLATE = """Answer the question: "[QUESTION]" according to the content of the video (A value of 0–20 indicates very poor quality, 20–40 indicates poor quality, 40–60 indicates fair quality, 60–80 indicates good quality, and 80–100 indicates excellent quality.). Directly provide your score within the <answer> </answer> tags (formatted as Score: b). Please note that there is no audio and the audio quality of the videos should not be considered. The final output should be with the following format:

# <answer>Score: b</answer>"""
# QUESTION_TEMPLATE = """[QUESTION]"""

class Qwen2VLGRPOTrainer_Refine(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation=None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation

        if isinstance(model, str):
            model_id = model
            # model_init_kwargs["use_cache"]=True
           
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5" in model_id:
                # breakpoint()
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model, 
                    torch_dtype=torch.bfloat16,
                    use_sliding_window=True,
                    **model_init_kwargs
                    )
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        # if is_deepspeed_zero3_enabled():
        if True:
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2.5" in model_id:
                # model_init_kwargs["use_cache"]=True
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, 
                    torch_dtype=torch.bfloat16,
                    use_sliding_window=True,
                    **model_init_kwargs
                    )
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5" in model_id or "Aria" in model_id:
                processing_class = Qwen2_5_VLProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        # print(self.max_completion_length)
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1, # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta
        # print(self.beta)

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]
    def _save_checkpoint(self, model, trial):
       
            super(Qwen2VLGRPOTrainer_Video_QA, self)._save_checkpoint(model, trial)

    # def _save(self, output_dir: Optional[str] = None, state_dict=None):
    #         # If we are executing this function, we are the process zero, so we don't check for that.
    #         output_dir = output_dir if output_dir is not None else self.args.output_dir
    #         os.makedirs(output_dir, exist_ok=True)
    #         logger.info(f"Saving model checkpoint to {output_dir}")

    #         supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
    #         # Save a trained model and configuration using `save_pretrained()`.
    #         # They can then be reloaded using `from_pretrained()`
    #         if not isinstance(self.model, supported_classes):
    #             if state_dict is None:
    #                 state_dict = self.model.state_dict()

    #             if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
    #                 self.accelerator.unwrap_model(self.model).save_pretrained(
    #                     output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
    #                 )
    #             else:
    #                 logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
    #                 if self.args.save_safetensors:
    #                     safetensors.torch.save_file(
    #                         state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
    #                     )
    #                 else:
    #                     torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
    #         else:
    #             self.model.save_pretrained(
    #                 output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
    #             )

    #         if self.tokenizer is not None:
    #             self.tokenizer.save_pretrained(output_dir)

    #         if self.processor is not None:
    #             self.processor.save_pretrained(output_dir)

    #         # Good practice: save your training arguments together with the trained model
    #         torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    def get_incentivizing_reward(self,model, input_ids, attention_mask, pixel_values_videos, video_grid_thw):
        def wa5(logprobs):
          
            # logprobs = np.array([logits["high"], logits["good"], logits["fair"], logits["poor"], logits["low"]])
            probs = torch.exp(logprobs)
            return probs
        def wa10(logprobs):
          
            # logprobs = np.array([logits["high"], logits["good"], logits["fair"], logits["poor"], logits["low"]])
            probs = torch.exp(logprobs) / torch.sum(torch.exp(logprobs))
            return torch.inner(probs, torch.tensor([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]).to(probs.device).to(probs.dtype))
        def find_last_element(lst,target):
            pos=[]
            count=0
            for num,i in enumerate(range(len(lst)-1, -1, -1)):
                if lst[i] in target and count<2:
                    count+=1
                    pos.append(len(lst)-1-num)
                if count==2:
                    return pos
            return -1  # 如果没有找到目标元素
        logits = model(input_ids, attention_mask=attention_mask, pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw).logits # (B, L, V)
        
            # logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values_videos, image_grid_thw=video_grid_thw).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        # wa5_score=[]
        rewards=[]
        for logits_row, input_ids_row in zip(logits, input_ids):
            #  log_probs = logits_row.index_select(1, torch.tensor([15,16,17,18,19])).log_softmax(dim=-1)
            # token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            # per_token_logps.append(token_log_prob)
            # num_quality_pos=find_last_element(input_ids_row,[15,16,17,18,19,20,21,22,23,24])
            # wa5_score=wa5(log_probs[num_quality_pos[2]].index_select(0, torch.tensor([15,16,17,18,19])))
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
            num_quality_pos=find_last_element(input_ids_row,[15,16,17,18,19,20,21,22,23,24])
            rewards.append(torch.exp(token_log_prob[num_quality_pos[1]]).unsqueeze(0))
            # wa5_score=wa5(log_probs[num_quality_pos[2]].index_select(0, torch.tensor([15,16,17,18,19]).to(log_probs.device)))
            # print(wa5_score)
        return rewards

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values_videos, video_grid_thw):
       
        logits = model(input_ids, attention_mask=attention_mask, pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw).logits  # (B, L, V)
        
            # logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values_videos, image_grid_thw=video_grid_thw).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs
    def make_conversation_image(self, example):
        example_prompt = QUESTION_TEMPLATE.replace("[QUESTION]", example["problem"]["question"])
        # example_prompt = example_prompt.replace("[OPTION]", str(example["problem"]["options"]))
        # print(example_prompt)
        return [
                # {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                        {"type": "text", "text": example_prompt},
                        {"type": "image", 
                        "image": example["video_path"], 
                        "max_pixels": 1920 * 1080,
                        },
                    ]
                },
            ]
    def make_conversation_image1(self, example):
        example_prompt = QUESTION_TEMPLATE1.replace("[QUESTION]", example["problem"]["question"])
        # example_prompt = example_prompt.replace("[OPTION]", str(example["problem"]["options"]))
        # print(example_prompt)
        return [
                # {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                        {"type": "text", "text": example_prompt},
                        {"type": "image", 
                        "image": example["video_path"], 
                        "max_pixels": 1920 * 1080,
                        },
                    ]
                },
            ]
    def make_conversation_video(self, example):
        example_prompt = QUESTION_TEMPLATE.replace("[QUESTION]", example["problem"]["question"])
        # example_prompt = example_prompt.replace("[OPTION]", str(example["problem"]["options"]))
        # print(example_prompt)
        return [
                # {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                        {"type": "text", "text": example_prompt},
                        {"type": "video", 
                        "video": example["video_path"], 
                        "total_pixels": 3584 * 28 * 28, 
                        "min_pixels": 16 * 28 * 28,
                        },
                    ]
                },
            ]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # print(inputs[0]["video_path"],0)
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if ".mp4" in inputs[0]["video_path"]:
            prompts = [self.make_conversation_video(example) for example in inputs]
            prompts1 = [self.make_conversation_video(example) for example in inputs]
        else:
            prompts = [self.make_conversation_image(example) for example in inputs]
            prompts1 = [self.make_conversation_image1(example) for example in inputs]
        # print(prompts)
        prompts_text = [self.processing_class.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts]
        prompts_text1 = [self.processing_class.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts1]
        
        
        video_inputs = [x["video_inputs"][0][0] for x in inputs if "video_inputs" in x]
        if video_inputs==[]:
            video_inputs=None
        image_inputs = [x["image_inputs"][0][0] for x in inputs if "image_inputs" in x]
        if image_inputs==[]:
            image_inputs=None
        if image_inputs is None:
            video_motion=inputs[0]["video_inputs"][0][1]
            fps_inputs = [x["video_kwargs"]["fps"] for x in inputs][0]
        else:
            video_motion=inputs[0]["image_inputs"][0][1]
            fps_inputs = [1.0]
        
        # image_inputs = [x["image_inputs"] for x in inputs if "video_inputs" in x]
        # fps_inputs = [x["video_kwargs"]["fps"] for x in inputs]

        # only support bs==1
        # video_inputs = video_inputs
        # fps_inputs = fps_inputs[0]
        # image_inputs = None

        prompt_inputs = self.processing_class(
            text=[prompts_text[0]], 
            images=image_inputs, 
            videos=video_inputs, 
            video_motion=video_motion,
            fps=[fps_inputs[0]], 
            padding=True, 
            return_tensors="pt", 
            padding_side="right", 
            add_special_tokens=False,
        )
        prompt_inputs_no_think_ref = self.processing_class(
            text=[prompts_text1 [0]+f"<answer>Score: {str(inputs[0]['solution']['answer'])[:3]}</answer>"], 
            images=image_inputs, 
            videos=video_inputs, 
            video_motion=video_motion,
            fps=[fps_inputs[0]], 
            padding=True, 
            return_tensors="pt", 
            padding_side="right", 
            add_special_tokens=False,
        )
        ref_probability=1-abs((int(str(inputs[0]['solution']['answer'])[2])-5)*0.1)
        # print(ref_probability)
        prompt_inputs_no_think = self.processing_class(
            text=[prompts_text1[0]], 
            images=image_inputs, 
            videos=video_inputs, 
            video_motion=video_motion,
            fps=[fps_inputs[0]], 
            padding=True, 
            return_tensors="pt", 
            padding_side="right", 
            add_special_tokens=False,
        )
        # image_inputs, video_inputs, video_kwargs = process_vision_info(prompts[0], return_video_kwargs=True)
        # fps_inputs = video_kwargs['fps']

        # prompt_inputs = self.processing_class(
        #     text=[prompts_text[0]], 
        #     images=image_inputs, 
        #     videos=[video_inputs[0]], 
        #     fps=[fps_inputs[0]], 
        #     padding=True, 
        #     return_tensors="pt", 
        #     padding_side="left", 
        #     add_special_tokens=False,
        # )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_inputs_no_think = super()._prepare_inputs(prompt_inputs_no_think)
        prompt_inputs_no_think_ref = super()._prepare_inputs(prompt_inputs_no_think_ref)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        prompt_ids_no_think, prompt_mask_no_think = prompt_inputs_no_think["input_ids"], prompt_inputs_no_think["attention_mask"]
        # pixel_values_videos = prompt_inputs["pixel_values_videos"]
        # try:
        prompt_inputs["pixel_values_videos"]=[prompt_inputs["pixel_values_videos"],video_motion]
        prompt_inputs_no_think["pixel_values_videos"]=[prompt_inputs_no_think["pixel_values_videos"],video_motion]
        video_grid_thw = prompt_inputs["video_grid_thw"]
        # except:
        #     prompt_inputs["pixel_values"]=[prompt_inputs["pixel_values"],video_motion]
        #     image_grid_thw = prompt_inputs["image_grid_thw"]
      
        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # print("123")
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
           
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            prompt_completion_ids_no_think = unwrapped_model.generate(**prompt_inputs_no_think, generation_config=self.generation_config)
           
            prompt_length_no_think = prompt_ids_no_think.size(1)
            prompt_ids_no_think = prompt_completion_ids_no_think[:, :prompt_length_no_think]
            completion_ids_no_think = prompt_completion_ids_no_think[:, prompt_length_no_think:]
            prompt_mask_no_think = prompt_mask_no_think.repeat_interleave(self.num_generations, dim=0)
        prompt_completions = self.processing_class.batch_decode(prompt_completion_ids, skip_special_tokens=True)
        # print(prompt_completions)
        def find_last_element(lst,target):
                pos=[]
                count=0
                for num,i in enumerate(range(len(lst)-1, -1, -1)):
                    if lst[i] in target and count<2:
                        count+=1
                        pos.append(len(lst)-1-num)
                    if count==2:
                        return pos
      
        prompt_completions_modified_ids=copy.deepcopy(prompt_completion_ids)
        for num in range(prompt_completions_modified_ids.shape[0]):
            num_quality_pos=find_last_element(prompt_completions_modified_ids[num],[15,16,17,18,19,20,21,22,23,24])
            num_quality_pos1=find_last_element(prompt_completion_ids[num],[15,16,17,18,19,20,21,22,23,24])
           
            num_quality_pos_no_think=find_last_element(prompt_inputs_no_think_ref["input_ids"][0],[15,16,17,18,19,20,21,22,23,24])
            for num1 in range(2):  
                prompt_completions_modified_ids[num][num_quality_pos[num1]]=prompt_inputs_no_think_ref["input_ids"][0][num_quality_pos_no_think[num1]]
                # print([prompt_completions_modified_ids[num][num_quality_pos[2]],prompt_completion_ids[num][num_quality_pos[2]]])
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()


        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        # try:
        pixel_values_videos = [prompt_inputs["pixel_values_videos"][0].repeat(self.num_generations, 1),prompt_inputs["pixel_values_videos"][1].repeat(self.num_generations, 1,1,1)]
        video_grid_thw = prompt_inputs["video_grid_thw"].repeat_interleave(self.num_generations, dim=0)
        think_rewards=self.get_incentivizing_reward(model, prompt_completions_modified_ids, attention_mask, pixel_values_videos, video_grid_thw)
        # with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
        #     # print("123")
        #     prompt_completion_ids_no_think = unwrapped_model.generate(**prompt_inputs_no_think, generation_config=self.generation_config)
           
        #     prompt_length_no_think = prompt_ids_no_think.size(1)
        #     prompt_ids_no_think = prompt_completion_ids_no_think[:, :prompt_length_no_think]
        #     completion_ids_no_think = prompt_completion_ids_no_think[:, prompt_length_no_think:]
        #     prompt_mask_no_think = prompt_mask_no_think.repeat_interleave(self.num_generations, dim=0)
        # prompt_completions = self.processing_class.batch_decode(prompt_completion_ids, skip_special_tokens=True)
        # print(prompt_completions)
      
        prompt_completions_modified_ids_no_think=copy.deepcopy(prompt_completion_ids_no_think)
        for num in range(prompt_completions_modified_ids_no_think.shape[0]):
            num_quality_pos=find_last_element(prompt_completions_modified_ids_no_think[num],[15,16,17,18,19,20,21,22,23,24])
            num_quality_pos_no_think=find_last_element(prompt_inputs_no_think_ref["input_ids"][0],[15,16,17,18,19,20,21,22,23,24])
            # print(prompt_completions_modified_ids_no_think[num].shape[0])
            # print(prompt_completions_modified_ids_no_think[num][-20])
            # print(num_quality_pos)
            for num1 in range(2):  
                prompt_completions_modified_ids_no_think[num][num_quality_pos[num1]]=prompt_inputs_no_think_ref["input_ids"][0][num_quality_pos_no_think[num1]]
        is_eos = completion_ids_no_think == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask_no_think = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        completions_no_think = self.processing_class.batch_decode(completion_ids_no_think, skip_special_tokens=True)
        print(completions_no_think)


        attention_mask_no_think = torch.cat([prompt_mask_no_think, completion_mask_no_think], dim=1)  # (B*G, P+C)
        # try:
        pixel_values_videos = [prompt_inputs["pixel_values_videos"][0].repeat(self.num_generations, 1),prompt_inputs["pixel_values_videos"][1].repeat(self.num_generations, 1,1,1)]
        video_grid_thw = prompt_inputs["video_grid_thw"].repeat_interleave(self.num_generations, dim=0)
        think_rewards=self.get_incentivizing_reward(model, prompt_completions_modified_ids, attention_mask, pixel_values_videos, video_grid_thw)
        no_think_rewards=self.get_incentivizing_reward(model, prompt_completions_modified_ids_no_think, attention_mask_no_think, pixel_values_videos, video_grid_thw)
        # print("THINK",think_rewards)
        # print("NO THINK",no_think_rewards)
        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values_videos, video_grid_thw)
       
        per_token_logps = (per_token_logps*attention_mask[:,1:])[:, prompt_length - 1 :]
    
        with torch.inference_mode():
                if self.ref_model is not None:
                   
                    ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, pixel_values_videos, video_grid_thw)
                    
                  
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values_videos, video_grid_thw)
       
        ref_per_token_logps = (ref_per_token_logps*attention_mask[:,1:])[:, prompt_length - 1 :]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        # completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        print(completions)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

      
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
              
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # print(example["solution"])
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        
        rewards_per_func_no_think = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions_no_think)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions_no_think)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
              
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # print(example["solution"])
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions_no_think, **reward_kwargs)
                rewards_per_func_no_think[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        # rlpr_rewards1=torch.clamp(torch.abs(torch.cat(no_think_rewards)-ref_probability)-torch.abs(torch.cat(think_rewards)-ref_probability),0,0.1)
        rlpr_rewards1=torch.clamp(torch.cat(think_rewards)-torch.cat(no_think_rewards),0,0.1)
        # rewards_per_func[:,0]=rewards_per_func[:,0]*torch.tensor(rlpr_rewards1>0)
        # rewards_per_func[:,0]=rewards_per_func[:,0]*torch.tensor(rlpr_rewards1>0)
        rewards = rewards_per_func.sum(dim=1)
        
        # no_think_rewards=self.get_incentivizing_reward(model, prompt_inputs_no_think["input_ids"].repeat_interleave(self.num_generations, dim=0), prompt_inputs_no_think["attention_mask"].repeat_interleave(self.num_generations, dim=0), pixel_values_videos, video_grid_thw)
      
        # rlpr_rewards1=torch.clamp(torch.cat(think_rewards)-torch.cat(no_think_rewards),0,0.1)
        # rlpr_rewards1=torch.clamp(torch.abs(torch.cat(no_think_rewards)-ref_probability)-torch.abs(torch.cat(think_rewards)-ref_probability),0,0.1)
        # rewards_per_func[]=rewards_per_func[]
        # rlpr_rewards=torch.clamp(torch.cat(think_rewards)-torch.cat(no_think_rewards),0,1)
        # print(rlpr_rewards)
        # rewards=rewards+rlpr_rewards1*(rewards//2)*(1-(rewards.sum(dim=0)//12))
        # rewards=rewards*(rlpr_rewards1//0.05)
        # rewards=rewards
        rewards=rewards+rlpr_rewards1*(rewards//2)
      
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        # if not all([rewards[num]==0 for num in range(rewards.shape[0])]):
        # advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        # print(rewards)
        # if not all([rewards[num]<2 for num in range(rewards.shape[0])]):
        #     # for num in range(rewards.shape[0]):
        #     #     if rewards[num]>2:
        #     #         rewards[num]=2.2
        #     advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        #     for num in range(advantages.shape[0]):
        #         if advantages[num]<=0:
        #             advantages[num]=0
                
        # else:
        #     advantages = (rewards - mean_grouped_rewards-0.2*1e-4) / (std_grouped_rewards + 1e-4)
       
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # per_token_loss = -(per_token_loss)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # print(inputs[0]["video_path"],1)
        
        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        rewards_per_func_no_think = self.accelerator.gather_for_metrics(rewards_per_func_no_think).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
            self._metrics[f"rewards_no_think/{reward_func_name}"].append(rewards_per_func_no_think[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        # self._metrics["reward_no_think"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

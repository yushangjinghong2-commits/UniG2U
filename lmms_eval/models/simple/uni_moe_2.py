import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add Uni-MoE repository to Python path
# Expected: lmms-eval/Uni-MoE/Uni-MoE-2/ directory at project root
wd = Path(__file__).parent.parent.parent.parent.resolve()
uni_moe_path = os.path.join(str(wd), "Uni-MoE", "Uni-MoE-2")
if os.path.exists(uni_moe_path):
    sys.path.insert(0, uni_moe_path)
    eval_logger.info(f"Added Uni-MoE path to sys.path: {uni_moe_path}")
else:
    eval_logger.warning(
        f"Uni-MoE repository not found at {uni_moe_path}. "
        f"Please clone it: cd {wd} && git clone https://github.com/HITsz-TMG/Uni-MoE.git"
    )


@register_model("uni_moe_2")
class UniMoE2(lmms):
    """
    Uni-MoE-2.0-Omni: Omnimodal Large Model with MoE Architecture

    Supports:
    - All-modality understanding (text, image, audio, video)
    - Image generation and editing
    - Speech generation

    Model: HIT-TMG/Uni-MoE-2.0-Omni
    Paper: https://arxiv.org/abs/2511.12609
    GitHub: https://github.com/HITsz-TMG/Uni-MoE/tree/master/Uni-MoE-2

    Example usage for understanding:
    python -m lmms_eval \
        --model uni_moe_2 \
        --model_args pretrained=HIT-TMG/Uni-MoE-2.0-Omni,mode=understanding \
        --tasks mmbench \
        --batch_size 1 \
        --output_path ./logs/
    """

    def __init__(
        self,
        pretrained: str = "HIT-TMG/Uni-MoE-2.0-Omni",
        mode: str = "understanding",
        device: str = "cuda",
        device_map: str = "auto",
        dtype: str = "bfloat16",
        batch_size: int = 1,
        max_new_tokens: int = 4096,
        temperature: float = 1.0,
        do_sample: bool = True,
        use_cache: bool = True,
        use_audio_in_video: bool = False,
        think_mode: bool = False,
        trust_remote_code: bool = True,
        continual_mode: bool = True,
        response_persistent_folder: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Validate mode
        if mode not in ["understanding", "generation"]:
            raise ValueError(
                f"mode must be 'understanding' or 'generation', got '{mode}'"
            )

        self.mode = mode
        self.pretrained = pretrained
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.use_cache = use_cache
        self.use_audio_in_video = use_audio_in_video
        self.think_mode = think_mode
        self.trust_remote_code = trust_remote_code
        self.continual_mode = continual_mode

        # Determine dtype
        if dtype == "bfloat16" or dtype == "bf16":
            self._dtype = torch.bfloat16
        elif dtype == "float16" or dtype == "fp16":
            self._dtype = torch.float16
        elif dtype == "float32" or dtype == "fp32":
            self._dtype = torch.float32
        else:
            self._dtype = torch.bfloat16

        # Setup response cache for continual mode
        if response_persistent_folder is None:
            self.response_persistent_folder = "./logs/uni_moe_2_persistent_folder"
        else:
            self.response_persistent_folder = response_persistent_folder

        self.response_cache = {}
        self.cache_mode = "start"

        if self.continual_mode:
            os.makedirs(self.response_persistent_folder, exist_ok=True)
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder, "uni_moe_2_response.json"
            )

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
                eval_logger.info(
                    f"Loaded cache: {len(self.response_cache)} records"
                )

        # Setup accelerator
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            if self.continual_mode:
                eval_logger.warning(
                    "Continual mode is not supported for distributed inference. "
                    "Automatically disabling continual_mode."
                )
                self.continual_mode = False
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
            self._device = torch.device(f"cuda:{self._rank}")
        else:
            self.accelerator = accelerator
            self._rank = 0
            self._world_size = 1
            self._device = torch.device(device)

        # Load model
        eval_logger.info(f"Loading Uni-MoE-2.0 model from {pretrained}")
        self._load_model()

        self.batch_size_per_gpu = int(batch_size)
        assert (
            self.batch_size_per_gpu == 1
        ), "batch_size > 1 not supported for Uni-MoE-2.0"

        eval_logger.info("Uni-MoE-2.0 model initialized successfully")

    def _load_model(self):
        """Load Uni-MoE-2.0 model and processor"""
        try:
            # Import Uni-MoE dependencies
            from uni_moe.model.modeling_out import (
                GrinQwen2VLOutForConditionalGeneration,
            )
            from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor
            from uni_moe.qwen_vl_utils import process_mm_info

            self.process_mm_info = process_mm_info

            # Load processor
            eval_logger.info("Loading processor...")
            self._processor = Qwen2VLProcessor.from_pretrained(self.pretrained)

            # Load model
            eval_logger.info("Loading model...")
            self._model = GrinQwen2VLOutForConditionalGeneration.from_pretrained(
                self.pretrained,
                torch_dtype=self._dtype,
                trust_remote_code=self.trust_remote_code,
            )
            self._model = self._model.to(self._device).eval()

            # Set processor data args from model config
            self._processor.data_args = self._model.config

            self._tokenizer = self._processor.tokenizer
            self._config = self._model.config

            eval_logger.info("Model loaded successfully")

        except ImportError as e:
            raise ImportError(
                f"Failed to import Uni-MoE dependencies. "
                f"Please ensure:\n"
                f"  1. Uni-MoE repository is cloned and installed\n"
                f"  2. Required dependencies are installed\n"
                f"Error: {e}"
            )

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def device(self):
        return self._device

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def processor(self):
        return self._processor

    @property
    def config(self):
        return self._config

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def flatten(self, input_list):
        """Flatten a nested list"""
        output = []
        for item in input_list:
            if isinstance(item, list):
                output.extend(self.flatten(item))
            else:
                output.append(item)
        return output

    def _prepare_messages(
        self, prompt: str, visuals: List, doc_id: str
    ) -> List[dict]:
        """
        Prepare messages in Uni-MoE format

        Args:
            prompt: Text prompt
            visuals: List of visual inputs (images/videos)
            doc_id: Document ID for logging

        Returns:
            List of message dicts in Uni-MoE format
        """
        content = []

        # Add visuals to content
        for visual in visuals:
            if visual is None:
                continue

            # Convert to PIL Image if needed
            if isinstance(visual, str):
                visual = Image.open(visual).convert("RGB")
            elif not isinstance(visual, Image.Image):
                eval_logger.warning(
                    f"Unsupported visual type: {type(visual)} for doc_id={doc_id}"
                )
                continue

            # Add image to content
            content.append({"type": "image", "image": visual})

        # Add text prompt
        # Replace placeholder tokens with Uni-MoE format
        prompt_text = prompt.replace("<image>", "").strip()
        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]

        return messages

    def understand(self, prompt: str, visuals: List, doc_id: str) -> str:
        """
        Understand multimodal input and generate text response

        Args:
            prompt: Input text prompt/question
            visuals: List of visual inputs
            doc_id: Document ID for logging

        Returns:
            Generated text answer
        """
        # Prepare messages
        messages = self._prepare_messages(prompt, visuals, doc_id)

        # Apply chat template
        texts = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process multimodal inputs
        image_inputs, video_inputs, audio_inputs = self.process_mm_info(messages)

        # Prepare inputs for model
        inputs = self.processor(
            text=[texts],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)

        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                use_cache=self.use_cache,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
            )

        # Decode output
        # Only decode the generated tokens (skip input tokens)
        generated_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
        output_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return output_text

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Main inference method"""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Uni-MoE-2.0 Generating",
        )

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            doc_uuid = get_uuid(task, split, doc_id)

            # Check cache
            if self.continual_mode and self.cache_mode == "resume":
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            prompt = contexts

            if self.mode == "understanding":
                # Image understanding mode
                if doc_to_visual is None:
                    eval_logger.warning(
                        f"No visual provided for understanding mode, doc_id={doc_id}"
                    )
                    res.append("")
                    pbar.update(1)
                    continue

                # Get visuals from doc_to_visual
                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                visuals = self.flatten(visuals)

                if not visuals or len(visuals) == 0:
                    eval_logger.warning(f"No visual data found for doc_id={doc_id}")
                    res.append("")
                    pbar.update(1)
                    continue

                # Generate response
                try:
                    output_text = self.understand(prompt, visuals, str(doc_id))
                    formatted_output = output_text
                except Exception as e:
                    eval_logger.error(f"Error processing doc_id={doc_id}: {e}")
                    import traceback

                    eval_logger.error(traceback.format_exc())
                    formatted_output = ""

            else:
                # Generation mode (to be implemented)
                eval_logger.warning(
                    f"Generation mode not yet implemented for doc_id={doc_id}"
                )
                formatted_output = ""

            res.append(formatted_output)

            # Update cache
            if self.continual_mode:
                self.response_cache[doc_uuid] = formatted_output
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f, ensure_ascii=False, indent=2)

            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported for generation models"""
        raise NotImplementedError(
            "Uni-MoE-2.0 is a generation model and does not support loglikelihood"
        )

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round dialogue generation"""
        raise NotImplementedError(
            "TODO: Implement multi-round dialogue generation for Uni-MoE-2.0"
        )

import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add Janus repository to Python path
wd = Path(__file__).parent.parent.parent.parent.resolve()
janus_path = os.path.join(str(wd), "Janus")
if os.path.exists(janus_path):
    sys.path.insert(0, janus_path)
    eval_logger.info(f"Added Janus path to sys.path: {janus_path}")


@register_model("janus_pro")
class JanusPro(lmms):
    """
    Janus-Pro: Unified multimodal understanding and generation model.
    https://huggingface.co/deepseek-ai/Janus-Pro-7B
    """

    def __init__(
        self,
        pretrained: str = "../models/Janus-Pro-7B",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache: bool = True,
        max_new_tokens: int = 4096,
        attn_implementation: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.pretrained = pretrained
        self.max_new_tokens = max_new_tokens
        self.use_cache = use_cache
        self.trust_remote_code = trust_remote_code

        # Setup accelerator
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        # Determine dtype
        if dtype == "bfloat16" or dtype == "bf16":
            self._dtype = torch.bfloat16
        elif dtype == "float16" or dtype == "fp16":
            self._dtype = torch.float16
        elif dtype == "float32" or dtype == "fp32":
            self._dtype = torch.float32
        else:
            self._dtype = torch.bfloat16

        # Load model
        eval_logger.info(f"Loading Janus-Pro model from {pretrained}")
        self._load_model(pretrained, attn_implementation)

        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "batch_size > 1 not supported for Janus-Pro"

        # Setup distributed training
        if accelerator.num_processes > 1:
            distributed_type_list = [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ]
            assert accelerator.distributed_type in distributed_type_list, (
                "Unsupported distributed type. Only DDP, FSDP, and DeepSpeed supported"
            )
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self._model)
            else:
                self._model = accelerator.prepare_model(
                    self._model, evaluation_mode=True
                )
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices with parallelism"
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self._rank = 0
            self._world_size = 1

        eval_logger.info("Janus-Pro model initialized successfully")

    def _load_model(self, pretrained: str, attn_implementation: Optional[str]):
        """Load Janus-Pro model and processor."""
        try:
            from transformers import AutoModelForCausalLM
            from janus.models import MultiModalityCausalLM, VLChatProcessor
            from janus.utils.io import load_pil_images
            
            eval_logger.info("Using Janus library for model loading")
            
            # Load processor
            self._processor = VLChatProcessor.from_pretrained(pretrained)
            self._tokenizer = self._processor.tokenizer
            
            # Load model
            eval_logger.info(f"Loading Janus-Pro model from {pretrained}")
            
            # Set environment variable to bypass torch.load security check for local models
            import os
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
            
            self._model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                trust_remote_code=True,
                torch_dtype=self._dtype,
            )
            self._model = self._model.to(self._device).eval()
            self._config = self._model.config
            
            eval_logger.info("Janus-Pro model loaded successfully")
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import Janus library. Please install it:\n"
                f"  cd lmms-eval && git clone https://github.com/deepseek-ai/Janus.git\n"
                f"Error: {e}"
            )
            

    @property
    def config(self):
        """Return the model config."""
        return self._config

    @property
    def tokenizer(self):
        """Return the tokenizer."""
        return self._tokenizer

    @property
    def processor(self):
        """Return the processor."""
        return self._processor

    @property
    def model(self):
        """Return the model, unwrapping it if using Accelerate."""
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        """Return the end of text token id."""
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        """Return the batch size."""
        return self.batch_size_per_gpu

    @property
    def device(self):
        """Return the device."""
        return self._device

    @property
    def rank(self):
        """Return the process rank."""
        return self._rank

    @property
    def world_size(self):
        """Return the world size."""
        return self._world_size

    def flatten(self, input_list):
        """Flatten a nested list."""
        new_list = []
        for i in input_list:
            for j in i:
                new_list.append(j)
        return new_list

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Compute log-likelihood (not implemented for Janus-Pro)."""
        raise NotImplementedError("Loglikelihood not implemented for Janus-Pro")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate text until stopping criteria are met."""
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )

        # Group requests by generation kwargs
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            (
                contexts,
                all_gen_kwargs,
                doc_to_visual,
                doc_id,
                task,
                split,
            ) = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [
                doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            visuals = self.flatten(visuals)

            # Get generation kwargs
            gen_kwargs = all_gen_kwargs[0]

            # Set default values
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            # Process each context
            assert len(contexts) == 1, "Batch size must be 1"
            context = contexts[0]

            # Prepare images
            images = []
            for visual in visuals:
                if isinstance(visual, str):
                    visual = Image.open(visual).convert("RGB")
                elif isinstance(visual, Image.Image):
                    visual = visual.convert("RGB")
                elif isinstance(visual, dict):
                    # Handle dict format - common in HuggingFace datasets
                    if "bytes" in visual:
                        from io import BytesIO
                        visual = Image.open(BytesIO(visual["bytes"])).convert("RGB")
                    elif "path" in visual:
                        visual = Image.open(visual["path"]).convert("RGB")
                    elif "image" in visual:
                        img = visual["image"]
                        if isinstance(img, str):
                            visual = Image.open(img).convert("RGB")
                        elif isinstance(img, Image.Image):
                            visual = img.convert("RGB")
                        else:
                            continue
                    else:
                        continue
                elif hasattr(visual, "convert"):
                    visual = visual.convert("RGB")
                else:
                    continue
                images.append(visual)

            # Set generation parameters
            max_new_tokens = gen_kwargs.get("max_new_tokens", self.max_new_tokens)
            temperature = gen_kwargs.get("temperature", 0.0)
            top_p = gen_kwargs.get("top_p", None)
            num_beams = gen_kwargs.get("num_beams", 1)

            # Generate response using Janus-Pro
            try:
                response = self._generate_response(
                    context=context,
                    images=images,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                )
            except Exception as e:
                eval_logger.error(f"Generation error: {e}")
                import traceback
                eval_logger.error(traceback.format_exc())
                response = ""

            # Clean up to free memory
            torch.cuda.empty_cache()

            res.append(response)
            self.cache_hook.add_partial(
                "generate_until", (context, gen_kwargs), response
            )
            pbar.update(1)

        # Reorder results to original order
        res = re_ords.get_original(res)
        pbar.close()
        return res

    def _generate_response(
        self,
        context: str,
        images: List[Image.Image],
        max_new_tokens: int,
        temperature: float,
        top_p: Optional[float],
        num_beams: int,
    ) -> str:
        """Generate response using Janus-Pro model with debug logs."""
        t0 = time.time()
        
        # 1. FIX: Generate dynamic placeholders for N images
        if images:
            # Create one placeholder per image
            image_placeholders = "<image_placeholder>\n" * len(images)
            user_content = image_placeholders + context
        else:
            user_content = context
            
        # Build conversation format
        conversation = [
            {
                "role": "User",
                "content": user_content,
                "images": images if images else [],
            },
            {"role": "Assistant", "content": ""},
        ]

        # 2. FIX: Use passed PIL images directly (don't call load_pil_images)
        pil_images = images
        
        # DEBUG LOG: Print when processor starts
        # print(f"[Debug] Processing {len(images)} images and prompt...")
        
        prepare_inputs = self._processor(
            conversations=conversation, 
            images=pil_images, 
            force_batchify=True
        ).to(self._device)

        # Run image encoder to get the image embeddings
        inputs_embeds = self._model.prepare_inputs_embeds(**prepare_inputs)

        # Prepare generation kwargs
        generate_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": prepare_inputs.attention_mask,
            "pad_token_id": self._tokenizer.eos_token_id,
            "bos_token_id": self._tokenizer.bos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "max_new_tokens": max_new_tokens,
            "use_cache": self.use_cache,
        }

        # Add sampling parameters
        if temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
            if top_p is not None:
                generate_kwargs["top_p"] = top_p
        else:
            generate_kwargs["do_sample"] = False

        if num_beams > 1:
            generate_kwargs["num_beams"] = num_beams

        # DEBUG LOG: Print before generation starts
        print(f"[Debug] Starting generation... (Prep time: {time.time()-t0:.2f}s)")
        
        # Generate response
        with torch.no_grad():
            outputs = self._model.language_model.generate(**generate_kwargs)

        # DEBUG LOG: Print after generation finishes
        # print(f"[Debug] Generation done (Total time: {time.time()-t0:.2f}s)")

        # Decode response
        answer = self._tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        del outputs, inputs_embeds, prepare_inputs
        return answer

    def generate_until_multi_round(self, requests) -> List[str]:
        """Generate for multi-round conversations (not implemented)."""
        raise NotImplementedError("Multi-round generation not yet implemented")
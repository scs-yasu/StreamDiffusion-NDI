import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

from utils.wrapper import StreamDiffusionWrapper

import torch

from PIL import Image

base_model = "stabilityai/sd-turbo"
taesd_model = "madebyollin/taesd"

default_prompt = "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"

class Pipeline:
    def __init__(self, device: torch.device, torch_dtype: torch.dtype):
        self.stream = StreamDiffusionWrapper(
            model_id_or_path=base_model,
            use_tiny_vae=True,
            device=device,
            dtype=torch_dtype,
            t_index_list=[35, 45],
            frame_buffer_size=1,
            width=512,
            height=512,
            use_lcm_lora=False,
            output_type="pil",
            warmup=10,
            vae_id=None,
            acceleration="tensorrt",
            mode="img2img",
            use_denoising_batch=True,
            cfg_type="none",
            use_safety_checker=True,
            # enable_similar_image_filter=True,
            # similar_image_filter_threshold=0.98,
            engine_dir="engines",
        )

        self.last_prompt = default_prompt
        self.stream.prepare(
            prompt=default_prompt,
            negative_prompt=default_negative_prompt,
            num_inference_steps=50,
            guidance_scale=1.2,
        )

    def predict(self, image: Image.Image, prompt: str) -> Image.Image:
        image_tensor = self.stream.preprocess_image(image)
        output_image = self.stream(image=image_tensor, prompt=prompt)

        return output_image

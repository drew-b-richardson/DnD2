"""Local image generation using diffusers + Stable Diffusion XL Turbo."""

import io


class ImageGenerator:
    MODEL_ID = "stabilityai/sdxl-turbo"

    _STYLE = (
        "pencil sketch, ink outline, black and white, fantasy line art, "
        "hatching, detailed linework, no color, monochrome"
    )
    _NEGATIVE = (
        "photorealistic, color, watercolor, oil painting, blurry, "
        "watermark, signature, 3d render, anime, low quality"
    )

    def __init__(self):
        self._pipe = None  # lazy-loaded on first generate() call

    def _load(self) -> None:
        import torch
        from diffusers import AutoPipelineForText2Image

        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # MPS + float16 produces black images — use float32 on Apple Silicon
        dtype = torch.float16 if device == "cuda" else torch.float32
        variant = "fp16" if device == "cuda" else None

        print(f"  [Image gen: loading {self.MODEL_ID} on {device}…]")
        pipe = AutoPipelineForText2Image.from_pretrained(
            self.MODEL_ID,
            torch_dtype=dtype,
            variant=variant,
        ).to(device)
        pipe.enable_attention_slicing()
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        print(f"  [Image gen: model ready on {device}]")

    def generate(self, prompt: str, output_path: str) -> str:
        """Generate a pencil-sketch illustration and save to output_path."""
        if self._pipe is None:
            self._load()

        full_prompt = f"{prompt}, {self._STYLE}"
        print(f"  [Image gen: prompt = {full_prompt[:120]}…]")

        result = self._pipe(
            prompt=full_prompt,
            negative_prompt=self._NEGATIVE,
            num_inference_steps=4,   # SDXL-Turbo optimal: 1-4 steps
            guidance_scale=0.0,      # Turbo requires guidance_scale=0
            width=512,
            height=512,
        )
        result.images[0].save(output_path)
        return output_path

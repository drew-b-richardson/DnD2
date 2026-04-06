"""Local image generation using diffusers + Stable Diffusion v1.5.

Uses SD v1.5 (~4 GB at float32) rather than SDXL-Turbo (~13 GB) so it can
coexist with a running Ollama model on Apple Silicon unified memory.
"""


class ImageGenerator:
    MODEL_ID = "runwayml/stable-diffusion-v1-5"

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
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Use float16 only on CUDA — MPS and CPU use float32 for stability
        dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"  [Image gen: loading {self.MODEL_ID} on {device}…]")
        pipe = StableDiffusionPipeline.from_pretrained(
            self.MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)

        # Faster scheduler — 20 steps instead of the default 50
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
        pipe.enable_attention_slicing()  # reduce peak RAM
        pipe.enable_vae_slicing()        # reduce peak RAM further
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
            num_inference_steps=20,
            guidance_scale=7.5,
            width=512,
            height=512,
        )
        result.images[0].save(output_path)
        return output_path

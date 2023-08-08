# Huggingface-compatible SDXL Unet rewrite

<a href="https://www.buymeacoffee.com/simoryu" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>


Why do this?

While huggingface is amazing, nowdays, its `UNet2DConditionModel`'s implementation has gotten extremely big. There are various configurations, branches during initialization, and many other things that make it hard to understand. (I am also partly to blame this, because LoRA with AttentionProcessor logic has also gotten rather huge part of the Unet implementation.) I would argue it got bigger than one can now reasonably understand and maintain, let alone extend. This will of course have pros and cons, but for many researchers, this is not ideal. 

Since SDXL will likely be used by many researchers, I think it is important to have concise implementations of the models, so that they can be easily understood and extended.

`sdxl_rewrite.py` tries to remove all the unnecessary parts of the original implementation, and tries to make it as concise as possible.

## Usage

```python
from sdxl_rewrite import UNet2DConditionModel
unet_new = UNet2DConditionModel().cuda().half()

# Load weights from the original model
from diffusers import DiffusionPipeline


pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
).to("cuda")
unet_new.load_state_dict(pipe.unet.state_dict())

```

In practice you would never do this. You would normaly copy this codebase and modify it to your needs, like putting adapters, loras, other modalities, etc etc.

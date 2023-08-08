# Huggingface Diffusers Compatible SDXL Unet Rewrite 

*Find this useful? I'm accumulating coffee until they become a A100 gpu.*

<a href="https://www.buymeacoffee.com/simoryu" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>


## Why do this?

While huggingface `diffusers` library is amazing, nowdays, its `UNet2DConditionModel`'s implementation has gotten extremely big. There are various configurations, branches during initialization, and many other "important-yet-not-related" stuff that make it hard to understand. (I am also partly to blame this, because LoRA with AttentionProcessor logic has also gotten rather huge part of the Unet implementation.) I would argue it got bigger than one researcher can now reasonably understand and maintain, let alone extend. This will of course have pros and cons, but for many researchers, this is not ideal. (Trust me, I use diffusers all the time and still get confused by the codebase.)

Since SDXL will likely be used by many researchers, I think it is very important to have concise implementations of the models, so that SDXL can be easily understood and extended.

`sdxl_rewrite.py` tries to remove all the unnecessary parts of the original implementation, and tries to make it as concise as possible.

## Usage

SDXL Rewrite tries to be directly compatible with the original `diffusers` library. You can use it like this:

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

# use the weights
pipe.unet = unet_new

```

Obviously in practice you would never do this. You would normaly copy this codebase and modify it to your needs, like putting adapters, loras, other modalities, etc etc.

Have a look at `example.ipynb` for bit more examples to use it directly with `diffusers` library.
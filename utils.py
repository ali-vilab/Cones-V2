from PIL import Image

import torch
import torch.nn.functional as F


def downsampling(img: torch.tensor, w: int, h: int) -> torch.tensor:
    return F.interpolate(
        img.unsqueeze(0).unsqueeze(1),
        size=(w, h),
        mode="bilinear",
        align_corners=True,
    ).squeeze()


def image_grid(images, rows=2, cols=2):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def latents_to_images(vae, latents, scale_factor=0.18215):
    """
    Decode latents to PIL images.
    """
    scaled_latents = 1.0 / scale_factor * latents.clone()
    images = vae.decode(scaled_latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()

    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

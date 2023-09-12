import argparse
import json
import os

from PIL import Image

from diffusers import StableDiffusionPipeline

from layout_guidance import layout_guidance_sampling
from utils import image_grid


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--inference_config",
        type=str,
        default=None,
        required=True,
        help='Path to a json file containing settings for inference, containing "residual_path", "prompt", '
             '"color_context", "edit_tokens", "layout", "subject_list".',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_results",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    return parser.parse_args()


def main(args):
    # Initialize pre-trained Stable Diffusion pipeline.
    pipeline = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path).to("cuda")

    # Load the settings required for inference from the configuration file.
    with open(args.inference_config, "r") as f:
        inference_cfg = json.load(f)

    prompt = inference_cfg[0]["prompt"]
    residual_dict = inference_cfg[0]["residual_dict"]
    subject_list = inference_cfg[0]["subject_list"]
    guidance_steps = inference_cfg[0]["guidance_steps"]
    guidance_weight = inference_cfg[0]["guidance_weight"]
    weight_negative = inference_cfg[0]["weight_negative"]
    layout = Image.open(inference_cfg[0]["layout"]).resize((768, 768)).convert("RGB")
    color_context = inference_cfg[0]["color_context"]
    subject_color_dict = {tuple(map(int, key.split(','))): value for key, value in color_context.items()}

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    subject_info = '_'.join([s[0] for s in sorted(subject_list)])
    prompt_info = '_'.join(prompt.split())
    save_dir = os.path.join(args.output_dir, subject_info, prompt_info)
    os.makedirs(save_dir, exist_ok=True)

    images = []

    for i in range(4):
        image = layout_guidance_sampling(
            seed=i,
            device="cuda:0",
            resolution=768,
            pipeline=pipeline,
            prompt=prompt,
            residual_dict=residual_dict,
            subject_list=subject_list,
            subject_color_dict=subject_color_dict,
            layout=layout,
            cfg_scale=7.5,
            inference_steps=50,
            guidance_steps=guidance_steps,
            guidance_weight=guidance_weight,
            weight_negative=weight_negative,
        )

        image.save(os.path.join(save_dir, f"{i}.png"))
        images.append(image)

    all_image = image_grid(images=images, rows=2, cols=2)
    all_image.save(os.path.join(save_dir, f"all_images.png"))


if __name__ == "__main__":
    args = get_args()
    main(args)

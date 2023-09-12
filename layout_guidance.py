import numpy as np
import math
import torch

from tqdm.auto import tqdm

from diffusers import LMSDiscreteScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput

from attention_control import register_attention_control
from utils import latents_to_images, downsampling


@torch.no_grad()
def layout_guidance_sampling(seed,
                             device,
                             resolution,
                             pipeline,
                             prompt="",
                             residual_dict=None,
                             subject_list=None,
                             subject_color_dict=None,
                             layout=None,
                             cfg_scale=7.5,
                             inference_steps=50,
                             guidance_steps=50,
                             guidance_weight=0.05,
                             weight_negative=-1e8):
    vae = pipeline.vae
    unet = pipeline.unet
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    unconditional_input_prompt = ""
    scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
    scheduler.set_timesteps(inference_steps, device=device)
    if guidance_steps > 0:
        guidance_steps = min(guidance_steps, inference_steps)
        scheduler_guidance = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        scheduler_guidance.set_timesteps(guidance_steps, device=device)

    # Process input prompt text
    text_input = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    # Edit text embedding conditions with residual token embeddings.
    cond_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    for name, token in subject_list:
        residual_token_embedding = torch.load(residual_dict[name])
        cond_embeddings[0][token] += residual_token_embedding.reshape(1024)

    # Process unconditional input "" for classifier-free guidance.
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [unconditional_input_prompt],
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    register_attention_control(unet)

    # Calculate the hidden features for each cross attention layer.
    hidden_states, uncond_hidden_states = _extract_cross_attention(tokenizer,
                                                                   device,
                                                                   layout,
                                                                   subject_color_dict,
                                                                   text_input,
                                                                   weight_negative)
    hidden_states["CONDITION_TENSOR"] = cond_embeddings
    uncond_hidden_states["CONDITION_TENSOR"] = uncond_embeddings
    hidden_states["function"] = lambda w, x, qk: (guidance_weight * w * math.log(1 + x ** 2)) * qk.std()
    uncond_hidden_states["function"] = lambda w, x, qk: 0.0

    # Sampling the initial latents.
    latent_size = (1, unet.in_channels, resolution // 8, resolution // 8)
    latents = torch.randn(latent_size, generator=torch.manual_seed(seed))
    latents = latents.to(device)
    latents = latents * scheduler.init_noise_sigma

    for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
        # Improve the harmony of generated images by self-recurrence.
        if i < guidance_steps:
            loop = 2
        else:
            loop = 1
        for k in range(loop):
            if i < guidance_steps:
                sigma = scheduler_guidance.sigmas[i]
                latent_model_input = scheduler.scale_model_input(latents, t)

                hidden_states.update({
                    "SIGMA": sigma,
                })

                noise_pred_text = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=hidden_states,
                ).sample

                uncond_hidden_states.update({
                    "SIGMA": sigma,
                })

                noise_pred_uncond = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=uncond_hidden_states,
                ).sample

                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
                latents = scheduler.step(noise_pred, t, latents, 1).prev_sample

                # Self-recurrence.
                if k < 1 and loop > 1:
                    noise_recurrent = torch.randn(latents.shape).to(device)
                    noise_scale = ((scheduler.sigmas[i] ** 2 - scheduler.sigmas[i + 1] ** 2) ** 0.5)
                    latents = latents + noise_scale * noise_recurrent
            else:
                latent_model_input = scheduler.scale_model_input(latents, t)
                noise_pred_text = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=cond_embeddings,
                ).sample

                latent_model_input = scheduler.scale_model_input(latents, t)

                noise_pred_uncond = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=uncond_embeddings,
                ).sample

                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
                latents = scheduler.step(noise_pred, t, latents, 1).prev_sample

    edited_images = latents_to_images(vae, latents)

    return StableDiffusionPipelineOutput(images=edited_images, nsfw_content_detected=None).images[0]


def _tokens_img_attention_weight(img_context_seperated, tokenized_texts, ratio: int = 8, original_shape=False):
    token_lis = tokenized_texts["input_ids"][0].tolist()
    w, h = img_context_seperated[0][1].shape

    w_r, h_r = round(w / ratio), round(h / ratio)
    ret_tensor = torch.zeros((w_r * h_r, len(token_lis)), dtype=torch.float32)
    for v_as_tokens, img_where_color in img_context_seperated:
        is_in = 0
        for idx, tok in enumerate(token_lis):
            if token_lis[idx: idx + len(v_as_tokens)] == v_as_tokens:
                is_in = 1

                ret_tensor[:, idx: idx + len(v_as_tokens)] += (
                    downsampling(img_where_color, w_r, h_r)
                    .reshape(-1, 1)
                    .repeat(1, len(v_as_tokens))
                )

        if not is_in == 1:
            print(f"Warning ratio {ratio} : tokens {v_as_tokens} not found in text")

    if original_shape:
        ret_tensor = ret_tensor.reshape((w_r, h_r, len(token_lis)))

    return ret_tensor


def _image_context_seperator(img, color_context: dict, _tokenizer, neg: float):
    ret_lists = []

    if img is not None:
        w, h = img.size
        matrix = np.zeros((h, w))
        for color, v in color_context.items():
            color = tuple(color)
            if len(color) > 3:
                color = color[:3]
            if isinstance(color, str):
                r, g, b = color[1:3], color[3:5], color[5:7]
                color = (int(r, 16), int(g, 16), int(b, 16))
            img_where_color = (np.array(img) == color).all(axis=-1)
            matrix[img_where_color] = 1

        for color, (subject, weight_active) in color_context.items():
            if len(color) > 3:
                color = color[:3]
            v_input = _tokenizer(
                subject,
                max_length=_tokenizer.model_max_length,
                truncation=True,
            )

            v_as_tokens = v_input["input_ids"][1:-1]
            if isinstance(color, str):
                r, g, b = color[1:3], color[3:5], color[5:7]
                color = (int(r, 16), int(g, 16), int(b, 16))
            img_where_color = (np.array(img) == color).all(axis=-1)
            matrix[img_where_color] = 1
            if not img_where_color.sum() > 0:
                print(f"Warning : not a single color {color} not found in image")

            img_where_color_init = torch.where(torch.tensor(img_where_color, dtype=torch.bool), weight_active, neg)

            img_where_color = torch.where(torch.from_numpy(matrix == 1) & (img_where_color_init == 0.0),
                                          torch.tensor(neg), img_where_color_init)

            # Add the image location corresponding to the token.
            ret_lists.append((v_as_tokens, img_where_color))
    else:
        w, h = 768, 768

    if len(ret_lists) == 0:
        ret_lists.append(([-1], torch.zeros((w, h), dtype=torch.float32)))
    return ret_lists, w, h


def _extract_cross_attention(tokenizer, device, color_map_image, color_context, text_input, neg):
    # Process color map image and context
    seperated_word_contexts, width, height = _image_context_seperator(
        color_map_image, color_context, tokenizer, neg
    )

    # Compute cross-attention weights
    cross_attention_weight_1 = _tokens_img_attention_weight(
        seperated_word_contexts, text_input, ratio=1, original_shape=True
    ).to(device)
    cross_attention_weight_8 = _tokens_img_attention_weight(
        seperated_word_contexts, text_input, ratio=8
    ).to(device)
    cross_attention_weight_16 = _tokens_img_attention_weight(
        seperated_word_contexts, text_input, ratio=16
    ).to(device)
    cross_attention_weight_32 = _tokens_img_attention_weight(
        seperated_word_contexts, text_input, ratio=32
    ).to(device)
    cross_attention_weight_64 = _tokens_img_attention_weight(
        seperated_word_contexts, text_input, ratio=64
    ).to(device)

    hidden_states = {
        "CA_WEIGHT_ORIG": cross_attention_weight_1,  # 768 x 768
        "CA_WEIGHT_9216": cross_attention_weight_8,  # 96 x 96
        "CA_WEIGHT_2304": cross_attention_weight_16,  # 48 x 48
        "CA_WEIGHT_576": cross_attention_weight_32,  # 24 x 24
        "CA_WEIGHT_144": cross_attention_weight_64,  # 12 x 12
    }

    uncond_hidden_states = {
        "CA_WEIGHT_ORIG": 0,
        "CA_WEIGHT_9216": 0,
        "CA_WEIGHT_2304": 0,
        "CA_WEIGHT_576": 0,
        "CA_WEIGHT_144": 0,
    }

    return hidden_states, uncond_hidden_states

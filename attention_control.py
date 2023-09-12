import math

import torch
import torch.nn.functional as F

from diffusers.models.cross_attention import CrossAttention


class Cones2AttnProcessor:
    def __init__(self):
        super().__init__()

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        query = attn.to_q(hidden_states)
        is_dict_format = True
        if encoder_hidden_states is not None:
            try:
                encoder_hidden = encoder_hidden_states["CONDITION_TENSOR"]
            except:
                encoder_hidden = encoder_hidden_states
                is_dict_format = False
            if attn.cross_attention_norm:
                encoder_hidden = attn.norm_cross(encoder_hidden)
        else:
            encoder_hidden = hidden_states

        key = attn.to_k(encoder_hidden)
        value = attn.to_v(encoder_hidden)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_size_of_img = attention_scores.size()[-2]

        if attention_scores.size()[2] == 77:
            if is_dict_format:
                f = encoder_hidden_states["function"]
                try:
                    w = encoder_hidden_states[f"CA_WEIGHT_{attention_size_of_img}"]
                except KeyError:
                    w = encoder_hidden_states[f"CA_WEIGHT_ORIG"]
                    if not isinstance(w, int):
                        img_h, img_w, nc = w.shape
                        ratio = math.sqrt(img_h * img_w / attention_size_of_img)
                        w = F.interpolate(w.permute(2, 0, 1).unsqueeze(0), scale_factor=1 / ratio, mode="bilinear",
                                          align_corners=True)
                        w = F.interpolate(w.reshape(1, nc, -1), size=(attention_size_of_img,), mode='nearest').permute(
                            2, 1, 0).squeeze()
                    else:
                        w = 0
                if type(w) is int and w == 0:
                    sigma = encoder_hidden_states["SIGMA"]
                    cross_attention_weight = f(w, sigma, attention_scores)
                else:
                    bias = torch.zeros_like(w)
                    bias[torch.where(w > 0)] = attention_scores.std() * 0
                    sigma = encoder_hidden_states["SIGMA"]
                    cross_attention_weight = f(w, sigma, attention_scores)
                    cross_attention_weight = cross_attention_weight + bias
            else:
                cross_attention_weight = 0.0
        else:
            cross_attention_weight = 0.0

        attention_scores = (attention_scores + cross_attention_weight) * attn.scale
        attention_probs = attention_scores.softmax(dim=-1)

        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def register_attention_control(unet):
    attn_procs = {}
    for name in unet.attn_processors.keys():
        attn_procs[name] = Cones2AttnProcessor()

    unet.set_attn_processor(attn_procs)

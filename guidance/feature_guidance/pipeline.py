from typing import Any, Dict, Optional
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from .utils import get_nn_feats, random_bipartite_soft_matching
import numpy
from collections import deque
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.distributed
import transformers
from collections import OrderedDict
from PIL import Image
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
import torch
import torch.nn.functional as F
from torch import nn
import diffusers
import pickle
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    UNet2DConditionModel,
    ImagePipelineOutput
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import Attention, AttnProcessor, XFormersAttnProcessor
from diffusers.utils.import_utils import is_xformers_available

import os
FIRST = True
IDX = 0
FEATURE_BANK_KEY = []
FEATURE_BANK_VALUE = []
OUTPUT_FEATURE = []
INFO_STRENGTH = 1
WEIGHT_1 = 0.0
WEIGHT_2 = 0.0
FEATURE_BANK_KEY_SECOND = []
FEATURE_BANK_VALUE_SECOND = []
OUTPUT_FEATURE_SECOND = []

FEATURE_BANK_KEY_THIRD = []
FEATURE_BANK_VALUE_THIRD = []
OUTPUT_FEATURE_THIRD = []


numpy.random.seed(32)
torch.manual_seed(32)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(32)

def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        img = numpy.random.randint(127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=numpy.uint8)
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)
    
class MyAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("MyAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def _step_new(self, keys, values, noise_key_dim):
        # update the feature bank
        with torch.no_grad():
            global FIRST, IDX, PATH, INFO_STRENGTH
            global FEATURE_BANK_KEY, FEATURE_BANK_VALUE, OUTPUT_FEATURE
            global FEATURE_BANK_KEY_SECOND, FEATURE_BANK_VALUE_SECOND, OUTPUT_FEATURE_SECOND
            global FEATURE_BANK_KEY_THIRD, FEATURE_BANK_VALUE_THIRD, OUTPUT_FEATURE_THIRD
            if INFO_STRENGTH == 1:
                new_keys = torch.cat([keys[:,:noise_key_dim,:]] + [FEATURE_BANK_KEY[IDX][:,:noise_key_dim,:].to('cuda')], dim=1)
            elif INFO_STRENGTH == 2:
                new_keys = torch.cat([keys[:,:noise_key_dim,:]] + [FEATURE_BANK_KEY_SECOND[IDX][:,:noise_key_dim,:].to('cuda')], dim=1)
            elif INFO_STRENGTH == 3:
                new_keys = torch.cat([keys[:,:noise_key_dim,:]] + [FEATURE_BANK_KEY_THIRD[IDX][:,:noise_key_dim,:].to('cuda')], dim=1)
            else: 
                raise ValueError('INFO_STRENGTH should be 1 or 2 or 3')
            m_kv, m_kvo = random_bipartite_soft_matching(metric=new_keys, ratio=0.5)
            return m_kv, m_kvo

    def combine_vectors(self, vectors):

        weight = 0.2 / (len(vectors) - 1)
        V_combined = vectors[-1] * 0.8
        for i in range(0, len(vectors) - 1):
            V_combined = V_combined + weight * vectors[i]
        return V_combined

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        is_self_attention=False,
        blend_a=None,
        blend_b=None,
    ):
        residual = hidden_states

        input_ndim = hidden_states.ndim


        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        initial_key = key.clone()
        initial_value = value.clone()
        # begin modify our pipeline
        if is_self_attention:
            global FIRST
            global IDX
            global INFO_STRENGTH
            global FEATURE_BANK_KEY, FEATURE_BANK_VALUE, OUTPUT_FEATURE, INFO_STRENGTH
            global FEATURE_BANK_KEY_SECOND, FEATURE_BANK_VALUE_SECOND, OUTPUT_FEATURE_SECOND
            global FEATURE_BANK_KEY_THIRD, FEATURE_BANK_VALUE_THIRD, OUTPUT_FEATURE_THIRD
            global WEIGHT_1, WEIGHT_2

            if FIRST == True:
                if INFO_STRENGTH == 1:
                    FEATURE_BANK_KEY.append(key.clone().to('cpu'))
                    FEATURE_BANK_VALUE.append(value.clone().to('cpu'))
                elif INFO_STRENGTH == 2:
                    FEATURE_BANK_KEY_SECOND.append(key.clone().to('cpu'))
                    FEATURE_BANK_VALUE_SECOND.append(value.clone().to('cpu'))
                elif INFO_STRENGTH == 3:
                    FEATURE_BANK_KEY_THIRD.append(key.clone().to('cpu'))
                    FEATURE_BANK_VALUE_THIRD.append(value.clone().to('cpu'))
                else:
                    raise ValueError("INFO_STRENGTH must be 1, 2 or 3")
            else:
                cached_key_present = key.clone()
                cached_value_present = value.clone()
                noise_key_dim = OUTPUT_FEATURE[IDX].shape[1]
                if INFO_STRENGTH == 1:
                    temp_key = FEATURE_BANK_KEY[IDX][:,:noise_key_dim,:].to('cuda')
                    temp_value = FEATURE_BANK_VALUE[IDX][:,:noise_key_dim,:].to('cuda')
                    key = torch.cat([key] + [temp_key], dim=1)
                    value = torch.cat([value] + [temp_value], dim=1)
                elif INFO_STRENGTH == 2:
                    temp_key = WEIGHT_1 * FEATURE_BANK_KEY[IDX][:,:noise_key_dim,:].to('cuda') + (1 - WEIGHT_1) * FEATURE_BANK_KEY_SECOND[IDX][:,:noise_key_dim,:].to('cuda')
                    temp_value = WEIGHT_1 * FEATURE_BANK_VALUE[IDX][:,:noise_key_dim,:].to('cuda') + (1 - WEIGHT_1) * FEATURE_BANK_VALUE_SECOND[IDX][:,:noise_key_dim,:].to('cuda')
                    key = torch.cat([key] + [temp_key], dim=1)
                    value = torch.cat([value] + [temp_value], dim=1)

                elif INFO_STRENGTH == 3:
                    temp_key = WEIGHT_1 * FEATURE_BANK_KEY[IDX][:,:noise_key_dim,:].to('cuda') + WEIGHT_2 * FEATURE_BANK_KEY_SECOND[IDX][:,:noise_key_dim,:].to('cuda') + (1 - WEIGHT_1 - WEIGHT_2)*FEATURE_BANK_KEY_THIRD[IDX][:,:noise_key_dim,:].to('cuda')
                    temp_value = WEIGHT_1 * FEATURE_BANK_VALUE[IDX][:,:noise_key_dim,:].to('cuda') + WEIGHT_2 * FEATURE_BANK_VALUE_SECOND[IDX][:,:noise_key_dim,:].to('cuda') + (1 - WEIGHT_1 - WEIGHT_2)*FEATURE_BANK_VALUE_THIRD[IDX][:,:noise_key_dim,:].to('cuda')
                    key = torch.cat([key] + [temp_key], dim=1)
                    value = torch.cat([value] + [temp_value], dim=1)
                else:
                    raise ValueError("INFO_STRENGTH must be 1, 2, or 3")


                m_kv, m_kvo = self._step_new(cached_key_present, cached_value_present, noise_key_dim)

                if INFO_STRENGTH == 1:
                    new_keys = torch.cat([cached_key_present[:,:noise_key_dim,:]] + [FEATURE_BANK_KEY[IDX][:,:noise_key_dim,:].to('cuda')], dim=1)
                    new_values = torch.cat([cached_value_present[:,:noise_key_dim,:]] + [FEATURE_BANK_VALUE[IDX][:,:noise_key_dim,:].to('cuda')], dim=1)
                    compact_keys, compact_values = m_kv(new_keys, new_values)
                    FEATURE_BANK_KEY[IDX][:,:noise_key_dim,:] = compact_keys.to('cpu')
                    FEATURE_BANK_VALUE[IDX][:,:noise_key_dim,:] = compact_values.to('cpu')
                elif INFO_STRENGTH == 2:
                    new_keys = torch.cat([cached_key_present[:,:noise_key_dim,:]] + [FEATURE_BANK_KEY_SECOND[IDX][:,:noise_key_dim,:].to('cuda')], dim=1)
                    new_values = torch.cat([cached_value_present[:,:noise_key_dim,:]] + [FEATURE_BANK_VALUE_SECOND[IDX][:,:noise_key_dim,:].to('cuda')], dim=1)
                    compact_keys, compact_values = m_kv(new_keys, new_values)
                    FEATURE_BANK_KEY_SECOND[IDX][:,:noise_key_dim,:] = compact_keys.to('cpu')
                    FEATURE_BANK_VALUE_SECOND[IDX][:,:noise_key_dim,:] = compact_values.to('cpu')
                elif INFO_STRENGTH == 3:
                    new_keys = torch.cat([cached_key_present[:,:noise_key_dim,:]] + [FEATURE_BANK_KEY_THIRD[IDX][:,:noise_key_dim,:].to('cuda')], dim=1)
                    new_values = torch.cat([cached_value_present[:,:noise_key_dim,:]] + [FEATURE_BANK_VALUE_THIRD[IDX][:,:noise_key_dim,:].to('cuda')], dim=1)
                    compact_keys, compact_values = m_kv(new_keys, new_values)
                    FEATURE_BANK_KEY_THIRD[IDX][:,:noise_key_dim,:] = compact_keys.to('cpu')
                    FEATURE_BANK_VALUE_THIRD[IDX][:,:noise_key_dim,:] = compact_values.to('cpu')
                else:
                    raise ValueError("INFO_STRENGTH must be 1, 2, or 3 or 4")

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)



        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor


        if is_self_attention:
            if FIRST == True:
                cached_output_states =  hidden_states.clone()
                OUTPUT_FEATURE.append(cached_output_states.to('cpu'))
                OUTPUT_FEATURE_SECOND.append(cached_output_states.to('cpu'))
                OUTPUT_FEATURE_THIRD.append(cached_output_states.to('cpu'))
            else:
                cached_output_states =  hidden_states.clone()
                if INFO_STRENGTH == 1:
                    nn_hidden_states = get_nn_feats(hidden_states, OUTPUT_FEATURE[IDX].to('cuda'), 0.98)
                    hidden_states = hidden_states * (1-0.8) + nn_hidden_states * 0.8
                    # update the feature bank
                    new_outs = torch.cat([cached_output_states] + [OUTPUT_FEATURE[IDX].to('cuda')], dim=1)
                    _, _, compact_outs = m_kvo(new_keys, new_values, new_outs)
                    OUTPUT_FEATURE[IDX] = compact_outs.to('cpu')
                elif INFO_STRENGTH == 2:
                    nn_hidden_states = get_nn_feats(hidden_states, OUTPUT_FEATURE_SECOND[IDX].to('cuda'), 0.98)
                    hidden_states = hidden_states * (1-0.8) + nn_hidden_states * 0.8
                    # update the feature bank
                    new_outs = torch.cat([cached_output_states] + [OUTPUT_FEATURE_SECOND[IDX].to('cuda')], dim=1)
                    _, _, compact_outs = m_kvo(new_keys, new_values, new_outs)
                    OUTPUT_FEATURE_SECOND[IDX] = compact_outs.to('cpu')
                elif INFO_STRENGTH == 3:
                    nn_hidden_states = get_nn_feats(hidden_states, OUTPUT_FEATURE_THIRD[IDX].to('cuda'), 0.98)
                    hidden_states = hidden_states * (1-0.8) + nn_hidden_states * 0.8
                    # update the feature bank
                    new_outs = torch.cat([cached_output_states] + [OUTPUT_FEATURE_THIRD[IDX].to('cuda')], dim=1)
                    _, _, compact_outs = m_kvo(new_keys, new_values, new_outs)
                    OUTPUT_FEATURE_THIRD[IDX] = compact_outs.to('cpu')
                else:
                    raise ValueError('INFO_STRENGTH must be 1, 2 or 3')

            IDX = IDX + 1 
        return hidden_states

class ReferenceOnlyAttnProc(torch.nn.Module):
    def __init__(
        self,
        chained_proc,
        enabled=False,
        name=None
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.chained_proc = chained_proc
        self.name = name

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None,
        mode="w", ref_dict: dict = None, is_cfg_guidance = False
    ) -> Any:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        is_self_attention = False

        if self.enabled:
            if mode == 'w':
                ref_dict[self.name] = encoder_hidden_states
            elif mode == 'r':
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict.pop(self.name)], dim=1)
                is_self_attention = True
            elif mode == 'm':
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict[self.name]], dim=1)
            else:
                assert False, mode
        res = self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask,is_self_attention=is_self_attention)
        return res


class RefOnlyNoisedUNet(torch.nn.Module):
    def __init__(self, unet: UNet2DConditionModel, train_sched: DDPMScheduler, val_sched: EulerAncestralDiscreteScheduler) -> None:
        super().__init__()
        self.unet = unet
        self.train_sched = train_sched
        self.val_sched = val_sched
        unet_lora_attn_procs = dict()
        for name, _ in unet.attn_processors.items():
            if torch.__version__ >= '2.0':
                default_attn_proc = MyAttnProcessor2_0()
            elif is_xformers_available():
                default_attn_proc = XFormersAttnProcessor()
            else:
                default_attn_proc = AttnProcessor()
            unet_lora_attn_procs[name] = ReferenceOnlyAttnProc(
                default_attn_proc, enabled=name.endswith("attn1.processor"), name=name
            )
        unet.set_attn_processor(unet_lora_attn_procs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward_cond(self, noisy_cond_lat, timestep, encoder_hidden_states, class_labels, ref_dict, is_cfg_guidance, **kwargs):
        if is_cfg_guidance:
            encoder_hidden_states = encoder_hidden_states[1:]
            class_labels = class_labels[1:]
            print("class_label!!!!!!!!!!!!!!!!!!!!!")
        self.unet(
            noisy_cond_lat, timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            cross_attention_kwargs=dict(mode="w", ref_dict=ref_dict),
            **kwargs
        )

    def forward(
        self, sample, timestep, encoder_hidden_states, class_labels=None,
        *args, cross_attention_kwargs,
        down_block_res_samples=None, mid_block_res_sample=None,
        **kwargs
    ):
        
        cond_lat = cross_attention_kwargs['cond_lat']
        is_cfg_guidance = cross_attention_kwargs.get('is_cfg_guidance', False)
        noise = torch.randn_like(cond_lat)
        if self.training:
            noisy_cond_lat = self.train_sched.add_noise(cond_lat, noise, timestep)
            noisy_cond_lat = self.train_sched.scale_model_input(noisy_cond_lat, timestep)
        # use scheduler to control
        else:
            noisy_cond_lat = self.val_sched.add_noise(cond_lat, noise, timestep.reshape(-1))
            noisy_cond_lat = self.val_sched.scale_model_input(noisy_cond_lat, timestep.reshape(-1))

        ref_dict = {}
        self.forward_cond(
            noisy_cond_lat, timestep,
            encoder_hidden_states, class_labels,
            ref_dict, is_cfg_guidance, **kwargs
        )

        weight_dtype = self.unet.dtype
        return self.unet(
            sample, timestep,
            encoder_hidden_states, *args,
            class_labels=class_labels,
            cross_attention_kwargs=dict(mode="r", ref_dict=ref_dict, is_cfg_guidance=is_cfg_guidance),
            down_block_additional_residuals=[
                sample.to(dtype=weight_dtype) for sample in down_block_res_samples
            ] if down_block_res_samples is not None else None,
            mid_block_additional_residual=(
                mid_block_res_sample.to(dtype=weight_dtype)
                if mid_block_res_sample is not None else None
            ),
            **kwargs
        )


def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents


def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents


def scale_image(image):
    image = image * 0.5 / 0.8
    return image


def unscale_image(image):
    image = image / 0.5 * 0.8
    return image


class Zero123PlusPipeline(diffusers.StableDiffusionPipeline):
    tokenizer: transformers.CLIPTokenizer
    text_encoder: transformers.CLIPTextModel
    vision_encoder: transformers.CLIPVisionModelWithProjection

    feature_extractor_clip: transformers.CLIPImageProcessor
    unet: UNet2DConditionModel
    scheduler: diffusers.schedulers.KarrasDiffusionSchedulers

    vae: AutoencoderKL
    ramping: nn.Linear

    feature_extractor_vae: transformers.CLIPImageProcessor

    depth_transforms_multi = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        vision_encoder: transformers.CLIPVisionModelWithProjection,
        feature_extractor_clip: CLIPImageProcessor, 
        feature_extractor_vae: CLIPImageProcessor,
        ramping_coefficients: Optional[list] = None,
        safety_checker=None,
    ):

        DiffusionPipeline.__init__(self)

        self.register_modules(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            unet=unet, scheduler=scheduler, safety_checker=None,
            vision_encoder=vision_encoder,
            feature_extractor_clip=feature_extractor_clip,
            feature_extractor_vae=feature_extractor_vae
        )
        self.register_to_config(ramping_coefficients=ramping_coefficients)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
    def prepare(self):
        train_sched = DDPMScheduler.from_config(self.scheduler.config)
        if isinstance(self.unet, UNet2DConditionModel):
            self.unet = RefOnlyNoisedUNet(self.unet, train_sched, self.scheduler).eval()

    def encode_condition_image(self, image: torch.Tensor):
        image = self.vae.encode(image).latent_dist.sample()
        return image
    
        
    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image = None,
        prompt = "",
        *args,
        num_images_per_prompt: Optional[int] = 1,
        guidance_scale=4.0,
        depth_image: Image.Image = None,
        output_type: Optional[str] = "pil",
        width=640,
        height=960,
        num_inference_steps=28,
        return_dict=True,
        is_first = False,
        info_strength = 5,
        weight_b1 = 0.0,
        weight_b2 = 0.0,
        weight_b3 = 0.0,
        **kwargs
    ):
        global FIRST
        FIRST = is_first
        global IDX
        IDX = 0
        global INFO_STRENGTH
        INFO_STRENGTH = info_strength
        global WEIGHT_1, WEIGHT_2
        WEIGHT_1 = weight_b1
        WEIGHT_2 = weight_b2
        
        if is_first:
            global FEATURE_BANK_KEY,FEATURE_BANK_VALUE,OUTPUT_FEATURE
            global FEATURE_BANK_KEY_SECOND,FEATURE_BANK_VALUE_SECOND,OUTPUT_FEATURE_SECOND
            global FEATURE_BANK_KEY_THIRD,FEATURE_BANK_VALUE_THIRD,OUTPUT_FEATURE_THIRD
            if info_strength == 1:
                FEATURE_BANK_VALUE=[]
                FEATURE_BANK_KEY=[]
                OUTPUT_FEATURE=[]
            elif info_strength == 2:
                FEATURE_BANK_VALUE_SECOND=[]
                FEATURE_BANK_KEY_SECOND=[]
                OUTPUT_FEATURE_SECOND=[]
            elif info_strength == 3:
                FEATURE_BANK_KEY_THIRD=[]
                FEATURE_BANK_VALUE_THIRD=[]
                OUTPUT_FEATURE_THIRD=[]
            else:
                raise ValueError("info strength must be 1, 2 or 3")

        
        generator = torch.Generator(device='cuda')
        generator.manual_seed(42)
        self.prepare()
        if image is None:
            raise ValueError("Inputting embeddings not supported for this pipeline. Please pass an image.")
        assert not isinstance(image, torch.Tensor)

        image = to_rgb_image(image)
        image_1 = self.feature_extractor_vae(images=image, return_tensors="pt").pixel_values
        image_2 = self.feature_extractor_clip(images=image, return_tensors="pt").pixel_values

        if depth_image is not None and hasattr(self.unet, "controlnet"):
            depth_image = to_rgb_image(depth_image)
            depth_image = self.depth_transforms_multi(depth_image).to(
                device=self.unet.controlnet.device, dtype=self.unet.controlnet.dtype
            )
        
        image = image_1.to(device=self.vae.device, dtype=self.vae.dtype)
        image_2 = image_2.to(device=self.vae.device, dtype=self.vae.dtype)
        cond_lat = self.encode_condition_image(image)

        if guidance_scale > 1:
            negative_lat = self.encode_condition_image(torch.zeros_like(image))
            cond_lat = torch.cat([negative_lat, cond_lat])
        
        encoded = self.vision_encoder(image_2, output_hidden_states=False)
        global_embeds = encoded.image_embeds
        global_embeds = global_embeds.unsqueeze(-2)
        
        encoder_hidden_states = self._encode_prompt(
            prompt,
            self.device,
            num_images_per_prompt,
            False
        )

        ramp = global_embeds.new_tensor(self.config.ramping_coefficients).unsqueeze(-1)
        encoder_hidden_states = encoder_hidden_states + global_embeds * ramp
        cak = dict(cond_lat=cond_lat)
        
        cak['cond_lat_back'] = None
        latents: torch.Tensor = super().__call__(
            None,
            *args,
            cross_attention_kwargs=cak,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=encoder_hidden_states,
            num_inference_steps=num_inference_steps,
            output_type='latent',
            width=width,
            height=height,
            generator=generator,
            **kwargs
        ).images
        latents = unscale_latents(latents)
        if not output_type == "latent":
            image = unscale_image(self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0])
        else:
            image = latents
        image = self.image_processor.postprocess(image, output_type=output_type)
        if not return_dict:
            return (image,)
        

        return ImagePipelineOutput(images=image)
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    raise Exception("unable to initialize CUDA")

import os
import gc
import random
import numpy as np
import argparse
import json
import datetime
from tqdm import tqdm
import decord
from contextlib import contextmanager
from time import perf_counter
from glob import glob
from PIL import Image

from torchvision.transforms import v2, InterpolationMode
from safetensors.torch import load_file, save_file

from transformers import CLIPTextModel, CLIPTokenizerFast, LlamaModel, LlamaTokenizerFast
from diffusers import AutoencoderKLHunyuanVideo, HunyuanVideoTransformer3DModel, FlowMatchEulerDiscreteScheduler
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
import bitsandbytes as bnb


DEFAULT_PROMPT_TEMPLATE = {
    "template": (
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
        "1. The main content and theme of the video."
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
        "4. background environment, light, style and atmosphere."
        "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    ),
    "crop_start": 95,
}

IMAGE_TYPES = [".jpg", ".png"]
VIDEO_TYPES = [".mp4", ".mkv", ".mov", ".avi", ".webm"]

BUCKET_RESOLUTIONS = {
    "1x1": [
        # (256, 256),
        (384, 384),
        (512, 512),
        (640, 640),
        (720, 720),
        (768, 768),
        (848, 848),
        (960, 960),
        (1024, 1024),
    ],
    "4x3": [
        # (320, 240),
        (384, 288),
        (512, 384),
        (640, 480),
        (768, 576),
        (960, 720),
        (1024, 768),
        (1152, 864),
        (1280, 960),
    ],
    "16x9": [
        # (256, 144),
        (512, 288),
        (768, 432),
        (848, 480),
        (960, 544),
        (1024, 576),
        (1280, 720),
    ],
}

def count_tokens(width, height, frames):
    return (width // 16) * (height // 16) * ((frames - 1) // 4 + 1)


class CombinedDataset(Dataset):
    def __init__(
        self,
        root_folder,
        token_limit = 10_000,
        limit_samples = None,
        max_frame_stride = 4,
        manual_resolution = None,
        manual_frames = None,
    ):
        self.root_folder = root_folder
        self.token_limit = token_limit
        self.max_frame_stride = max_frame_stride
        self.manual_resolution = manual_resolution
        self.manual_frames = manual_frames
        
        # search for all files matching image or video extensions
        self.media_files = []
        for ext in IMAGE_TYPES + VIDEO_TYPES:
            self.media_files.extend(
                glob(os.path.join(self.root_folder, "**", "*" + ext), recursive=True)
            )
        
        # pull samples evenly from the whole dataset
        if limit_samples is not None:
            stride = max(1, len(self.media_files) // limit_samples)
            self.media_files = self.media_files[::stride]
            self.media_files = self.media_files[:limit_samples]
    
    def __len__(self):
        return len(self.media_files)
    
    def find_max_frames(self, width, height):
        if self.manual_frames is not None:
            return self.manual_frames
        
        frames = 1
        tokens = count_tokens(width, height, frames)
        while tokens < self.token_limit:
            new_frames = frames + 4
            new_tokens = count_tokens(width, height, new_frames)
            if new_tokens < self.token_limit:
                frames = new_frames
                tokens = new_tokens
            else:
                return frames
    
    def get_ar_buckets(self, width, height):
        if self.manual_resolution is not None:
            return [(self.manual_resolution, self.manual_resolution)]
        
        ar = width / height
        if ar > 1.555:
            buckets = BUCKET_RESOLUTIONS["16x9"]
        elif ar > 1.166:
            buckets = BUCKET_RESOLUTIONS["4x3"]
        elif ar > 0.875:
            buckets = BUCKET_RESOLUTIONS["1x1"]
        elif ar > 0.656:
            buckets = [b[::-1] for b in BUCKET_RESOLUTIONS["4x3"]]
        else:
            buckets = [b[::-1] for b in BUCKET_RESOLUTIONS["16x9"]]
        
        return [b for b in buckets if b[0] <= width and b[1] <= height]
    
    def __getitem__(self, idx):
        ext = os.path.splitext(self.media_files[idx])[1].lower()
        if ext in IMAGE_TYPES:
            image = Image.open(self.media_files[idx]).convert('RGB')
            pixels = torch.as_tensor(np.array(image)).unsqueeze(0) # FHWC
            buckets = self.get_ar_buckets(pixels.shape[2], pixels.shape[1])
            width, height = random.choice(buckets)
        else:
            vr = decord.VideoReader(self.media_files[idx])
            orig_height, orig_width = vr[0].shape[:2]
            orig_frames = len(vr)
            
            # randomize resolution bucket and frame length
            buckets = self.get_ar_buckets(orig_width, orig_height)
            width, height = random.choice(buckets)
            max_frames = self.find_max_frames(width, height)
            stride = max(min(random.randint(1, self.max_frame_stride), orig_frames // max_frames), 1)
            
            # sample a clip from the video based on frame stride and length
            seg_len = min(stride * max_frames, orig_frames)
            start_frame = random.randint(0, orig_frames - seg_len)
            pixels = vr[start_frame : start_frame+seg_len : stride]
            max_frames = ((pixels.shape[0] - 1) // 4) * 4 + 1
            pixels = pixels[:max_frames] # clip frames to match vae
        
        # determine crop dimensions to prevent stretching during resize
        pixels_ar = pixels.shape[2] / pixels.shape[1]
        target_ar = width / height
        if pixels_ar > target_ar:
            crop_width = min(int(pixels.shape[1] * target_ar), pixels.shape[2])
            crop_height = pixels.shape[1]
        elif pixels_ar < target_ar:
            crop_width = pixels.shape[2]
            crop_height = min(int(pixels.shape[2] / target_ar), pixels.shape[1])
        else:
            crop_width = pixels.shape[2]
            crop_height = pixels.shape[1]
        
        # convert to expected dtype, resolution, shape, and value range
        transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomCrop(size=(crop_height, crop_width)),
            v2.Resize(size=(height, width)),
        ])
        
        pixels = pixels.movedim(3, 1).unsqueeze(0).contiguous() # FHWC -> FCHW -> BFCHW
        pixels = transform(pixels) * 2 - 1
        pixels = torch.clamp(torch.nan_to_num(pixels), min=-1, max=1)
        
        # load precomputed text embeddings from file
        embedding_file = os.path.splitext(self.media_files[idx])[0] + "_hyv.safetensors"
        if not os.path.exists(embedding_file):
            embedding_file = os.path.join(
                os.path.dirname(self.media_files[idx]),
                random.choice(["caption_original_hyv.safetensors", "caption_florence_hyv.safetensors"]),
            )
        
        if os.path.exists(embedding_file):
            embedding_dict = load_file(embedding_file)
        else:
            raise Exception(f"No embedding file found for {self.media_files[idx]}, you may need to precompute embeddings with --cache_embeddings")
        
        return {"pixels": pixels, "embedding_dict": embedding_dict}


@contextmanager
def temp_rng(new_seed=None):
	"""
    https://github.com/fpgaminer/bigasp-training/blob/main/utils.py#L73
	Context manager that saves and restores the RNG state of PyTorch, NumPy and Python.
	If new_seed is not None, the RNG state is set to this value before the context is entered.
	"""

	# Save RNG state
	old_torch_rng_state = torch.get_rng_state()
	old_torch_cuda_rng_state = torch.cuda.get_rng_state()
	old_numpy_rng_state = np.random.get_state()
	old_python_rng_state = random.getstate()

	# Set new seed
	if new_seed is not None:
		torch.manual_seed(new_seed)
		torch.cuda.manual_seed(new_seed)
		np.random.seed(new_seed)
		random.seed(new_seed)

	yield

	# Restore RNG state
	torch.set_rng_state(old_torch_rng_state)
	torch.cuda.set_rng_state(old_torch_cuda_rng_state)
	np.random.set_state(old_numpy_rng_state)
	random.setstate(old_python_rng_state)


@contextmanager
def timer(message=""):
    start_time = perf_counter()
    yield
    end_time = perf_counter()
    print(f"{message} {end_time - start_time:0.2f} seconds")


def make_dir(base, folder):
    new_dir = os.path.join(base, folder)
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


def download_model(args):
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_type = "model",
        repo_id = "hunyuanvideo-community/HunyuanVideo",
        local_dir = "./models",
        max_workers = 1,
    )
    
    if args.skyreels_i2v:
        snapshot_download(
            repo_type = "model",
            repo_id = "Skywork/SkyReels-V1-Hunyuan-I2V",
            local_dir = "./models/transformer-skyreels-i2v",
            max_workers = 1,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description = "HunyuanVideo training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument(
        "--download_model",
        action = "store_true",
        help = "auto download hunyuanvideo-community/HunyuanVideo to ./models if it's missing",
        )
    parser.add_argument(
        "--skyreels_i2v",
        action = "store_true",
        help = "download/train skyreels image to video model",
        )
    parser.add_argument(
        "--cache_embeddings",
        action = "store_true",
        help = "preprocess dataset to encode captions",
        )
    parser.add_argument(
        "--pretrained_model",
        type = str,
        default = "./models",
        help = "Path to pretrained model base directory",
        )
    parser.add_argument(
        "--quant_type",
        type = str,
        default = "nf4",
        help = "Bit depth for the base model, default config with nf4=16GB",
        choices=["nf4", "int8", "bf16"],
        )
    parser.add_argument(
        "--init_lora",
        type = str,
        default = None,
        help = "LoRA checkpoint to load instead of random init, must be the same rank and target layers",
        )
    parser.add_argument(
        "--dataset",
        type = str,
        required = True,
        help = "Path to dataset directory with train and val subdirectories",
        )
    parser.add_argument(
        "--val_samples",
        type = int,
        default = 4,
        help = "Maximum number of videos to use for validation loss"
        )
    parser.add_argument(
        "--output_dir",
        type = str,
        default = "./outputs",
        help = "Output directory for training results"
        )
    parser.add_argument(
        "--seed",
        type = int,
        default = 42,
        help = "Seed for reproducible training"
        )
    parser.add_argument(
        "--resolution",
        type = int,
        default = None,
        help = "Manual override resolution for training/testing"
        )
    parser.add_argument(
        "--num_frames",
        type = int,
        default = None,
        help = "Manual override number of frames per video, must be divisible by 4+1"
        )
    parser.add_argument(
        "--token_limit",
        type = int,
        default = 10_000,
        help = "Combined resolution/frame limit based on transformer patch sequence length: (width // 16) * (height // 16) * ((frames - 1) // 4 + 1)"
        )
    parser.add_argument(
        "--max_frame_stride",
        type = int,
        default = 2,
        help = "1: use native framerate only. Higher values allow randomly choosing lower framerates (skipping frames to speed up the video)"
        )
    parser.add_argument(
        "--learning_rate",
        type = float,
        default = 2e-4,
        help = "Base learning rate",
        )
    parser.add_argument(
        "--lora_rank",
        type = int,
        default = 16,
        help = "The dimension of the LoRA update matrices",
        )
    parser.add_argument(
        "--lora_alpha",
        type = int,
        default = None,
        help = "The alpha value for LoRA, defaults to alpha=rank. Note: changing alpha will affect the learning rate, and if alpha=rank then changing rank will also affect learning rate",
        )
    parser.add_argument(
        "--val_steps",
        type = int,
        default = 50,
        help = "Validate after every n steps",
        )
    parser.add_argument(
        "--checkpointing_steps",
        type = int,
        default = 50,
        help = "Save a checkpoint of the training state every X steps",
        )
    parser.add_argument(
        "--max_train_steps",
        type = int,
        default = 1000,
        help = "Total number of training steps",
        )
    parser.add_argument(
        "--warped_noise",
        action = "store_true",
        help = "Use warped noise from Go-With-The-Flow instead of pure random noise",
        )
    
    args = parser.parse_args()
    return args


def cache_embeddings(args):
    print("loading CLIP")
    with timer("loaded CLIP in"):
        tokenizer_clip = CLIPTokenizerFast.from_pretrained(args.pretrained_model, subfolder="tokenizer_2")
        text_encoder_clip = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder_2").to(device=device, dtype=torch.bfloat16)
        text_encoder_clip.requires_grad_(False)
    
    print("loading Llama")
    with timer("loaded Llama in"):
        tokenizer_llama = LlamaTokenizerFast.from_pretrained(args.pretrained_model, subfolder="tokenizer")
        text_encoder_llama = LlamaModel.from_pretrained(args.pretrained_model, subfolder="text_encoder").to(device=device, dtype=torch.bfloat16)
        text_encoder_llama.requires_grad_(False)
    
    def encode_clip(prompt):
        input_ids = tokenizer_clip(
            prompt,
            padding = "max_length",
            max_length = 77,
            truncation = True,
            return_tensors = "pt",
        ).input_ids.to(text_encoder_clip.device)
        
        prompt_embeds = text_encoder_clip(
            input_ids,
            output_hidden_states = False,
        ).pooler_output
        
        return prompt_embeds
    
    def encode_llama(
        prompt,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        max_sequence_length = 256,
        num_hidden_layers_to_skip = 2,
    ):
        prompt = prompt_template["template"].format(prompt)
        crop_start = prompt_template.get("crop_start", None)
        max_sequence_length += crop_start
        
        text_inputs = tokenizer_llama(
            prompt,
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=True,
        )
        text_input_ids = text_inputs.input_ids.to(device=text_encoder_llama.device)
        prompt_attention_mask = text_inputs.attention_mask.to(device=text_encoder_llama.device)
        
        prompt_embeds = text_encoder_llama(
            input_ids = text_input_ids,
            attention_mask = prompt_attention_mask,
            output_hidden_states = True,
        ).hidden_states[-(num_hidden_layers_to_skip + 1)]
        
        if crop_start is not None and crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_attention_mask = prompt_attention_mask[:, crop_start:]
        
        return prompt_embeds, prompt_attention_mask
    
    def preprocess_captions(dataset_path):
        caption_files = glob(os.path.join(dataset_path, "**", "*.txt" ), recursive=True)
        for file in tqdm(caption_files):
            embedding_path = os.path.splitext(file)[0] + "_hyv.safetensors"
            
            if not os.path.exists(embedding_path):
                with open(file, "r") as f:
                    caption = f.read()
                
                clip_embed = encode_clip(caption)
                llama_embed, llama_mask = encode_llama(caption)
                
                embedding_dict = {"clip_embed": clip_embed, "llama_embed": llama_embed, "llama_mask": llama_mask}
                save_file(embedding_dict, embedding_path)
    
    print("preprocessing caption embeddings")
    preprocess_captions(args.dataset)
    
    del tokenizer_clip, text_encoder_clip, tokenizer_llama, text_encoder_llama
    gc.collect()
    torch.cuda.empty_cache()


def main(args):
    decord.bridge.set_bridge('torch')
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    real_output_dir = make_dir(args.output_dir, date_time)
    checkpoint_dir = make_dir(real_output_dir, "checkpoints")
    t_writer = SummaryWriter(log_dir=real_output_dir, flush_secs=60)
    with open(os.path.join(real_output_dir, "command_args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)
    
    def collate_batch(batch):
        pixels = torch.cat([sample["pixels"] for sample in batch], dim=0)
        clip_embed = torch.cat([sample["embedding_dict"]["clip_embed"] for sample in batch], dim=0)
        llama_embed = torch.cat([sample["embedding_dict"]["llama_embed"] for sample in batch], dim=0)
        llama_mask = torch.cat([sample["embedding_dict"]["llama_mask"] for sample in batch], dim=0)
        return pixels, clip_embed, llama_embed, llama_mask
    
    train_dataset = os.path.join(args.dataset, "train")
    if not os.path.exists(train_dataset):
        train_dataset = args.dataset
        print(f"WARNING: train subfolder not found, using root folder {train_dataset} as train dataset")
    
    val_dataset = None
    for subfolder in ["val", "validation", "test"]:
        subfolder_path = os.path.join(args.dataset, subfolder)
        if os.path.exists(subfolder_path):
            val_dataset = subfolder_path
            break
    
    if val_dataset is None:
        val_dataset = args.dataset
        print(f"WARNING: val/validation/test subfolder not found, using root folder {val_dataset} for stable loss validation")
        print("\033[33mThis will make it impossible to judge overfitting by the validation loss. Using a val split held out from training is highly recommended\033[m")
    
    with timer("scanned dataset in"):
        train_dataset = CombinedDataset(
            root_folder = train_dataset,
            token_limit = args.token_limit,
            max_frame_stride = args.max_frame_stride,
            manual_resolution = args.resolution,
            manual_frames = args.num_frames,
        )
        val_dataset = CombinedDataset(
            root_folder = val_dataset,
            token_limit = args.token_limit,
            limit_samples = args.val_samples,
            max_frame_stride = args.max_frame_stride,
            manual_resolution = args.resolution,
            manual_frames = args.num_frames,
        )
    
    train_dataloader = DataLoader(
        train_dataset,
        shuffle = True,
        collate_fn = collate_batch,
        batch_size = 1,
        num_workers = 0,
        pin_memory = True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle = False,
        collate_fn = collate_batch,
        batch_size = 1,
        num_workers = 0,
        pin_memory = True,
    )
    
    # noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
    
    with timer("loaded VAE in"):
        vae = AutoencoderKLHunyuanVideo.from_pretrained(args.pretrained_model, subfolder="vae").to(device=device, dtype=torch.float16)
        vae.requires_grad_(False)
        vae.enable_tiling(
            tile_sample_min_height = 256,
            tile_sample_min_width = 256,
            tile_sample_min_num_frames = 48,
            tile_sample_stride_height = 192,
            tile_sample_stride_width = 192,
            tile_sample_stride_num_frames = 32,
        )
    
    with timer("loaded diffusion model in"):
        if args.quant_type == "nf4":
            quant_config = DiffusersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif args.quant_type == "int8":
            quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_config = None
        
        if args.skyreels_i2v:
            transformer_subfolder = "transformer-skyreels-i2v"
        else:
            transformer_subfolder = "transformer"
        
        diffusion_model = HunyuanVideoTransformer3DModel.from_pretrained(
            args.pretrained_model,
            subfolder = transformer_subfolder,
            quantization_config = quant_config,
            torch_dtype = torch.bfloat16,
        )
        
        diffusion_model.requires_grad_(False)
        diffusion_model.enable_gradient_checkpointing()
        torch.cuda.empty_cache()
    
    with timer("added LoRA in"):
        lora_params = []
        attn_blocks = ["transformer_blocks", "single_transformer_blocks"]
        lora_keys = ["to_k", "to_q", "to_v", "to_out.0", "proj_mlp"] # mmdit img attention + single blocks attention
        # lora_keys += ["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"] # mmdit text attention
        # lora_keys += ["ff.net", "proj_out"] # mmdit img mlp + single blocks mlp
        # lora_keys += ["ff_context.net"] # mmdit text mlp
        for name, param in diffusion_model.named_parameters():
            name = name.replace(".weight", "").replace(".bias", "")
            for block in attn_blocks:
                if name.startswith(block):
                    for key in lora_keys:
                        if key in name:
                            lora_params.append(name)
        
        lora_config = LoraConfig(
            r = args.lora_rank,
            lora_alpha = args.lora_alpha or args.lora_rank,
            init_lora_weights = "gaussian",
            target_modules = lora_params,
        )
        diffusion_model.add_adapter(lora_config)
        
        if args.init_lora is not None:
            loaded_lora_sd = load_file(args.init_lora)
            outcome = set_peft_model_state_dict(diffusion_model, loaded_lora_sd)
            if len(outcome.unexpected_keys) > 0:
                for key in outcome.unexpected_keys:
                    print(key)
        
        lora_parameters = []
        total_parameters = 0
        for param in diffusion_model.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)
                lora_parameters.append(param)
                total_parameters += param.numel()
    print(f"total trainable parameters: {total_parameters:,}")
    
    # Instead of having just one optimizer, we will have a dict of optimizers
    # for every parameter so we could reference them in our hook.
    optimizer_dict = {p: bnb.optim.AdamW8bit([p], lr=args.learning_rate) for p in lora_parameters}
    
    # Define our hook, which will call the optimizer step() and zero_grad()
    def optimizer_hook(parameter) -> None:
        optimizer_dict[parameter].step()
        optimizer_dict[parameter].zero_grad()
    
    # Register the hook onto every trainable parameter
    for p in lora_parameters:
        p.register_post_accumulate_grad_hook(optimizer_hook)
    
    if args.warped_noise:
        from noise_warp.GetWarpedNoiseFromVideo import GetWarpedNoiseFromVideo
        get_warped_noise = GetWarpedNoiseFromVideo(raft_size="large", device=device, dtype=torch.float32)
    
    def prepare_conditions(batch):
        pixels, clip_embed, llama_embed, llama_mask = batch
        pixels = pixels.movedim(1, 2).to(device=vae.device, dtype=vae.dtype) # BFCHW -> BCFHW
        latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
        
        if args.skyreels_i2v:
            image_cond_latents = torch.zeros_like(latents)
            image_latents = vae.encode(pixels[:, :, 0].unsqueeze(2)).latent_dist.sample() * vae.config.scaling_factor
            image_cond_latents[:, :, 0] = image_latents[:, :, 0]
            del image_latents
        
        t_writer.add_scalar("debug/context_len", latents.shape[-3] * (latents.shape[-2] / 2) * (latents.shape[-1] / 2), global_step)
        t_writer.add_scalar("debug/width", pixels.shape[-1], global_step)
        t_writer.add_scalar("debug/height", pixels.shape[-2], global_step)
        t_writer.add_scalar("debug/frames", pixels.shape[-3], global_step)
        
        if args.warped_noise:
            noise = get_warped_noise(
                pixels.movedim(2, 1)[0], # BCFHW -> BFCHW -> FCHW
                degradation = torch.rand(1).item(),
                noise_channels = 16,
                target_latent_count = latents.shape[2],
            ).movedim(0, 1).unsqueeze(0).to(latents) # FCHW -> CFHW -> BCFHW
        else:
            noise = torch.randn_like(latents)
        
        # TODO: add sd3/flux timestep density sampling?
        sigma = torch.rand(latents.shape[0])
        timesteps = torch.round(sigma * 1000).long()
        sigma = sigma[:, None, None, None, None].to(latents)
        noisy_model_input = (noise * sigma) + (latents * (1 - sigma))
        
        if args.skyreels_i2v:
            noisy_model_input = torch.cat([noisy_model_input, image_cond_latents], dim=1)
        
        guidance_scale = 1.0
        guidance = torch.tensor([guidance_scale] * latents.shape[0], dtype=torch.float32, device=device) * 1000.0
        
        return {
            "target":            (noise - latents).to(device=device),
            "noisy_model_input": noisy_model_input.to(device=device, dtype=torch.bfloat16),
            "timesteps":                 timesteps.to(device=device, dtype=torch.bfloat16),
            "llama_embed":             llama_embed.to(device=device, dtype=torch.bfloat16),
            "llama_mask":               llama_mask.to(device=device, dtype=torch.bfloat16),
            "clip_embed":               clip_embed.to(device=device, dtype=torch.bfloat16),
            "guidance":                   guidance.to(device=device, dtype=torch.bfloat16),
        }
    
    def predict_loss(conditions):
        pred = diffusion_model(
            hidden_states          = conditions["noisy_model_input"],
            timestep               = conditions["timesteps"],
            encoder_hidden_states  = conditions["llama_embed"],
            encoder_attention_mask = conditions["llama_mask"],
            pooled_projections     = conditions["clip_embed"],
            guidance               = conditions["guidance"],
            return_dict = False,
        )[0]
        return F.mse_loss(pred.float(), conditions["target"].float())
    
    gc.collect()
    torch.cuda.empty_cache()
    diffusion_model.train()
    
    global_step = 0
    progress_bar = tqdm(range(0, args.max_train_steps))
    while global_step < args.max_train_steps:
        for step, batch in enumerate(train_dataloader):
            start_step = perf_counter()
            with torch.inference_mode():
                conditions = prepare_conditions(batch)
            
            torch.cuda.empty_cache()
            loss = predict_loss(conditions)
            t_writer.add_scalar("loss/train", loss.detach().item(), global_step)
            
            loss.backward()
            
            progress_bar.update(1)
            global_step += 1
            torch.cuda.empty_cache()
            t_writer.add_scalar("debug/step_time", perf_counter() - start_step, global_step)
            
            if global_step == 1 or global_step % args.val_steps == 0:
                with torch.inference_mode(), temp_rng(args.seed):
                    val_loss = 0.0
                    for step, batch in enumerate(tqdm(val_dataloader, desc="validation", leave=False)):
                        conditions = prepare_conditions(batch)
                        torch.cuda.empty_cache()
                        loss = predict_loss(conditions)
                        val_loss += loss.detach().item()
                        torch.cuda.empty_cache()
                    t_writer.add_scalar("loss/validation", val_loss / len(val_dataloader), global_step)
                progress_bar.unpause()
            
            if global_step >= args.max_train_steps or global_step % args.checkpointing_steps == 0:
                save_file(
                    get_peft_model_state_dict(diffusion_model),
                    os.path.join(checkpoint_dir, f"hyv-lora-{global_step:08}.safetensors"),
                )
            
            if global_step >= args.max_train_steps:
                break

# train
    # basic t2i and randn noise to start
    # guidance=1
    # uncond/caption dropout?


if __name__ == "__main__":
    args = parse_args()
    
    if args.download_model:
        download_model(args)
        exit()
    
    if args.cache_embeddings and args.dataset != "pexels":
        cache_embeddings(args)
        exit()
    
    main(args)
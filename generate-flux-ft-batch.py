import json
import os
import argparse
import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel
from optimum.quanto import freeze
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from typing import List, Dict
import logging

def load_json(json_path: str) -> Dict:
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def initialise_pipeline(my_flux_ft_safetensor_path: str, dtype=torch.bfloat16, device=None) -> FluxPipeline:
    bfl_repo = "black-forest-labs/FLUX.1-dev"

    transformer = FluxTransformer2DModel.from_single_file(
        f"/bigstor/annaha/diffusion-ft/models/flux/{my_flux_ft_safetensor_path}",
        torch_dtype=dtype
    )
    freeze(transformer)
    transformer.to(device)

    text_encoder_2 = T5EncoderModel.from_pretrained(
        bfl_repo,
        subfolder="text_encoder_2",
        torch_dtype=dtype
    )
    freeze(text_encoder_2)
    text_encoder_2.to(device)

    pipe = FluxPipeline.from_pretrained(
        bfl_repo,
        transformer=None,
        text_encoder_2=None,
        torch_dtype=dtype
    )
    pipe.transformer = transformer
    pipe.text_encoder_2 = text_encoder_2

    # pipe.enable_model_cpu_offload()

    pipe.to(device)

    return pipe

def generate_and_save_images(pipe, prompts, output_dir, seeds, image_ids, device):
    generators = [torch.Generator(device).manual_seed(s) for s in seeds]
    try:
        images = pipe(
            prompt=prompts,
            guidance_scale=3.5,
            output_type="pil",
            num_inference_steps=50,
            generator=generators
        ).images
    except Exception as e:
        logging.error(f"Error during image generation: {e}")
        raise  

    for image, image_id in zip(images, image_ids):
        image_filename = f"{image_id}.png"
        image_path = os.path.join(output_dir, image_filename)
        try:
            image.save(image_path)
        except Exception as e:
            logging.error(f"Error saving image {image_filename}: {e}")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        help='Path to the JSON configuration file.'
    )
    parser.add_argument(
        '--safetensor_model',
        type=str,
        required=True,
        help='The safetensor file model name (e.g., 500-safetensor-4e-6.safetensors).'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Number of images to generate in each batch.'
    )
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - Process {process_index} - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    config_path = args.config_path

    if accelerator.is_local_main_process:
        logger.info(f"Loading configuration from {config_path}...")
    config = load_json(config_path)

    base_output_dir = "/bigstor/annaha/diffusion-ft/generated-imgs/flux-ft"
    model_basename = os.path.splitext(os.path.basename(args.safetensor_model))[0]
    images_output_dir = os.path.join(base_output_dir, model_basename)

    os.makedirs(images_output_dir, exist_ok=True)
    if not os.access(images_output_dir, os.W_OK):
        logger.error(f"Cannot write to the directory {images_output_dir}. Check permissions.")
        return

    if accelerator.is_local_main_process:
        logger.info(f"Images will be saved to {images_output_dir}")
        logger.info(f"Loading fid_prompts from {args.config_path}...")
    data = load_json(args.config_path)

    fid_prompts = data.get("fid_prompts", [])
    if not fid_prompts:
        logger.warning("No fid_prompts found in the JSON file.")
        return

    total_prompts = len(fid_prompts)
    prompts_per_process = total_prompts // num_processes
    remainder = total_prompts % num_processes

    start_index = process_index * prompts_per_process + min(process_index, remainder)
    end_index = start_index + prompts_per_process + (1 if process_index < remainder else 0)
    fid_prompts = fid_prompts[start_index:end_index]

    logger.info(f"Process {process_index}: Assigned {len(fid_prompts)} prompts.")

    if not fid_prompts:
        logger.warning(f"Process {process_index}: No prompts to process.")
        return

    logger.info("Initialising Flux pipeline...")
    pipe = initialise_pipeline(args.safetensor_model, device=device)

    logger.info("Starting image generation...")

    seed = 188459528  

    batch_size = args.batch_size
    total_batches = (len(fid_prompts) + batch_size - 1) // batch_size

    pbar = tqdm(total=total_batches, desc=f"Process {process_index} generating images", position=process_index, leave=False)

    for batch_start in range(0, len(fid_prompts), batch_size):
        batch = fid_prompts[batch_start:batch_start + batch_size]
        prompts = []
        seeds = []
        image_ids = []

        for item in batch:
            image_id = item.get("image_id")
            caption = item.get("caption")

            if image_id is None or caption is None:
                logger.warning(f"Process {process_index}: Skipping invalid entry: {item}")
                continue

            prompts.append(caption)
            image_ids.append(image_id)

        if not prompts:
            pbar.update(1)
            continue

        seeds = [seed] * len(prompts)

        try:
            logger.info(f"Process {process_index}: Generating batch {batch_start // batch_size + 1}/{total_batches} with {len(prompts)} prompts.")
            generate_and_save_images(
                pipe=pipe,
                prompts=prompts,
                output_dir=images_output_dir,
                seeds=seeds,
                image_ids=image_ids,
                device=device
            )
            logger.info(f"Process {process_index}: Batch {batch_start // batch_size + 1} completed successfully.")
        except Exception as e:
            logger.error(f"Process {process_index}: Error generating batch starting at index {batch_start}: {e}")

        pbar.update(1)

    pbar.close()

    if accelerator.is_local_main_process:
        logger.info("Image generation completed.")

if __name__ == "__main__":
    main()


# cd /home/annaha/projects/provenance/diffusion-ft/generate-images/
# use the flux env!! source env/flux_venv/bin/activate
# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes=3 generate-flux-ft-batch.py --config_path /home/annaha/projects/provenance/diffusion-ft/sd-scripts/25000total-clean-warmup1-cogvlm-flux.json --safetensor_model 500-safetensor-4e-6.safetensors --batch_size 4

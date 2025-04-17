import os
import numpy as np
from argparse import ArgumentParser
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from data_cfd import InferDataset
from pipeline import Pipeline
from utils import seed_everything


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument('--prompts', default='')
    parser.add_argument('--lora-ckpt', required=True)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--output', default='infer-output')
    # parser.add_argument('--latents')
    parser.add_argument('--src', type=str, default='data/case_data2/fluent_data_fig')
    
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--save-mid', action='store_true')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--target-mode', default='RGB', choices=['RGB', 'F'])
    parser.add_argument('--target-scale', default=1, type=float)
    parser.add_argument('--target-pred-type', default='v_prediction', choices=['epsilon', 'sample', 'v_prediction'])
    parser.add_argument('--self-attn-only', action='store_true')
    parser.add_argument('--disable-prompts', action='store_true')
    parser.add_argument('--use-oracle-ddim', action='store_true')
    parser.add_argument('--onepass', action='store_true')
    parser.add_argument('--seed', type=int, default=666666)
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()

    if not args.src and not args.prompts:
        raise ValueError('at least provide --prompts or --src.')
   
    cfg = OmegaConf.load(args.config)
    os.makedirs(os.path.join(args.output), exist_ok=True)
    
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pipeline = Pipeline(
            cfg.pretrained,
            self_attn_only=args.self_attn_only,
            disable_prompts=args.disable_prompts,
            onepass=args.onepass,
            prediction_type=args.target_pred_type,
            use_oracle_ddim=args.use_oracle_ddim,
            lora_ckpt=args.lora_ckpt,
            enable_xformers=True,
            device=device,
            mixed_precision='fp16',
    )
    generator = torch.Generator().manual_seed(args.seed)
    dataset = InferDataset(
        args.prompts, 
        pipeline.tokenizer, 
        args.latents, 
        args.src, 
        args.num_samples, 
        generator)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers)

    # Denoise
    for i, batch in enumerate(tqdm(dataloader, ncols=100)):
        
        if args.save_mid:
            for key in batch['key']:
                os.makedirs(f'mid/{args.output}/{key}', exist_ok=True)
        
        output = pipeline.infer_batch(
                text_inputs=batch['text'],
                src_imgs=batch['inputs'],
                src_inference_steps=cfg.num_inference_steps,
                trg_inference_steps=cfg.target_inference_steps,
                src_guidance_scale=cfg.guidance_scale,
                trg_guidance_scale=cfg.target_guidance_scale,
                return_mid_latents=args.save_mid,
        )

        if args.save_mid:
            for ti, t in enumerate(pipeline.scheduler.timesteps):
                mid = pipeline.decode_latents(output.mid_latents[ti])
                for img, key in zip(mid, batch['key']):
                    img.save(f'mid/{args.output}/{key}/{t:03}.png')

        trgs = pipeline.decode_latents(output.trg_latents, args.target_mode, args.target_scale)
        for trg, key in zip(trgs, batch['key']):
            visual, arr = trg
            visual.save(os.path.join(args.output, f'{key}.png'))
            # np.savez_compressed(os.path.join(args.output, f'{key}.npz'), x=arr)


if __name__ == '__main__':
    main()

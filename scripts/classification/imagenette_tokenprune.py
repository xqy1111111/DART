#!/usr/bin/env python3
import os
import sys
import argparse
import time
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
from pathlib import Path
import json
import math

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TokenPruningViT(nn.Module):
    """ViT with token pruning capability"""

    def __init__(self, model_name, method='none', ratio=0.0, prune_layer=6,
                 pivot_tokens=8, device='cuda'):
        super().__init__()
        self.base_model = timm.create_model(model_name, pretrained=True, num_classes=10)
        self.method = method
        self.ratio = ratio
        self.prune_layer = prune_layer
        self.pivot_tokens = pivot_tokens
        self.device = device

        # Get model info
        self.num_layers = len(self.base_model.blocks)
        self.embed_dim = self.base_model.embed_dim

        # Hook for extracting features at prune layer
        self.features = {}
        if method != 'none':
            self.base_model.blocks[prune_layer].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        """Hook to capture intermediate features"""
        self.features['layer_output'] = output

    def get_retained_tokens_dart(self, x, n_keep):
        """DART-style token selection based on duplication"""
        # x shape: [B, N+1, D] where N is num_patches, +1 for cls token
        B, N_total, D = x.shape
        N_patches = N_total - 1  # exclude cls token

        if n_keep >= N_patches:
            return torch.arange(1, N_total, device=x.device)

        # Extract patch tokens (exclude cls)
        patch_tokens = x[:, 1:, :]  # [B, N, D]

        # Select pivot tokens using L1 norm
        norms = torch.norm(patch_tokens[0], p=1, dim=-1)  # [N]
        pivot_indices = norms.topk(self.pivot_tokens).indices

        # Calculate cosine similarity between all tokens and pivots
        all_indices = set(range(N_patches))
        selected_indices = set(pivot_indices.tolist())

        # For each pivot, find most dissimilar tokens
        remaining_indices = list(all_indices - selected_indices)
        tokens_per_pivot = (n_keep - len(selected_indices)) // self.pivot_tokens

        for pivot_idx in pivot_indices:
            if len(remaining_indices) == 0:
                break

            pivot_token = patch_tokens[0, pivot_idx, :]  # [D]
            remaining_tokens = patch_tokens[0, remaining_indices, :]  # [R, D]

            # Cosine similarity (negative for dissimilarity)
            cos_sim = F.cosine_similarity(pivot_token.unsqueeze(0), remaining_tokens, dim=-1)
            dissimilar_indices = (-cos_sim).topk(min(tokens_per_pivot, len(remaining_indices))).indices

            # Add most dissimilar tokens
            for idx in dissimilar_indices:
                selected_indices.add(remaining_indices[idx])

            # Remove selected from remaining
            remaining_indices = [idx for i, idx in enumerate(remaining_indices)
                               if i not in dissimilar_indices]

        # Fill remaining slots if needed
        while len(selected_indices) < n_keep and remaining_indices:
            selected_indices.add(remaining_indices.pop(0))

        # Convert to tensor and add 1 for cls token offset
        retained_indices = torch.tensor(sorted(list(selected_indices)), device=x.device) + 1
        return retained_indices

    def get_retained_tokens_random(self, x, n_keep):
        """Random token selection"""
        B, N_total, D = x.shape
        N_patches = N_total - 1

        if n_keep >= N_patches:
            return torch.arange(1, N_total, device=x.device)

        # Random selection from patch tokens
        patch_indices = torch.randperm(N_patches, device=x.device)[:n_keep]
        return patch_indices + 1  # +1 for cls token offset

    def get_retained_tokens_knorm(self, x, n_keep):
        """K-norm based token selection"""
        B, N_total, D = x.shape
        N_patches = N_total - 1

        if n_keep >= N_patches:
            return torch.arange(1, N_total, device=x.device)

        # Use L1 norm of patch tokens
        patch_tokens = x[:, 1:, :]  # [B, N, D]
        norms = torch.norm(patch_tokens[0], p=1, dim=-1)  # [N]
        top_indices = norms.topk(n_keep).indices
        return top_indices + 1  # +1 for cls token offset

    def prune_tokens(self, x):
        """Apply token pruning based on method"""
        if self.method == 'none' or self.ratio == 0.0:
            return x

        B, N_total, D = x.shape
        N_patches = N_total - 1  # exclude cls token
        n_keep = int(N_patches * (1 - self.ratio))
        # Apply max_num_trunction if provided (>0)
        max_keep = getattr(self, 'max_num_trunction', 0)
        if isinstance(max_keep, int) and max_keep > 0:
            n_keep = min(n_keep, max_keep)

        if self.method == 'dart':
            retained_indices = self.get_retained_tokens_dart(x, n_keep)
        elif self.method == 'random':
            retained_indices = self.get_retained_tokens_random(x, n_keep)
        elif self.method == 'knorm':
            retained_indices = self.get_retained_tokens_knorm(x, n_keep)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Keep cls token + selected patch tokens
        cls_token = x[:, 0:1, :]  # [B, 1, D]
        selected_patches = x[:, retained_indices, :]  # [B, n_keep, D]

        return torch.cat([cls_token, selected_patches], dim=1)

    def forward(self, x):
        """Forward pass with token pruning"""
        # Patch embedding
        x = self.base_model.patch_embed(x)

        # Add cls token and pos embedding
        cls_token = self.base_model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.base_model.pos_drop(x + self.base_model.pos_embed)

        # Forward through blocks
        for i, block in enumerate(self.base_model.blocks):
            x = block(x)

            # Apply pruning after specified layer
            if i == self.prune_layer and self.method != 'none':
                x = self.prune_tokens(x)

        # Final layers
        x = self.base_model.norm(x)
        x = self.base_model.head(x[:, 0])  # cls token for classification

        return x

def get_data_loaders(data_dir, img_size=224, batch_size=128, num_workers=4):
    """Create train and validation data loaders"""

    # ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), train_transform)
    val_dataset = ImageFolder(os.path.join(data_dir, 'val'), val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

def evaluate_model(model, val_loader, device):
    """Evaluate model accuracy and throughput"""
    model.eval()
    correct = 0
    total = 0

    start_time = time.time()

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    end_time = time.time()

    accuracy = 100 * correct / total
    throughput = total / (end_time - start_time)  # images/sec

    return accuracy, throughput


def train_model(model, train_loader, val_loader, device, epochs=0, lr=5e-4, weight_decay=0.05, freeze_backbone=True):
    """Optional training (fine-tuning) before pruning/eval.
    If freeze_backbone is True, only trains the classification head.
    Returns: best_state_dict (dict) or None if no training.
    """
    if epochs is None or epochs <= 0:
        return None

    # Freeze backbone if requested
    if freeze_backbone:
        for p in model.base_model.parameters():
            p.requires_grad = False
        for p in model.base_model.head.parameters():
            p.requires_grad = True
        params = model.base_model.head.parameters()
    else:
        params = model.parameters()

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            seen += bs

        scheduler.step()

        # Validation
        val_acc, val_throughput = evaluate_model(model, val_loader, device)
        avg_loss = running_loss / max(1, seen)
        print(f"[Epoch {epoch}/{epochs}] train_loss={avg_loss:.4f} val_acc={val_acc:.2f}% thpt={val_throughput:.2f}")

        if val_acc > best_acc:
            best_acc = val_acc
            # Keep a CPU copy of the best weights
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        # Load best back into model
        model.load_state_dict(best_state, strict=True)
    return best_state


def save_checkpoint(model, accuracy, throughput, args, checkpoint_dir):
    """Save model checkpoint and results"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create checkpoint filename
    filename = f"{args.model}_{args.method}_ratio{args.ratio}_layer{args.prune_layer}_seed{args.seed}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    # Save checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'accuracy': accuracy,
        'throughput': throughput,
        'model_name': args.model,
        'method': args.method,
        'ratio': args.ratio,
        'prune_layer': args.prune_layer,
        'seed': args.seed
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    return checkpoint_path

def main():
    parser = argparse.ArgumentParser(description='ImageNette Token Pruning Experiments')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to ImageNette dataset')
    parser.add_argument('--model', type=str, default='vit_base_patch16_224',
                       choices=['vit_base_patch16_224', 'vit_small_patch16_224'],
                       help='Model architecture')
    parser.add_argument('--method', type=str, default='none',
                       choices=['none', 'dart', 'random', 'knorm'],
                       help='Token pruning method')
    parser.add_argument('--reduction-ratio', type=float, default=None,
                       help='Pruning reduction ratio (0.0-1.0); if None, fallback to --ratio')
    parser.add_argument('--ratio', type=float, default=0.0,
                       help='Deprecated: use --reduction-ratio')
    parser.add_argument('--prune-layer', type=int, default=6,
                       help='Layer to apply pruning after')
    parser.add_argument('--pivot-image-token', type=int, default=8,
                       help='Number of image pivot tokens for DART method')
    parser.add_argument('--pivot-text-token', type=int, default=0,
                       help='Number of text pivot tokens for DART method (0 for classification)')
    parser.add_argument('--max-num-trunction', type=int, default=0,
                       help='Max number of tokens to keep after pruning (0 disables limit)')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--load-checkpoint', type=str, default='',
                       help='Path to a checkpoint to load weights before eval')

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.results_dir, exist_ok=True)

    # Prefer reduction_ratio if provided
    rr = args.reduction_ratio if args.reduction_ratio is not None else args.ratio

    print(f"Creating model: {args.model} with method: {args.method}, reduction_ratio: {rr}")
    model = TokenPruningViT(
        model_name=args.model,
        method=args.method,
        ratio=rr,
        prune_layer=args.prune_layer,
        pivot_tokens=args.pivot_image_token + args.pivot_text_token,
        device=args.device
    ).to(args.device)
    # Attach max_num_trunction to model for pruning cap
    setattr(model, 'max_num_trunction', args.max_num_trunction)

    print(f"Loading data from: {args.data_dir}")
    train_loader, val_loader = get_data_loaders(
        args.data_dir, args.img_size, args.batch_size, args.num_workers
    )

    train_epochs = int(os.environ.get('TRAIN_EPOCHS', '0'))
    if train_epochs > 0:
        print(f"Training for {train_epochs} epoch(s) before pruning/eval ...")
        train_model(
            model, train_loader, val_loader, args.device,
            epochs=train_epochs, lr=5e-4, weight_decay=0.05, freeze_backbone=True
        )

    print("Evaluating model...")
    accuracy, throughput = evaluate_model(model, val_loader, args.device)

    print(f"Results - Accuracy: {accuracy:.2f}%, Throughput: {throughput:.2f} images/sec")

    checkpoint_path = save_checkpoint(model, accuracy, throughput, args, args.checkpoint_dir)

    results_file = os.path.join(args.results_dir, 'imagenette_results.csv')
    file_exists = os.path.exists(results_file)

    with open(results_file, 'a', newline='') as csvfile:
        fieldnames = ['model', 'method', 'ratio', 'prune_layer', 'seed',
                     'accuracy', 'throughput', 'checkpoint_path', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'model': args.model,
            'method': args.method,
            'ratio': args.ratio,
            'prune_layer': args.prune_layer,
            'seed': args.seed,
            'accuracy': accuracy,
            'throughput': throughput,
            'checkpoint_path': checkpoint_path,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })

    print(f"Results saved to: {results_file}")

if __name__ == '__main__':
    main()

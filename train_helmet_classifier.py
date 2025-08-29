import os
import json
import argparse
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from helmet_classifier import HelmetClassifier, create_model_config

try:
    from timm.utils import ModelEmaV2
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False


class HelmetDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int],
                 transform: Optional[transforms.Compose] = None,
                 input_size: Tuple[int, int] = (224, 224)):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        self.input_size = input_size
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', self.input_size, (0,0,0))
        img = self.transform(img)
        return img, label


def create_data_transforms(input_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf


def prepare_dataset_from_directory(data_dir: str, test_size: float = 0.2, val_size: float = 0.2):
    image_paths, labels = [], []
    class_to_idx = {'no_helmet': 0, 'wearing_helmet': 1}
    for cls, idx in class_to_idx.items():
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for name in os.listdir(cls_dir):
            if name.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
                image_paths.append(os.path.join(cls_dir,name))
                labels.append(idx)

    print(f"Found {len(image_paths)} images")
    if len(set(labels)) > 1:
        dist = dict(zip(*np.unique(labels, return_counts=True)))
        print(f"Class distribution: {dist}")

    train_p, temp_p, train_y, temp_y = train_test_split(
        image_paths, labels, test_size=(test_size+val_size),
        stratify=labels if len(set(labels))>1 else None, random_state=42
    )
    val_p, test_p, val_y, test_y = train_test_split(
        temp_p, temp_y, test_size=(test_size/(test_size+val_size)),
        stratify=temp_y if len(set(temp_y))>1 else None, random_state=42
    )
    return train_p, val_p, test_p, train_y, val_y, test_y


def train_epoch(model, loader, criterion, optimizer, device, scaler=None, ema: Optional[ModelEmaV2]=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="Training")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(imgs)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if ema is not None:
            ema.update(model)

        running_loss += loss.item()
        _, pred = torch.max(logits, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

        pbar.set_postfix({
            'Loss': f'{running_loss/(pbar.n+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })

    return running_loss/len(loader), correct/total


@torch.no_grad()
def validate_epoch(model, loader, criterion, device, use_ema_model=False, ema: Optional[ModelEmaV2]=None):
    model_to_eval = ema.module if (use_ema_model and ema is not None) else model
    model_to_eval.eval()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="Validation")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model_to_eval(imgs)
        loss = criterion(logits, labels)

        running_loss += loss.item()
        _, pred = torch.max(logits, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

        pbar.set_postfix({
            'Loss': f'{running_loss/(pbar.n+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })

    return running_loss/len(loader), correct/total


def plot_training_history(history: Dict[str, List], save_path: str = None):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(12,10))
    ax1.plot(history['train_loss']); ax1.plot(history['val_loss']); ax1.set_title('Loss'); ax1.grid(True)
    ax2.plot(history['train_acc']); ax2.plot(history['val_acc']); ax2.set_title('Accuracy'); ax2.grid(True)
    ax3.plot(history['lr']); ax3.set_title('Learning Rate'); ax3.grid(True)
    ax4_t = ax4.twinx()
    ax4.plot(history['train_loss'],'b-'); ax4.plot(history['val_loss'],'b--')
    ax4_t.plot(history['train_acc'],'r-'); ax4_t.plot(history['val_acc'],'r--')
    ax4.set_title('Loss & Acc'); ax4.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plots saved to: {save_path}")
    plt.show()


def main():
    ap = argparse.ArgumentParser(description='Train Helmet Classification Model')
    ap.add_argument('--data_dir', type=str, required=True)
    ap.add_argument('--save_dir', type=str, default='./checkpoints')
    ap.add_argument('--backbone', type=str, default='convnext_tiny',
                    help="torchvision: resnet18/resnet34/efficientnet_b0; timm: convnext_tiny, efficientnetv2_s, resnet50d, swin_tiny_patch4_window7_224, ...")
    ap.add_argument('--epochs', type=int, default=70)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--learning_rate', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=5e-2)
    ap.add_argument('--input_size', type=int, default=256)
    ap.add_argument('--warmup_epochs', type=int, default=5)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--label_smoothing', type=float, default=0.05)
    ap.add_argument('--use_amp', action='store_true', default=True)
    ap.add_argument('--use_ema', action='store_true', default=True,
                    help='Use EMA if timm installed')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    tr_p, va_p, te_p, tr_y, va_y, te_y = prepare_dataset_from_directory(args.data_dir)
    train_tf, val_tf = create_data_transforms(args.input_size)
    train_ds = HelmetDataset(tr_p, tr_y, train_tf, (args.input_size,args.input_size))
    val_ds = HelmetDataset(va_p, va_y, val_tf, (args.input_size,args.input_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = HelmetClassifier(backbone=args.backbone, num_classes=2, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    total_epochs = args.epochs
    warmup_epochs = min(args.warmup_epochs, max(1, total_epochs//10))
    main_epochs = total_epochs - warmup_epochs
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=main_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    ema = None
    use_ema_flag = args.use_ema and _HAS_TIMM
    if use_ema_flag:
        ema = ModelEmaV2(model, decay=0.9999)
        print("EMA enabled.")
    else:
        if args.use_ema and not _HAS_TIMM:
            print("EMA requested but timm not installed; skipping EMA.")

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_val = 0.0

    print(f"Training on {device}")
    print(f"Epochs={args.epochs}  AMP={args.use_amp}  EMA={use_ema_flag}  Warmup={warmup_epochs}")

    for epoch in range(total_epochs):
        print(f"\nEpoch {epoch+1}/{total_epochs}")
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, ema)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, use_ema_model=True, ema=ema)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")
        print(f"LR: {current_lr:.6f}")

        if val_acc > best_val:
            best_val = val_acc
            best_path = os.path.join(args.save_dir, 'best_model.pth')
            to_save = ema.module if ema is not None else model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'model_config': create_model_config(backbone=args.backbone, num_classes=2),
            }, best_path)
            print(f"New best saved: {best_path} (val_acc={val_acc:.4f})")

    plot_path = os.path.join(args.save_dir, 'training_curves.png')
    plot_training_history(history, plot_path)
    hist_path = os.path.join(args.save_dir, 'training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {hist_path}")
    print(f"Best val acc: {best_val:.4f}")


if __name__ == "__main__":
    main()

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DATA_ROOT     = r"C:\Users\ymodi\Desktop\Coding\Osteoporosis\archive"
IMG_SIZE      = 224
BATCH_SIZE    = 16      
EPOCHS        = 40
LR            = 3e-4
FREEZE_EPOCHS = 5
THRESHOLD     = 0.50    
PATIENCE      = 8
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED          = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}\n")

class_map = {
    os.path.join(DATA_ROOT, "normal", "normal"):                 0,
    os.path.join(DATA_ROOT, "osteoporosis", "osteoporosis"):     1,
}

all_paths, all_labels = [], []
for folder_path, label in class_map.items():
    if not os.path.exists(folder_path):
        print(f"WARNING: folder not found → {folder_path}")
        continue
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            all_paths.append(os.path.join(folder_path, fname))
            all_labels.append(label)

print(f"Total images       : {len(all_paths)}")
print(f"  Class 0 (Normal) : {all_labels.count(0)}")
print(f"  Class 1 (Disease): {all_labels.count(1)}\n")

idx = list(range(len(all_paths)))
train_idx, temp_idx = train_test_split(
    idx, test_size=0.30, stratify=all_labels, random_state=SEED
)
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    stratify=[all_labels[i] for i in temp_idx],
    random_state=SEED
)
print(f"Split — Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}\n")

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=8),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

class XrayDataset(Dataset):
    def __init__(self, indices, paths, labels, transform):
        self.paths     = [paths[i]  for i in indices]
        self.labels    = [labels[i] for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), torch.tensor(self.labels[idx], dtype=torch.float32)


train_ds = XrayDataset(train_idx, all_paths, all_labels, train_transform)
val_ds   = XrayDataset(val_idx,   all_paths, all_labels, val_test_transform)
test_ds  = XrayDataset(test_idx,  all_paths, all_labels, val_test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, 128),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(128, 1),
)

model     = model.to(DEVICE)
criterion = nn.BCEWithLogitsLoss()   
print(f"Model     : EfficientNetB0")
print(f"Total params    : {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

def run_epoch(loader, train=True, optimizer=None):
    model.train() if train else model.eval()
    total_loss, all_probs, all_true = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.unsqueeze(1).to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            all_probs.extend(probs)
            all_true.extend(labels.cpu().numpy().flatten())
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    try:
        auc = roc_auc_score(all_true, all_probs)
    except Exception:
        auc = 0.0
    return avg_loss, auc, np.array(all_probs), np.array(all_true)

print("=" * 55)
print("PHASE 1 — Training head only (backbone frozen)")
print("=" * 55)
optimizer    = optim.AdamW(model.classifier.parameters(), lr=LR, weight_decay=1e-4)
best_val_auc = 0.0
best_weights = None
no_improve   = 0

for epoch in range(1, FREEZE_EPOCHS + 1):
    tr_loss, tr_auc, _, _ = run_epoch(train_loader, train=True,  optimizer=optimizer)
    vl_loss, vl_auc, _, _ = run_epoch(val_loader,   train=False)
    marker = " ← best" if vl_auc > best_val_auc else ""
    print(f"Epoch {epoch:02d}/{FREEZE_EPOCHS} | "
          f"Train Loss: {tr_loss:.4f} AUC: {tr_auc:.4f} | "
          f"Val Loss: {vl_loss:.4f} AUC: {vl_auc:.4f}{marker}")
    if vl_auc > best_val_auc:
        best_val_auc = vl_auc
        best_weights = {k: v.clone() for k, v in model.state_dict().items()}

print(f"\n{'='*55}")
print("PHASE 2 — Fine-tuning full network (backbone unfrozen)")
print("=" * 55)

for param in model.parameters():
    param.requires_grad = True

optimizer = optim.AdamW(model.parameters(), lr=LR * 0.05, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

for epoch in range(FREEZE_EPOCHS + 1, EPOCHS + 1):
    tr_loss, tr_auc, _, _ = run_epoch(train_loader, train=True,  optimizer=optimizer)
    vl_loss, vl_auc, _, _ = run_epoch(val_loader,   train=False)
    scheduler.step()

    marker = " ← best" if vl_auc > best_val_auc else ""
    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Train Loss: {tr_loss:.4f} AUC: {tr_auc:.4f} | "
          f"Val Loss: {vl_loss:.4f} AUC: {vl_auc:.4f}{marker}")

    if vl_auc > best_val_auc:
        best_val_auc = vl_auc
        best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve   = 0
    else:
        no_improve  += 1

    if no_improve >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} — no Val AUC improvement for {PATIENCE} epochs.")
        break

model.load_state_dict(best_weights)
print(f"\nBest Val AUC: {best_val_auc:.4f} — best weights restored\n")

_, _, all_probs, all_true = run_epoch(test_loader, train=False)
all_preds = (all_probs > THRESHOLD).astype(int)

print("=" * 55)
print("CNN IMAGE BRANCH — EVALUATION")
print("=" * 55)
print(f"Test Accuracy : {accuracy_score(all_true, all_preds) * 100:.2f}%")
print(f"ROC-AUC       : {roc_auc_score(all_true, all_probs):.4f}")
print(f"\nClassification Report (threshold={THRESHOLD}):\n")
print(classification_report(all_true, all_preds,
                             target_names=["No Disease", "Disease"]))

fusion_output = pd.DataFrame({
    "prob_image_negative": 1 - all_probs,
    "prob_image_positive": all_probs,
    "pred_image":          all_preds,
    "true_label":          all_true.astype(int),
})

print("\n" + "=" * 55)
print("FUSION LAYER — IMAGE BRANCH SCORES (first 10 rows)")
print("=" * 55)
print(fusion_output.head(10).to_string())

print("\n" + "=" * 55)
print("FUSION LAYER — SCORE DISTRIBUTION SUMMARY")
print("=" * 55)
print(fusion_output[["prob_image_negative", "prob_image_positive"]].describe().round(4))

fusion_output.to_csv("cnn_image_branch_scores.csv", index=True)
torch.save(model.state_dict(), "efficientnetb0_osteoporosis.pth")
print("\nSaved → cnn_image_branch_scores.csv")
print("Saved → efficientnetb0_osteoporosis.pth")
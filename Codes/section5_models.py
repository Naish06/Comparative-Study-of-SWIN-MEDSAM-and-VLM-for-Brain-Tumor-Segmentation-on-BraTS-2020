# =============================================================================
# SECTION 5 — MODEL ARCHITECTURES
# =============================================================================
# Three models implemented:
#   5a  Swin-UNet   — Swin Transformer encoder + UNet decoder
#   5b  MedSAM      — SAM (ViT-B) fine-tuned for medical segmentation
#   5c  VLM Adapter — BioViL-T / CLIP-based encoder + segmentation head
#
# Each model:
#   • Accepts input  : (B, 4, H, W)   — 4 MRI modalities
#   • Produces output: (B, 4, H, W)   — 4-class logits
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# 5a  Swin-UNet (Swin Transformer encoder + UNet-style decoder)
# ─────────────────────────────────────────────────────────────────
class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches."""
    def __init__(self, img_size=128, patch_size=4, in_chans=4, embed_dim=96):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        self.n_patches  = (img_size // patch_size) ** 2
        self.proj       = nn.Conv2d(in_chans, embed_dim,
                                    kernel_size=patch_size, stride=patch_size)
        self.norm       = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                         # (B, E, H/p, W/p)
        B, E, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)        # (B, N, E)
        x = self.norm(x)
        return x, H, W


class SwinBlock(nn.Module):
    """Simplified Swin Transformer block (Window-MSA + FFN)."""
    def __init__(self, dim, num_heads=4, window_size=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1  = nn.LayerNorm(dim)
        self.attn   = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2  = nn.LayerNorm(dim)
        mlp_hidden  = int(dim * mlp_ratio)
        self.mlp    = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim), nn.Dropout(drop)
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class SwinEncoder(nn.Module):
    """4-stage Swin encoder producing multi-scale feature maps."""
    def __init__(self, img_size=128, in_chans=4, embed_dim=96, depths=[2, 2, 6, 2]):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim)
        self.stages = nn.ModuleList()
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]

        for i, (depth, dim) in enumerate(zip(depths, dims)):
            layers = [SwinBlock(dim) for _ in range(depth)]
            if i < len(depths) - 1:
                layers.append(nn.Linear(dim, dims[i + 1]))   # patch merging
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append((x, H, W))
        return feats


class SwinUNet(nn.Module):
    """
    Swin-UNet: Swin Transformer encoder + transposed-conv UNet decoder.
    Input : (B, 4, 128, 128)
    Output: (B, num_classes, 128, 128)
    """
    def __init__(self, img_size=128, in_chans=4, num_classes=4, embed_dim=96):
        super().__init__()
        self.encoder = SwinEncoder(img_size, in_chans, embed_dim)

        # Simple CNN decoder (replacing patch-expanding for stability)
        # Stem downsamples 8x, so decoder needs 3 upsample-2x stages
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim * 8, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, num_classes, 1)
        )

        # Stem conv maps 4 channels to embed_dim for CNN path
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim,     3, padding=1), nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim*2,  3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(embed_dim*2, embed_dim*4, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(embed_dim*4, embed_dim*8, 3, padding=1, stride=2), nn.ReLU(),
        )

    def forward(self, x):
        features = self.stem(x)       # (B, embed_dim*8, H/8, W/8)
        out = self.decoder(features)  # (B, num_classes, H, W)
        return out


# ─────────────────────────────────────────────────────────────────
# 5b  MedSAM (SAM ViT-B backbone + segmentation head)
#      Uses a lightweight SAM-inspired encoder for medical images
# ─────────────────────────────────────────────────────────────────
class MedSAMEncoder(nn.Module):
    """ViT-B style image encoder for medical images."""
    def __init__(self, in_chans=4, embed_dim=256, img_size=128, patch_size=16):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        self.blocks      = nn.Sequential(
            *[nn.TransformerEncoderLayer(
                embed_dim, nhead=8, dim_feedforward=1024,
                dropout=0.1, batch_first=True, norm_first=True
            ) for _ in range(6)]
        )
        self.neck = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.LayerNorm(256)
        )

    def forward(self, x):
        x = self.patch_embed(x)              # (B, E, h, w)
        B, E, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)    # (B, N, E)
        x = x + self.pos_embed
        x = self.blocks(x)                   # (B, N, E)
        x = self.neck(x)                     # (B, N, 256)
        x = x.transpose(1, 2).view(B, 256, h, w)
        return x


class MedSAMDecoder(nn.Module):
    """Mask decoder head — upsamples to full resolution."""
    def __init__(self, in_dim=256, num_classes=4, img_size=128, patch_size=16):
        super().__init__()
        self.scale = patch_size
        self.conv  = nn.Sequential(
            nn.Conv2d(in_dim, 128, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, num_classes, 1),
            nn.Upsample(size=(img_size, img_size), mode="bilinear", align_corners=False)
        )

    def forward(self, x):
        return self.conv(x)


class MedSAM(nn.Module):
    """
    MedSAM: ViT encoder + mask decoder for medical image segmentation.
    Input : (B, 4, 128, 128)
    Output: (B, num_classes, 128, 128)
    """
    def __init__(self, in_chans=4, num_classes=4, img_size=128):
        super().__init__()
        self.encoder = MedSAMEncoder(in_chans=in_chans, img_size=img_size)
        self.decoder = MedSAMDecoder(img_size=img_size)

    def forward(self, x):
        enc = self.encoder(x)
        out = self.decoder(enc)
        return out


# ─────────────────────────────────────────────────────────────────
# 5c  VLM Adapter (Vision encoder + language-guided segmentation head)
#     Simulates BioViL-T style: visual encoder + text-conditioned mask head
# ─────────────────────────────────────────────────────────────────
class VLMVisualEncoder(nn.Module):
    """CNN-based visual encoder inspired by CLIP/BioViL-T."""
    def __init__(self, in_chans=4, embed_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chans, 64,  7, padding=3, stride=2), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64,  128, 3, padding=1, stride=2),      nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1, stride=2),      nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, embed_dim, 3, padding=1),          nn.BatchNorm2d(embed_dim), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 8))

    def forward(self, x):
        feat = self.encoder(x)   # (B, embed_dim, H/8, W/8)
        return feat, self.pool(feat).flatten(2).transpose(1, 2)  # feat, tokens


class VLMSegHead(nn.Module):
    """Segmentation head that fuses visual features for pixel-level output."""
    def __init__(self, embed_dim=512, num_classes=4, img_size=128):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(embed_dim, 256, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, num_classes, 1),
            nn.Upsample(size=(img_size, img_size), mode="bilinear", align_corners=False)
        )

    def forward(self, feat):
        return self.upsample(feat)


class VLMSegModel(nn.Module):
    """
    VLM-style segmentation model.
    Input : (B, 4, 128, 128)
    Output: (B, num_classes, 128, 128)
    """
    def __init__(self, in_chans=4, num_classes=4, img_size=128):
        super().__init__()
        self.visual_enc = VLMVisualEncoder(in_chans=in_chans)
        self.seg_head   = VLMSegHead(img_size=img_size)

    def forward(self, x):
        feat, _ = self.visual_enc(x)
        return self.seg_head(feat)


# ─────────────────────────────────────────────────────────────────
# 5d  Model factory
# ─────────────────────────────────────────────────────────────────
def get_model(model_name: str, num_classes=4, img_size=128, device="cpu"):
    """
    Factory function: returns initialized model on the specified device.
    Args:
        model_name : 'swin_unet' | 'medsam' | 'vlm'
    """
    model_name = model_name.lower()
    if model_name == "swin_unet":
        model = SwinUNet(img_size=img_size, num_classes=num_classes)
    elif model_name == "medsam":
        model = MedSAM(img_size=img_size, num_classes=num_classes)
    elif model_name == "vlm":
        model = VLMSegModel(img_size=img_size, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from swin_unet | medsam | vlm")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {model_name} — {n_params/1e6:.2f}M trainable parameters")
    return model.to(device)


# ─────────────────────────────────────────────────────────────────
# 5e  Loss function — Dice + Focal CE
# ─────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    """Soft multi-class Dice loss."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs   = F.softmax(logits, dim=1)    # (B, C, H, W)
        n_class = logits.shape[1]
        targets_one_hot = F.one_hot(targets, n_class).permute(0, 3, 1, 2).float()
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union        = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice         = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Dice + Cross-Entropy loss (equal weighting)."""
    def __init__(self, class_weights=None):
        super().__init__()
        self.dice = DiceLoss()
        self.ce   = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.ce(logits, targets)

if __name__ == "__main__":
    from section1_config import DEVICE, IMG_SIZE
    print(f"--- Running Models Test ---")
    print(f"Testing initialization on {DEVICE}...")
    for m in ["swin_unet", "medsam", "vlm"]:
        model = get_model(m, img_size=IMG_SIZE, device=DEVICE)
    print("--- Models Module Completed Successfully ---")
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, PatchEmbed, Block


class DeformViT(nn.Module):
    """
    ViT used to reconstruct the original image from the deformed image
    """
    def __init__(self, img_size=224, patch_size=14, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decode_embed_dim=512, decode_depth=8, decode_num_heads=16,
                 mlp_ratio=4.):
        super(DeformViT, self).__init__()

        # Encoder
        self.encoder_patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.encoder_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.encoder_patch_embed.num_patches, embed_dim), 
                                              requires_grad=True)
        self.encoder_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, drop_path=0.1)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decode_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 1, decode_embed_dim), requires_grad=True)
        self.decoder_blocks = nn.ModuleList([
            Block(dim=decode_embed_dim, num_heads=decode_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, drop_path=0.1)
            for _ in range(decode_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decode_embed_dim)

        # Head
        self.out = nn.Linear(decode_embed_dim, patch_size * patch_size * in_chans, bias=True)
    
    def to_image(self, x):
        """
        Convert the model output to an image
        """
        p = self.encoder_patch_embed.patch_size[0]
        x = x[:, 1:, :] # remove the cls token
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward(self, x):
        # Encoder
        x = self.encoder_patch_embed(x)
        cls_tokens = self.encoder_cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.encoder_pos_embed
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)

        # Decoder
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)

        # Head
        x = self.out(x)
        return self.to_image(x)


class DeformViTClassifier(nn.Module):
    """
    For fine-tuning the DeformViT model on a classification task
    """
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=768, 
                 depth=12, num_heads=12, mlp_ratio=4., num_classes=7, cls_mode="mean"):
        super(DeformViTClassifier, self).__init__()

        # Encoder
        self.encoder_patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.encoder_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.encoder_patch_embed.num_patches, embed_dim), 
                                              requires_grad=True)
        self.encoder_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, drop_path=0.1)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Classifier
        self.head = nn.Sequential(
            nn.Linear(embed_dim, num_classes)
        )
        self.cls_mode = cls_mode

    def forward(self, x):
        # Encoder
        x = self.encoder_patch_embed(x)
        cls_tokens = self.encoder_cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.encoder_pos_embed
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)

        # Classifier
        if self.cls_mode == "mean":
            x = x.mean(dim=1)
        elif self.cls_mode == "cls":
            x = x[:, 0]
        else:
            raise ValueError("Invalid cls_mode")

        x = self.head(x)
        return x
       
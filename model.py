import sys
import os
import torch
import torch.nn as nn


current_dir = os.getcwd()
eegpt_path = os.path.join(current_dir, 'EEGPT')
if eegpt_path not in sys.path:
    sys.path.append(eegpt_path)

try:
    from downstream.Modules.models.EEGPT_mcae import EEGTransformer
except ImportError:
    
    from downstream.Modules.models.EEGPT_mcae import EEGTransformer

class EEGPT_SeizureDetector(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()

        self.backbone = EEGTransformer(
            img_size=(58, 1024), 
            patch_size=64,
            embed_dim=512,       
            depth=12,            
            num_heads=8,         
        )

        if os.path.exists(pretrained_path):
            ckpt = torch.load(pretrained_path, map_location="cpu")
          
            if "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt
                
            msg = self.backbone.load_state_dict(state_dict, strict=False)
            print(f" Weights Loaded. Missing keys: {len(msg.missing_keys)} (Expected for heads)")
        else:
            print(f" Warning: Checkpoint not found at {pretrained_path}. Using random weights.")

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Input x: (Batch, 58, 1024)
        """
        
        features = self.backbone(x) 
        features = features.squeeze(2) 
        logits = self.head(features) 
        
        return logits.squeeze(-1) 
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

current_dir = os.getcwd()
eegpt_path = os.path.join(current_dir, 'EEGPT')
if eegpt_path not in sys.path:
    sys.path.append(eegpt_path)

from dataset import SeizureDataset 
from model import EEGPT_SeizureDetector

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

POS_WEIGHT = torch.tensor([15.0]).to(DEVICE) 

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x) 
        
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} Loss: {loss.item():.4f}")
            
    return total_loss / len(loader)

def evaluate(model, loader):
    """
    Calculates Precision, Recall, and F1 for the token-level predictions.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
           
            logits = model(x)
            
            probs = torch.sigmoid(logits)
            
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())
    
    #Metrics
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0) # Sensitivity
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    return precision, recall, f1

def main():
    train_dataset = SeizureDataset(data_dir="./processed_data/")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Training on {len(train_dataset)} windows...")

    PRETRAINED_PATH = "checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt" 
    model = EEGPT_SeizureDetector(pretrained_path=PRETRAINED_PATH)
    model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Average Training Loss: {avg_loss:.4f}")
        prec, rec, f1 = evaluate(model, train_loader)
        print(f"Metrics -> Precision: {prec:.4f}, Recall (Sensitivity): {rec:.4f}, F1: {f1:.4f}")
        
        save_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint: {save_path}")

if __name__ == "__main__":
    main()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

from utils.losses import SoftCrossEntropyLoss
from utils.dataset import load_generations
from models import MLP, MLPWithPooling, TransformerClassifier, Conv1DNet
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import os
import argparse
from tqdm.auto import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument('--dataset_name', type=str, default='helm')
    parser.add_argument('--topic', type=str, default='falcon40b')
    parser.add_argument('--classifier', type=str, default='mlp')
    parser.add_argument('--prompt', type=str, default='cot')
    parser.add_argument('--pseudo_label', type=str, default='verb', choices=['verb', 'entropy', 'perp'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--phi', type=float, default=1)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--layer', default='1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--save_classifier', action='store_true')
    args = parser.parse_args()
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Load activations and labels
    X_arr, soft_labels, true_labels = load_generations(args)
    labels = np.stack((soft_labels, true_labels), axis=1)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_arr, labels, test_size=0.2, random_state=42, stratify=labels[:,1]
    )
    
    if args.normalize:
        std = np.std(X_arr, axis=0, keepdims=True, dtype=np.float32)
        mean = np.mean(X_arr, axis=0, keepdims=True)
        X_arr = (X_arr - mean)/(std + np.full_like(std, 1e-6))
        
    # Convert arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Define DataLoader for batch processing
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize classifier, loss function, and optimizer
    if args.classifier == 'mlp':
        classifier = MLP(in_dim=model.config.hidden_size)
    elif args.classifier == 'mlp_pool':
        classifier = MLPWithPooling(in_dim=model.config.hidden_size)
    elif args.classifier == 'transformer':
        classifier = TransformerClassifier(in_dim=model.config.hidden_size)
    elif args.classifier == 'conv1d':
        classifier = Conv1DNet(in_dim=model.config.hidden_size)
    criterion = SoftCrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-2, weight_decay=1e-5)

    # Training loop
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc, wait, patience = 0, 0, 5
    
    for epoch in tqdm(range(args.num_epochs)):
        classifier.train()
        correct = 0
        train_loss = 0
        torch.manual_seed(args.seed)
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = classifier(inputs)[0] 
            preds = outputs.argmax(dim=1)
            pred_probs = torch.softmax(outputs, dim=1) 
            # Bootstrapping
            bootstrap_targets = args.beta * torch.stack([torch.abs(1-targets[:,0]), targets[:,0]], dim=1) + (1 - args.beta) * pred_probs
            
            # Compute loss
            ce_loss = criterion(pred_probs, bootstrap_targets)
            rce_loss = criterion(bootstrap_targets, pred_probs)
            loss = ce_loss + args.phi*rce_loss
            loss.backward()    
            optimizer.step()
            correct += (preds == targets[:,1]).sum().item()
        train_accuracy = correct / len(train_dataset)
        train_losses.append(loss)
        train_accs.append(train_accuracy)

        # Validation loop
        classifier.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = classifier(inputs)[0]
                preds = outputs.argmax(dim=1)
                correct += (preds == targets[:,1]).sum().item()
            
        # Calculate and print average validation accuracy
        val_accuracy = correct / len(val_dataset)
        val_accs.append(val_accuracy)
        
        # Check for improvement in validation accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            wait = 0 
            tqdm.write("Validation accuracy improved. Resetting patience counter.")
            tqdm.write(f"Epoch {epoch+1}/{args.num_epochs}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        else:
            wait += 1
            tqdm.write(f"No improvement at Epoch {epoch+1}. Patience counter: {wait}/{patience}")
            
        # Early stopping condition
        if wait >= patience:
            print("Early stopping triggered. Stopping training.")
            break

    if args.save_classifier:
        save_path = f'results/{args.dataset_name}/{args.prompt}/{args.topic}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(classifier.state_dict(), os.path.join(save_path, f'{args.model.split('/')[-1]}_{args.classifier}_classifier.pth'))
    

import os
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from sklearn.manifold import TSNE

def plot_probe_accs(args, all_accs_np):
    # Plotting the array as a color map
    
    if args.activation_type=='attention':
        # sorted_head_accs = np.sort(all_head_accs_np, axis=1)[:,::-1]
        sorted_head_accs = all_accs_np
        plt.imshow(sorted_head_accs, cmap='viridis', interpolation='nearest',  origin='lower')
        plt.xlabel('Head')
    elif args.activation_type=='embedding':
        sorted_head_accs = np.stack((all_accs_np, all_accs_np))
        sorted_head_accs = sorted_head_accs.transpose()
        plt.imshow(sorted_head_accs, cmap='viridis', interpolation='nearest',  origin='lower')
    
    # Remove gridline from x-axis
    plt.xticks([])
    plt.ylabel('Layer')
    plt.colorbar(label='Accuracy')  
    plt.title(f"{args.model} {args.activation_type}")
    plt.tight_layout()
    save_path = f'results/{args.dataset_name}/prompt'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f'{args.model.split('/')[-1]}_{args.activation_type}_probe_accs.png'), bbox_inches='tight')

def plot_train(args, train_accs, val_accs):
    
    x = np.arange(args.num_epochs)

    # Plot the first line on the left y-axis
    plt.plot(x, val_accs, label='val')
    plt.plot(x, train_accs, label='train')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    save_path = f'results/{args.dataset_name}'
    if args.indiv_data:
        save_path = os.path.join(save_path, args.indiv_data)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f'{args.model.split('/')[-1]}_{args.activation_type}_{args.classifier}_classifier.png'))
    
    
def softlabel_hist(args, soft_labels, labels):
    
    soft_labels = (soft_labels - np.min(soft_labels))/(np.max(soft_labels) - np.min(soft_labels))
    ppl_false = soft_labels[labels==0]
    ppl_true = soft_labels[labels==1]

    plt.hist(ppl_false, bins=30, alpha=0.7, label='False', color='blue')
    plt.hist(ppl_true, bins=30, alpha=0.7, label='True', color='orange')
    plt.title('Verbalised Confidence of TrueFalse-Animals')
    plt.legend()
    plt.savefig(f'hist_verb_{args.dataset_name}.png')

def plot_tsne(classifier, train_loader, y_train, val_loader, y_val, perplexity=10, learning_rate=200, random_state=42):
    classifier_embs = []
    with torch.no_grad():
        for inputs, _ in train_loader:
            emb = classifier(inputs)[1]
            classifier_embs.append(emb)
        for inputs, _ in val_loader:
            emb = classifier(inputs)[1]
            classifier_embs.append(emb)
    classifier_embs = torch.cat(classifier_embs, dim=0)
    y = np.concatenate((y_train, y_val),axis=0)
    classifier_embs = classifier_embs.detach().cpu().numpy()
    
    # Compute t-SNE embedding
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, max_iter=1000, verbose=1, random_state=random_state)
    x_embedded = tsne.fit_transform(classifier_embs)

    # Plot the t-SNE results
    plt.figure(figsize=(8, 6))
    cmap = colors.ListedColormap(['orange', 'blue'])
    scatter = plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=y, cmap=cmap, alpha=0.7)
    plt.title('SAPLAMA', y=-.2, fontsize=28)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)    
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'tsne.png', dpi=1000)

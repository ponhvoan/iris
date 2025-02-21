import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim=4096):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        out_linear = self.fc4(out)
        return out_linear, out
    
class MLPWithPooling(nn.Module):
    def __init__(self, in_dim=4096, embed_dim=128, hidden_dim=64, num_vectors=32):
        super().__init__()
        # Shared embedding layer
        self.embed = nn.Linear(in_dim, embed_dim)
        # Aggregation -> none-learnable pooling in this simple example
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        embedded = F.relu(self.embed(x))  # (batch_size, 32, embed_dim)
        pooled = torch.mean(embedded, dim=1)  
        out = F.relu(self.fc1(pooled))
        out_linear = self.fc2(out)
        return out_linear, out


class Conv1DNet(nn.Module):
    def __init__(self, in_dim=4096, embed_dim=256, num_conv_filters=128, kernel_size=3):
        super().__init__()
        # Linear layer to project -> embed_dim
        self.embedding = nn.Linear(in_dim, embed_dim)

        # Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=embed_dim, 
            out_channels=num_conv_filters, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2  # to preserve length
        )
        self.conv2 = nn.Conv1d(
            in_channels=num_conv_filters, 
            out_channels=num_conv_filters, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2
        )
        # Final fully connected layer
        self.fc = nn.Linear(num_conv_filters, 2)

    def forward(self, x):
        """
        x shape: (batch_size, 32, 4096)
        """
        batch_size = x.size(0)
        x = self.embedding(x)  # -> (batch_size, 32, embed_dim)
        x = x.permute(0, 2, 1)  # -> (batch_size, embed_dim, 32)

        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  
        logits = self.fc(x)  

        return logits, x

class TransformerClassifier(nn.Module):
    def __init__(self, in_dim=4096, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        # Linear to embed each vector
        self.input_proj = nn.Linear(in_dim, embed_dim)

        # A stack of transformer encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim*4,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Final FC layer
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, x):
        x = x.permute(1, 0, 2) 
        x = torch.relu(self.input_proj(x))  # (32, batch_size, embed_dim)
        encoded = self.transformer_encoder(x)
        encoded_mean = encoded.mean(dim=0)
        logits = self.fc(encoded_mean)
        return logits, encoded_mean

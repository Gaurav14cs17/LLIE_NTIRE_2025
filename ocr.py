import torch
import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

class AttentionOCR(nn.Module):
    def _init_(self, num_classes, hidden_size=256, embed_size=256):
        super(AttentionOCR, self)._init_()

        # Feature extractor (Pretrained CNN like ResNet or VGG)
        base_model = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])  # Remove FC layers

        # Fully connected layer to adjust feature map channels
        self.fc = nn.Linear(512, hidden_size)

        # LSTM Decoder with Attention
        self.embedding = nn.Embedding(num_classes, embed_size)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True)
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)

        # Output layer
        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, images, targets=None, max_length=30):
        # Extract features
        features = self.feature_extractor(images)  # (B, C, H, W)
        b, c, h, w = features.size()
        features = features.view(b, c, h * w).permute(0, 2, 1)  # (B, H*W, C)
        # Reduce feature dimension
        features = self.fc(features)  # (B, H*W, hidden_size)

        # Initialize hidden state
        hidden = torch.zeros(1, b, features.size(-1)).to(images.device)

        # Decoder with Attention
        outputs = []
        input_char = torch.zeros(b, dtype=torch.long).to(images.device)  # Start token

        for t in range(max_length):
            embedded = self.embedding(input_char)  # (B, embed_size)

            # Attention mechanism
            attn_weights = torch.softmax(self.attn(hidden.squeeze(0)), dim=-1)  # (B, hidden_size)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), features).squeeze(1)  # (B, hidden_size)

            # Combine with embedded input
            rnn_input = torch.cat((embedded, attn_applied), dim=-1)  # (B, embed_size + hidden_size)
            rnn_input = self.attn_combine(rnn_input).unsqueeze(1)

            # LSTM step
            output, (hidden, _) = self.lstm(rnn_input, (hidden, hidden))

            # Output character prediction
            char_output = self.fc_out(output.squeeze(1))
            outputs.append(char_output)

            # Get next input character (teacher forcing during training)
            if targets is not None:
                input_char = targets[:, t]
            else:
                input_char = char_output.argmax(dim=-1)
        return torch.stack(outputs, dim=1)  # (B, max_length, num_classes)


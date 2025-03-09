import torch
import torch.nn as nn
import torchvision.models as models


class AttentionOCR(nn.Module):
    def __init__(self, num_classes=10, hidden_size=256, embed_size=256):
        super(AttentionOCR, self).__init__()
        # Store hidden_size as an instance variable
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        # Feature extractor (ResNet18 without the final FC layers)
        base_model = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])  # Remove FC and avgpool layers

        # Linear layer to project ResNet features to hidden_size
        self.fc = nn.Linear(512, hidden_size)

        # Embedding layer for target characters
        self.embedding = nn.Embedding(num_classes, embed_size)

        # Attention mechanism
        self.attn = nn.Linear(hidden_size + embed_size, hidden_size)
        self.attn_combine = nn.Linear(hidden_size + embed_size, hidden_size)

        # LSTM Decoder
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # Output layer
        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, images, targets=None, max_length=30):
        # Extract features from the image using ResNet
        features = self.feature_extractor(images)  # (B, 512, H, W)
        b, c, h, w = features.size()
        features = features.view(b, c, h * w).permute(0, 2, 1)  # (B, H*W, 512)
        features = self.fc(features)  # (B, H*W, hidden_size)

        # Initialize hidden state and cell state for LSTM
        hidden = torch.zeros(1, b, self.hidden_size).to(images.device)  # (1, B, hidden_size)
        cell = torch.zeros(1, b, self.hidden_size).to(images.device)  # (1, B, hidden_size)

        # Start token (all zeros)
        input_char = torch.zeros(b, dtype=torch.long).to(images.device)  # (B,)

        # Store outputs
        outputs = []

        for t in range(max_length):
            # Embed the input character
            embedded = self.embedding(input_char)  # (B, embed_size)

            # Compute attention weights
            attn_input = torch.cat((hidden.squeeze(0), embedded), dim=-1)  # (B, hidden_size + embed_size)
            attn_weights = torch.softmax(self.attn(attn_input), dim=-1)  # (B, hidden_size)

            # Apply attention to encoder features
            attn_weights = attn_weights.unsqueeze(2)  # (B, hidden_size, 1)
            attn_applied = torch.bmm(features, attn_weights).squeeze(2)  # (B, H*W)

            # Normalize attention weights
            attn_weights = torch.softmax(attn_applied, dim=-1)  # (B, H*W)
            attn_weights = attn_weights.unsqueeze(1)  # (B, 1, H*W)

            # Compute weighted sum of encoder features
            attn_applied = torch.bmm(attn_weights, features).squeeze(1)  # (B, hidden_size)

            # Combine attention context with embedded input
            rnn_input = torch.cat((embedded, attn_applied), dim=-1)  # (B, embed_size + hidden_size)
            rnn_input = self.attn_combine(rnn_input).unsqueeze(1)  # (B, 1, hidden_size)

            # LSTM step
            output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))  # output: (B, 1, hidden_size)

            # Predict the next character
            char_output = self.fc_out(output.squeeze(1))  # (B, num_classes)
            outputs.append(char_output)

            # Prepare the next input character
            if targets is not None:
                # Teacher forcing: use the ground truth as the next input
                input_char = targets[:, t]
            else:
                # Inference: use the predicted character as the next input
                input_char = char_output.argmax(dim=-1)

        # Stack outputs along the time dimension
        return torch.stack(outputs, dim=1)  # (B, max_length, num_classes)


if __name__ == '__main__':
    # Test the model
    image = torch.randn((1, 3, 256, 256))  # Batch of 1 image
    model = AttentionOCR()
    output = model(image)
    print("Output Shape:", output.shape)  # Should be (1, 30, 10)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.inception import BasicConv2d, InceptionA


class InceptNet(nn.Module):
    def __init__(self, input_channels=1):
        super(InceptNet, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            BasicConv2d(input_channels, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3),
            BasicConv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BasicConv2d(64, 80, kernel_size=1),
            BasicConv2d(80, 192, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Inception blocks
        self.inception_blocks = nn.Sequential(
            InceptionA(192, pool_features=32),
            InceptionA(256, pool_features=64),
            InceptionA(288, pool_features=64),
        )

    def forward(self, x):
        x = self.conv_layers(x)  # Apply convolutional layers
        x = self.inception_blocks(x)  # Apply inception blocks
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size, sos_id, eos_id):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.sos_id = sos_id
        self.eos_id = eos_id

        # Embedding layer
        self.emb = nn.Embedding(vocab_size, hidden_size)

        # GRU layer
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # Output layer
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, hidden):
        # Embed the input
        embedded = self.emb(inputs)

        # Pass through GRU
        outputs, hidden = self.rnn(embedded, hidden)

        # Apply output layer
        outputs = self.out(outputs)
        return outputs, hidden


class OCR_Model(nn.Module):
    def __init__(self, img_width, img_height, nh, n_classes, max_len, SOS_token, EOS_token):
        super(OCR_Model, self).__init__()

        # Feature extractor
        self.incept_model = InceptNet(input_channels=1)

        # Decoder
        self.decoder = Decoder(n_classes, max_len, nh, SOS_token, EOS_token)

    def forward(self, input_, target_seq=None, teacher_forcing_ratio=0):
        # Extract features using InceptNet
        encoder_outputs = self.incept_model(input_)

        # Flatten encoder outputs
        b, fc, fh, fw = encoder_outputs.size()
        encoder_outputs = encoder_outputs.view(b, -1, fc)

        # Initialize hidden state for decoder
        hidden = torch.zeros(1, b, self.decoder.hidden_size, device=input_.device)

        # Decode
        if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
            # Teacher forcing: use ground truth as input
            decoder_outputs, _ = self.decoder(target_seq, hidden)
        else:
            # Inference: use predicted tokens as input
            decoder_input = torch.full((b, 1), self.decoder.sos_id, device=input_.device)
            decoder_outputs = []
            for _ in range(self.decoder.max_len):
                decoder_output, hidden = self.decoder(decoder_input, hidden)
                decoder_outputs.append(decoder_output)
                decoder_input = decoder_output.argmax(-1)
            decoder_outputs = torch.cat(decoder_outputs, dim=1)

        return decoder_outputs


if __name__ == '__main__':
    # Test InceptNet
    incept_model = InceptNet(input_channels=1)
    x = torch.randn(1, 1, 160, 60)
    print("Input shape:", x.shape)
    f = incept_model(x)
    print("InceptNet output shape:", f.shape)

    # Test OCR_Model
    img_width, img_height = 160, 60
    nh = 512
    n_classes = 38
    max_len = 10
    SOS_token = 0
    EOS_token = 1

    ocr_model = OCR_Model(img_width, img_height, nh, n_classes, max_len, SOS_token, EOS_token)
    x = torch.randn(1, 1, img_height, img_width)
    print("OCR_Model input shape:", x.shape)
    output = ocr_model(x)
    print("OCR_Model output shape:", output.shape)

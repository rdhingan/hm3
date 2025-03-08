from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels,stride=1):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
            self.relu = torch.nn.ReLU()
            self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            if in_channels != out_channels:
                self.skip = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            else:
                self.skip = torch.nn.Identity()

        def forward(self, x):
            identity = self.skip(x)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x) 
            identity = self.pool(identity) 
            return identity + x

    def __init__(self, channels_l0=16, n_blocks=2, num_classes=6):
        super().__init__()
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d(1)
        cnn_layers = [
            torch.nn.Conv2d(3, channels_l0, kernel_size=11, stride=2, padding=5),
            torch.nn.ReLU(),          
        ]
        
        c1 = channels_l0
        for _ in range(n_blocks):
            c2 = c1 * 2  # Double channels at each block
            cnn_layers.append(self.Block(c1, c2))
            # cnn_layers.append(torch.nn.ReLU())
            c1 = c2
        #  One to One convolation :- We need output of 1 channel for each pixel in the image and kernel size is 1    
        cnn_layers.append(torch.nn.Conv2d(c2, 6, kernel_size=1))
        # cnn_layers.append(torch.nn.AdaptiveAvgPool2d(1))
        self.network = torch.nn.Sequential(*cnn_layers)

    def forward(self, x):
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        y = self.adaptive_pool(self.network(z))
        return y.squeeze()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))


        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        
        def up_conv(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.encoder1 = conv_block(in_channels, 16)   # 3 --> 64
        self.encoder2 = conv_block(16, 32)  # 64 --> 128
        self.encoder3 = conv_block(32, 64)  # 128 --> 256
        self.encoder4 = conv_block(64, 128)  # 256 --> 512
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = conv_block(128, 256)  # 512 --> 1024
        
        self.upconv4 = up_conv(256, 128)    # 1024 --> 512
        self.decoder4 = conv_block(256, 128)   # 1024 --> 512
        
        self.upconv3 = up_conv(128, 64)    # 512 --> 256
        self.decoder3 = conv_block(128, 64)  # 512 --> 256
        
        self.upconv2 = up_conv(64, 32)    # 256 --> 128
        self.decoder2 = conv_block(64, 32)  # 256 --> 128
        
        self.upconv1 = up_conv(32, 16)  # 128 --> 64
        self.decoder1 = conv_block(32, 16)  # 128 --> 64
        
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1) # Segmentation output
        self.depth_conv = nn.Conv2d(16, 1, kernel_size=1)  # Depth prediction layer ✅


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        enc1 = self.encoder1(z)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        bottleneck = self.bottleneck(self.pool(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        logits = self.final_conv(dec1)  # Segmentation logits
        raw_depth = self.depth_conv(dec1)  # Raw depth output
        return logits, raw_depth.squeeze(1)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()

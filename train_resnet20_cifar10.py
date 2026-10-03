#!/usr/bin/env python3
"""
Train a ResNet-20 (CIFAR-10, BatchNorm-free variant) on real CIFAR-10 data
and export to ONNX.

This is the classic ResNet-20 architecture from He et al. 2015 (3 stages of
3 BasicBlocks each, 16->32->64 channels), with BatchNorm layers removed --
a common, legitimate simplification. This isolates the two genuinely new
elements this model introduces over the pipeline's earlier models (Conv2D,
and real skip-connection Adds combining two different points in the graph)
without also introducing BatchNorm as a third new element at the same time.
BatchNorm can be added back as a clean follow-up once Conv2D + skip
connections are confirmed working through the full annotation/raising
pipeline.

Usage:
    python3 train_resnet20_cifar10.py --onnx-out resnet20_cifar10.onnx \
        --epochs 20 --subset-size 10000
"""

import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Conv -> ReLU -> Conv -> (+skip) -> ReLU, no BatchNorm.

    Uses EXPLICIT padding (a separate pad step) before each 3x3 conv,
    instead of relying on Conv2d's own implicit padding -- numerically
    identical (spatial dimensions unchanged either way), but exports to
    ONNX as a separate onnx.Pad + a boundary-free (VALID) onnx.Conv,
    rather than a single onnx.Conv carrying its own pads attribute. This
    was needed to work around an onnx-mlir krnl-to-affine lowering bug
    ("dimensional operand cannot be used as a symbol") specifically
    triggered by Conv's own implicit same-style padding boundary
    computation.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=0, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=0, bias=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            # 1x1 conv to match spatial size and channel count for the
            # skip connection, matching the classic ResNet "option B"
            # downsample. 1x1 convs have no padding to begin with, so no
            # explicit-pad change needed here.
            self.downsample = nn.Conv2d(in_channels, out_channels,
                                        kernel_size=1, stride=stride, bias=True)

    def forward(self, x):
        identity = x
        out = F.pad(x, (1, 1, 1, 1))  # explicit pad=1 on all sides, replacing conv1's old implicit padding=1
        out = F.relu(self.conv1(out))
        out = F.pad(out, (1, 1, 1, 1))  # explicit pad=1, replacing conv2's old implicit padding=1
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity  # the actual skip connection
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0, bias=True)

        self.stage1 = self._make_stage(16, 16, num_blocks=3, stride=1)
        self.stage2 = self._make_stage(16, 32, num_blocks=3, stride=2)
        self.stage3 = self._make_stage(32, 64, num_blocks=3, stride=2)

        self.fc = nn.Linear(64, num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1))
        x = F.relu(self.conv1(x))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, 1)  # global average pool -> (N, 64, 1, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.softmax(x, dim=1)


def load_cifar10(subset_size, seed=0):
    from torchvision import datasets, transforms
    print("Fetching CIFAR-10 (cached after first run)...", file=sys.stderr)
    transform = transforms.Compose([transforms.ToTensor()])  # plain [0,1] scaling
    train_set = datasets.CIFAR10(root="./cifar10_data", train=True,
                                 download=True, transform=transform)

    if subset_size and subset_size < len(train_set):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(train_set), generator=g)[:subset_size]
        train_set = torch.utils.data.Subset(train_set, idx.tolist())

    return train_set


def train(model, train_set, epochs, batch_size, lr, device):
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                         shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()  # model already applies softmax; use log+NLL

    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            probs = model(images)
            loss = criterion(torch.log(probs.clamp_min(1e-9)), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (probs.argmax(1) == labels).sum().item()
            total += images.size(0)

        print(f"epoch {epoch+1}/{epochs}: loss={total_loss/total:.4f} "
              f"acc={correct/total:.4f}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-out", default="resnet20_cifar10.onnx")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--subset-size", type=int, default=10000,
                         help="Number of training images to use (default: 10000, "
                              "full CIFAR-10 train set is 50000). Use 0 for full set.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", file=sys.stderr)

    model = ResNet20().to(device)

    subset_size = None if args.subset_size == 0 else args.subset_size
    train_set = load_cifar10(subset_size, seed=args.seed)
    print(f"Training on {len(train_set)} images", file=sys.stderr)

    train(model, train_set, args.epochs, args.batch_size, args.lr, device)

    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    torch.onnx.export(
        model, dummy_input, args.onnx_out,
        input_names=["image"], output_names=["prediction"],
        opset_version=13,
        dynamo=False,  # the newer default exporter silently falls back to a
                       # higher opset and uses less standard op translations
                       # -- the legacy exporter gives more predictable,
                       # onnx-mlir-friendly output.
    )

    # The legacy exporter represents F.pad's padding amounts as a dynamic,
    # runtime-computed subgraph (Constant/ConstantOfShape/Concat/Slice/
    # Transpose/Cast chains) even though they're genuinely compile-time
    # constants here -- onnx-simplifier folds this down to clean static
    # values. skipped_optimizers=["fuse_pad_into_conv"] is essential: the
    # default behavior would otherwise merge the explicit Pad back into
    # Conv's own implicit padding attribute, which is exactly the pattern
    # that triggers onnx-mlir's krnl-to-affine "dimensional operand cannot
    # be used as a symbol" bug -- undoing the whole point of using
    # explicit padding in the first place.
    import onnx
    from onnxsim import simplify
    model_onnx = onnx.load(args.onnx_out)
    model_simplified, check = simplify(model_onnx,
                                       skipped_optimizers=["fuse_pad_into_conv"])
    if not check:
        print("WARNING: onnxsim's correctness check failed -- inspect the "
              "output carefully before trusting it", file=sys.stderr)
    onnx.save(model_simplified, args.onnx_out)
    print(f"Saved {args.onnx_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
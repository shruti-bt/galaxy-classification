r"""This files implements a testing steps for your model."""

import argparse
import torch
import torch.nn as nn
from torch.serialization import load

from utils import load_data, progress_bar, load_model
from collections import OrderedDict

def run(args: argparse.Namespace, device: torch.device) -> None:
    r"""It used to test vgg implementation on CIFAR-10 dataset.
    
    Arguments:
    ---------
        args (argparse.Namespace): Collection of command line arguments.
        device (torch.device): Physical device used for testing.
        
    """
    test_dataloader, classes = load_data(args)
    model = load_model(args.model_name).to(device)
    checkpoint = torch.load(
        # os.path.join(args.weigths_path, f"{args.model_name}-{args.dataset}.pth"),
        f"{args.weigths_path}/{args.model_name}-{args.dataset}.pth",
        map_location=device
    )
    state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in checkpoint['model'].items())
    model.load_state_dict(state_dict)
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(test_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


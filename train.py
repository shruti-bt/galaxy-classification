r"""This file implements a training step for model."""

import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from model import vgg
from utils import load_data, load_model, progress_bar, plot_metrics, load_model

def run(args, device):
    r"""TO BE DOCUMENTED."""
    train_loader, val_loader = load_data(args, is_train=True)
    model = load_model(args).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    for epoch in range(1, args.num_epochs+1):
        print('\nEpoch: %d' % epoch)

        # Training
        # --------
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            corrects += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (running_loss/(batch_idx+1), 100.*corrects/total, corrects, total))

        train_loss.append(running_loss / len(train_loader))
        train_acc.append(corrects / total)

        # Validation
        # ----------
        model.eval()
        running_loss = 0.0
        corrects = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                corrects += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (running_loss/(batch_idx+1), 100.*corrects/total, corrects, total))

        val_loss.append(running_loss / len(val_loader))
        val_acc.append(corrects / total)

        scheduler.step()
        
        # Save checkpoints
        # ----------------
        acc = 100.*corrects/total
        if acc > args.best_acc:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, args.weights_path)
        args.best_acc = acc

    # plot metrics
    plot_metrics(
        series=[train_acc, val_acc], 
        labels=["Train", "Val"], 
        xlabel="Epoch", 
        ylabel="Accuracy", 
        xticks=np.arange(0, args.num_epochs+1, args.num_epochs//8),
        yticks=np.arange(0, 1.01, 0.2),
        save_path=f"{args.model_name}-{args.dataset}_acc.svg"
    )

    plot_metrics(
        series=[train_loss, val_loss], 
        labels=["Train", "Val"], 
        xlabel="Epoch", 
        ylabel="Loss", 
        xticks=np.arange(0, args.num_epochs+1, args.num_epochs//8),
        yticks=None,
        save_path=f"{args.model_name}-{args.dataset}_loss.svg"
    )


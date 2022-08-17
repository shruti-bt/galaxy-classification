import torch
import torch.nn as nn
import torch.optim as optim

import hydra
import logging
from omegaconf import DictConfig, OmegaConf

from galaxy_cls.dataset import load_data
from galaxy_cls.model import get_model
from galaxy_cls.trainer import Trainer
import torch.optim.lr_scheduler as lr_scheduler


@hydra.main(config_path="./",config_name="config.yaml")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Selected device as - {device}.")

    train_dataloader, test_dataloader = load_data(cfg.data_path, cfg.batch_size, cfg.img_mean, cfg.img_std, is_train=True)
    model = get_model(cfg.model_name, cfg.num_classes)
    # print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10,25], gamma=0.1)
    scheduler.print_lr(True, optimizer, cfg.learning_rate, epoch=5)

    trainer = Trainer(cfg.print_freq, model, criterion, optimizer, scheduler, device)
    trainer.fit(train_dataloader, test_dataloader, cfg.num_epochs)


if __name__ == "__main__":
    main()
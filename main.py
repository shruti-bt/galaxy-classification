import torch
import torch.nn as nn
import torch.optim as optim

import hydra
from omegaconf import DictConfig, OmegaConf

from galaxy_cls.dataset import load_data
from galaxy_cls.model import get_model
from galaxy_cls.trainer import Trainer


@hydra.main(version_base=None, config_name="config.yaml")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader, test_dataloader = load_data(cfg.data_path, cfg.batch_size, cfg.img_mean, cfg.img_std, is_train=True)
    model = get_model(cfg.model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)

    trainer = Trainer(cfg.print_freq, model, criterion, optimizer, device)
    trainer.fit(train_dataloader, test_dataloader, cfg.num_epochs)


if __name__ == "__main__":
    main()
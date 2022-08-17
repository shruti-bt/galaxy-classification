import logging
import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Trainer():
    def __init__(self, print_freq, model, criterion, optimizer, scheduler, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.print_freq = print_freq

    def training_step(self, inputs, labels):
        self.model.train()
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        corrects = (predicted == labels).sum().item()

        return loss.item(), corrects

    def validation_step(self, inputs, labels):
        self.model.eval()
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        corrects = (predicted == labels).sum().item()

        return loss.item(), corrects

    def fit(self, train_data, valid_data, num_epochs):
        prev_val_acc = 0.0
        for epoch in range(num_epochs):

            logging.info(f"Epoch [{epoch+1:2d}/{num_epochs}]")
            logging.info("=============")

            train_loss = 0.0
            train_acc = 0
            train_imgs = 0

            for batch_idx, (inputs, labels) in enumerate(train_data):
                loss, acc = self.training_step(inputs, labels)
                train_loss += loss
                train_acc += acc
                train_imgs += labels.size(0)
            
                if batch_idx % self.print_freq == 0:
                    logging.info(f"\t[{batch_idx}/{len(train_data)}] Loss: {train_loss/(batch_idx+1):.4f} Acc: {train_acc/train_imgs:.4f}")

            valid_loss = 0.0
            valid_acc = 0
            valid_imgs = 0

            for batch_idx, (inputs, labels) in enumerate(valid_data):
                loss, acc = self.validation_step(inputs, labels)
                valid_loss += loss
                valid_acc += acc
                valid_imgs += labels.size(0)
            
                if batch_idx % self.print_freq == 0:
                    logging.info(f"\t[{batch_idx}/{len(valid_data)}] Loss: {valid_loss/(batch_idx+1):.4f} Acc: {valid_acc/valid_imgs:.4f}")


            self.scheduler.step()
            print(self.optimizer.state_dict()['param_groups'][0]['lr'])
            total_val_acc = valid_acc / valid_imgs
            logging.info(f"Train Loss: {train_loss/len(train_data):.4f} Acc: {train_acc/train_imgs:.4f} | Val Loss: {valid_loss/len(valid_data):.4f} Acc: {total_val_acc:.4f}")

            if total_val_acc > prev_val_acc:
                self.save_weights("best_model.pt")
        
            self.save_weights("last_model.pt")



    def validate(self, valid_data):
        pass

    def predict(self, test_data):
        pass

    def save_weights(self, model_name) -> None:
        torch.save(self.model.state_dict(), model_name)

    def load_weights(self, model, path) -> torch.nn.Module:
        model.load_state_dict(torch.load(path))
        return model
                
           


if __name__ == "__main__":
    pass


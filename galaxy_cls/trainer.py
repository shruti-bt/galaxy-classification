import enum
import torch

class Trainer():
    def __init__(self, print_freq, model, criterion, optimizer, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
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
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1:2d}/{num_epochs}]")
            print("=============")

            train_loss = 0.0
            train_acc = 0
            train_imgs = 0

            for batch_idx, (inputs, labels) in enumerate(train_data):
                loss, acc = self.training_step(inputs, labels)
                train_loss += loss
                train_acc += acc
                train_imgs += labels.size(0)
            
                if batch_idx % self.print_freq == 0:
                    print(f"\t[{batch_idx}/{len(train_data)}] Loss: {train_loss/(batch_idx+1):.4f} Acc: {train_acc/train_imgs:.4f}")

            valid_loss = 0.0
            valid_acc = 0
            valid_imgs = 0

            for batch_idx, (inputs, labels) in enumerate(valid_data):
                loss, acc = self.validation_step(inputs, labels)
                valid_loss += loss
                valid_acc += acc
                valid_imgs += labels.size(0)
            
                if batch_idx % self.print_freq == 0:
                    print(f"\t[{batch_idx}/{len(valid_data)}] Loss: {valid_loss/(batch_idx+1):.4f} Acc: {valid_acc/valid_imgs:.4f}")

            print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {valid_loss:.4f} Acc: {valid_acc:.4f}")


    def validate(self, valid_data):
        pass

    def predict(self, test_data):
        pass
                
           


if __name__ == "__main__":
    pass

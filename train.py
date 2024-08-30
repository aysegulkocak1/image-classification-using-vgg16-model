import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import itertools
from tqdm import tqdm  

class VGG:
    def __init__(self):
        self.args = self.parse_arg()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def parse_arg(self):
        parser = argparse.ArgumentParser(description="VGG16 Model Training Script")
        parser.add_argument("--dataroot", type=str, required=True)
        parser.add_argument("--validroot", type=str, required=True)
        parser.add_argument("--datainfo", type=str, required=True, help="root of data.txt")
        parser.add_argument("--epochs", type=int, default=10, help="total training epochs")
        parser.add_argument("--img_size", type=int, default=224, help="train, val image size (pixels)")
        parser.add_argument("--save_dir", type=str, default="models", help="Directory to save the models and plots")
        args = parser.parse_args()

        args.dataroot = os.path.abspath(args.dataroot)
        args.validroot = os.path.abspath(args.validroot)
        args.datainfo = os.path.abspath(args.datainfo)
        args.save_dir = os.path.abspath(args.save_dir)
        print("Dataroot:", args.dataroot)
        print("Validroot:", args.validroot)
        print("Absolute Dataroot Path:", os.path.abspath(args.dataroot))
        print("Absolute Validroot Path:", os.path.abspath(args.validroot))
        print("Save Directory:", args.save_dir)
        return args

    def parse_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        classes = []
        num_of_classes = 0
        batch_size = 0

        for line in lines:
            line = line.strip()
            if line.startswith("classes"):
                classes_str = line.split('=')[1].strip()[1:-1]
                classes = [item.strip() for item in classes_str.split(',')]

            elif line.startswith("num_of_classes"):
                num_of_classes = int(line.split('=')[1].strip())

            elif line.startswith("batch_size"):
                batch_size = int(line.split('=')[1].strip())

        return classes, num_of_classes, batch_size

    def load_model(self):
        file_root = self.args.datainfo
        classes, num_classes, batch_size = self.parse_file(file_root)
        image_size = self.args.img_size

        model = models.vgg16(weights=True)
        for param in model.parameters():
            param.requires_grad = False

        model.classifier[6] = nn.Linear(4096, num_classes)
        model = model.to(self.device)

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = datasets.ImageFolder(root=self.args.dataroot, transform=transform)
        valid_dataset = datasets.ImageFolder(root=self.args.validroot, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_accuracy = 0.0
        best_model_path = os.path.join(self.args.save_dir, 'best_model.pth')
        last_model_path = os.path.join(self.args.save_dir, 'last_model.pth')

        os.makedirs(self.args.save_dir, exist_ok=True)

        train_losses = []
        train_accuracies = []
        valid_losses = []
        valid_accuracies = []
        
        for epoch in range(self.args.epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{self.args.epochs}', unit='batch') as pbar:
                for images, labels in train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    pbar.set_postfix(loss=running_loss/len(train_loader), accuracy=100 * correct / total)
                    pbar.update(1)

            avg_train_loss = running_loss / len(train_loader)
            avg_train_accuracy = 100 * correct / total
            train_losses.append(avg_train_loss)
            train_accuracies.append(avg_train_accuracy)

            model.eval()
            valid_running_loss = 0.0
            valid_correct = 0
            valid_total = 0

            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    valid_running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    valid_total += labels.size(0)
                    valid_correct += (predicted == labels).sum().item()

            avg_valid_loss = valid_running_loss / len(valid_loader)
            avg_valid_accuracy = 100 * valid_correct / valid_total
            valid_losses.append(avg_valid_loss)
            valid_accuracies.append(avg_valid_accuracy)

            print(f"Epoch [{epoch+1}/{self.args.epochs}], Train Loss: {avg_train_loss}, Train Accuracy: {avg_train_accuracy}")
            print(f"Epoch [{epoch+1}/{self.args.epochs}], Valid Loss: {avg_valid_loss}, Valid Accuracy: {avg_valid_accuracy}")

            if avg_valid_accuracy > best_accuracy:
                best_accuracy = avg_valid_accuracy
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved with Accuracy: {best_accuracy}")

            torch.save(model.state_dict(), last_model_path)

        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        cm = confusion_matrix(y_true, y_pred)
        self.plot_confusion(cm, num_classes)

        self.save_metrics(train_losses, train_accuracies, valid_losses, valid_accuracies)

    def plot_confusion(self, cm, num_classes):
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        classes = [str(i) for i in range(num_classes)]
        plt.xticks(np.arange(num_classes), classes, rotation=45)
        plt.yticks(np.arange(num_classes), classes)

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.save_dir, 'confusion_matrix.png'))
        plt.close()

    def save_metrics(self, train_losses, train_accuracies, valid_losses, valid_accuracies):
        epochs = range(1, len(train_losses) + 1)
        
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
        plt.plot(epochs, valid_losses, 'ro-', label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
        plt.plot(epochs, valid_accuracies, 'ro-', label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.save_dir, 'training_validation_metrics.png'))
        plt.close()

if __name__ == "__main__":
    vgg = VGG()
    vgg.load_model()

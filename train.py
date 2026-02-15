import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CIFAR10Dataset
from model import FashionNet

def evaluate_model(model, test_loader):
     model.eval()
     correct = 0
     total = 0

     with torch.no_grad():
          for batch in test_loader:
               bilder, labels = batch
               predictions = model(bilder)

               _, predicted = torch.max(predictions, 1)

               total += labels.size(0)
               correct += torch.sum(predicted == labels).item()

     accuracy = 100 * correct / total
     return accuracy


def train_model(epochs=20, batch_size=32, lr=0.001):
    dataset = CIFAR10Dataset(train=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = FashionNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in loader:
            bilder, labels = batch
            predictions = model(bilder)
            loss = criterion(predictions, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        train_losses.append(avg_loss)
        
        test_dataset_temp = CIFAR10Dataset(train=False)
        test_loader_temp = DataLoader(test_dataset_temp, batch_size=batch_size)
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader_temp:
                bilder, labels = batch
                predictions = model(bilder)
                loss = criterion(predictions, labels)
                test_loss += loss.item()
        
        test_loss = test_loss / len(test_loader_temp)
        test_losses.append(test_loss)
        model.train()
        
        print(f"epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    test_dataset = CIFAR10Dataset(train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    filename = f'model_ep{epochs}_bs{batch_size}_lr{lr}.pth'
    torch.save(model.state_dict(), filename)
    print(f"Modell sparad: {filename}")
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_accuracy': accuracy,
        'model_file': filename
    }
if __name__ == "__main__":
    train_model()
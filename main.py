from train import train_model

if __name__ == "__main__":
    print("\nEXPERIMENT 1:")
    train_model(epochs=20, batch_size=32, lr=0.001)
    
    print("\nEXPERIMENT 2:")
    train_model(epochs=20, batch_size=32, lr=0.01)
    
    print("\nEXPERIMENT 3:")
    train_model(epochs=20, batch_size=64, lr=0.001)
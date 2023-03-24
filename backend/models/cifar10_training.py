import torch
from models.cifar10_model import CIFAR10Model
from utils.locks import training_lock

def start_training(epochs=5):
    with training_lock:
        model = CIFAR10Model()
        model.train(epochs=epochs)

        # Save the trained model
        torch.save(model.model.state_dict(), 'trained_model.pt')

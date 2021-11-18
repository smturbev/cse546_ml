if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer
    from losses import MSELossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer
    from .optimizers import SGDOptimizer
    from .losses import MSELossLayer
    from .train import plot_model_guesses, train


from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)

class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_0 = LinearLayer(2,2, generator=RNG)

    def forward(self, inputs):
        return self.linear_0(inputs)

class SigNetwork(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear_0 = LinearLayer(2, hidden_size, generator=RNG)
        self.linear_1 = LinearLayer(hidden_size, 2, generator=RNG)

    def forward(self, inputs):
        sig = SigmoidLayer()
        x = self.linear_0(inputs)
        x = sig(x)
        return self.linear_1(x)

class ReLUNetwork(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear_0 = LinearLayer(2, hidden_size, generator=RNG)
        self.linear_1 = LinearLayer(hidden_size, 2, generator=RNG)

    def forward(self, inputs):
        relu = ReLULayer()
        x = self.linear_0(inputs)
        x = relu.forward(x)
        return self.linear_1(x)

class SigReLUNetwork(nn.Module):
    def __init__(self, hidden_size0, hidden_size1):
        super().__init__()
        self.linear_0 = LinearLayer(2, hidden_size0, generator=RNG)
        self.linear_1 = LinearLayer(hidden_size0, hidden_size1, generator=RNG)
        self.linear_2 = LinearLayer(hidden_size1, 2, generator=RNG)

    def forward(self, inputs):
        sig = SigmoidLayer()
        relu = ReLULayer()
        x = self.linear_0(inputs)
        x = sig(x)
        x = self.linear_1(x)
        x = relu(x)
        return self.linear_2(x)

class ReLUSigNetwork(nn.Module):
    def __init__(self, hidden_size0, hidden_size1):
        super().__init__()
        self.linear_0 = LinearLayer(2, hidden_size0, generator=RNG)
        self.linear_1 = LinearLayer(hidden_size0, hidden_size1, generator=RNG)
        self.linear_2 = LinearLayer(hidden_size1, 2, generator=RNG)

    def forward(self, inputs):
        sig = SigmoidLayer()
        relu = ReLULayer()
        x = self.linear_0(inputs)
        x = relu(x)
        x = self.linear_1(x)
        x = sig(x)
        return self.linear_2(x)

@problem.tag("hw3-A")
def accuracy_score(model: nn.Module, dataloader: DataLoader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for MSE.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is also a 2-d vector of floats, but specifically with one being 1.0, while other is 0.0.
            Index of 1.0 in target corresponds to the true class.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to CrossEntropy accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    accuracy = 0
    for batch in dataloader:
        data, labels = batch
        data, labels = data, labels
        y_hat = model(data)
        y_pred = torch.argmax(y_hat,1)
        y_labels = torch.argmax(labels,1)
        accuracy += torch.sum(y_pred==y_labels).item()
    return accuracy/len(dataloader.dataset)


@problem.tag("hw3-A")
def mse_parameter_search(
    dataloader_train: DataLoader, dataloader_val: DataLoader, epochs=100
) -> Dict[str, Any]:
    """
    Main subroutine of the MSE problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers

    Args:
        dataloader_train (DataLoader): Dataloader for training dataset.
        dataloader_val (DataLoader): Dataloader for validation dataset.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """
    model_dict ={}

    # Linear
    model = Linear()
    loss = MSELossLayer()
    train_dict = train(dataloader_train, model=model, criterion=loss, optimizer=SGDOptimizer, val_loader=dataloader_val)
    model_dict["linear"] = {"train":train_dict["train"], "val":train_dict["val"], "model":model}

    # one hidden layer, sigmoid
    model = SigNetwork(2)
    loss = MSELossLayer()
    train_dict = train(dataloader_train, model=model, criterion=loss, optimizer=SGDOptimizer, val_loader=dataloader_val)
    model_dict["hidden1_sig"] = {"train":train_dict["train"], "val":train_dict["val"], "model":model}

    # one hidden layer, ReLU
    model = ReLUNetwork(2)
    loss = MSELossLayer()
    train_dict = train(dataloader_train, model=model, criterion=loss, optimizer=SGDOptimizer, val_loader=dataloader_val)
    model_dict["hidden1_ReLU"] = {"train":train_dict["train"], "val":train_dict["val"], "model":model}

    # two hidden layer, Sig then ReLU
    model = SigReLUNetwork(2,2)
    loss = MSELossLayer()
    train_dict = train(dataloader_train, model=model, criterion=loss, optimizer=SGDOptimizer, val_loader=dataloader_val)
    model_dict["hidden2_Sig_ReLU"] = {"train":train_dict["train"], "val":train_dict["val"], "model":model}

    # two hidden layer, ReLU then Sig
    model = ReLUSigNetwork(2,2)
    loss = MSELossLayer()
    train_dict = train(dataloader_train, model=model, criterion=loss, optimizer=SGDOptimizer, val_loader=dataloader_val)
    model_dict["hidden2_ReLU_Sig"] = {"train":train_dict["train"], "val":train_dict["val"], "model":model}

    return model_dict


@problem.tag("hw3-A", start_line=21)
def main():
    """
    Main function of the MSE problem.
    It should:
        1. Call mse_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me MSE loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    mse_dataloader_train = DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(to_one_hot(y))),
        batch_size=32,
        shuffle=True,
        generator=RNG,
    )
    mse_dataloader_val = DataLoader(
        TensorDataset(
            torch.from_numpy(x_val).float(), torch.from_numpy(to_one_hot(y_val))
        ),
        batch_size=32,
        shuffle=False,
    )
    mse_dataloader_test = DataLoader(
        TensorDataset(
            torch.from_numpy(x_test).float(), torch.from_numpy(to_one_hot(y_test))
        ),
        batch_size=32,
        shuffle=False,
    )

    mse_dict = mse_parameter_search(mse_dataloader_train, mse_dataloader_val)

    mins = {}
    for key in mse_dict:
        mins[key] = (np.min(mse_dict[key]["val"]))
    print(mins)
    ind_min = np.argmin(list(mins.values()))
    best_model = list(mins.keys())[ind_min]
    print("best model", best_model)
    print("Accuracy:", accuracy_score(mse_dict[best_model]["model"], mse_dataloader_test))

    plot_model_guesses(mse_dataloader_test, mse_dict[best_model]["model"], title="MSE "+best_model)

    plt.figure(figsize=(7,5))
    colors = ["C0","C1","C2","C3","C4","C5"]
    i = 0
    for key in mse_dict:
        plt.plot(range(100), mse_dict[key]["train"], "-", color=colors[i], label=key+" training loss")
        plt.plot(range(100), mse_dict[key]["val"], "--", color=colors[i], label=key+" testing loss")
        i +=1
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("MSE")
    plt.legend()
    plt.show()


def to_one_hot(a: np.ndarray) -> np.ndarray:
    """Helper function. Converts data from categorical to one-hot encoded.

    Args:
        a (np.ndarray): Input array of integers with shape (n,).

    Returns:
        np.ndarray: Array with shape (n, c), where c is maximal element of a.
            Each element of a, has a corresponding one-hot encoded vector of length c.
    """
    r = np.zeros((len(a), 2))
    r[np.arange(len(a)), a] = 1
    return r


if __name__ == "__main__":
    main()

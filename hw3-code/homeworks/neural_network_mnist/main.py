# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

# added for visualization from section notebook
import torchvision.utils


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.alpha0 = 1/math.sqrt(d)
        rand = Uniform(torch.tensor([-self.alpha0]), torch.tensor([self.alpha0]))
        self.weight0 = rand.sample([h,d]).view(h,d)
        self.bias0 = rand.sample([h,])
        self.alpha1 = 1/math.sqrt(h)
        rand = Uniform(torch.tensor([-self.alpha1]), torch.tensor([self.alpha1]))
        self.weight1 = rand.sample([k,h]).view(k,h)
        self.bias1 = rand.sample([k,])
        self.params = [self.weight1, self.weight0, self.bias1, self.bias0]
        for params in self.params:
            params.requires_grad = True
            params = Parameter(params)


    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        x = torch.add(torch.matmul(self.weight0, x.transpose(0,1)), self.bias0)
        x = relu(x)
        x = torch.add(torch.matmul(self.weight1, x), self.bias1)
        return x.transpose(0,1)
        # print(x.size(), self.weight0.size(), self.bias0.size())
        # inner0 = torch.matmul(self.weight0, x.transpose(0,1))
        # print(inner0.size(), self.bias0.size()) # 64 x 32
        # inner = torch.add(inner0, self.bias0)
        # print(inner.size())
        # relu_inner = relu(inner)
        # var= torch.add(torch.matmul(self.weight1, relu_inner), self.bias1)
        # print(var.size())
        # return var.transpose(0,1)


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.alpha0 = 1/math.sqrt(d)
        rand = Uniform(torch.tensor([-self.alpha0]), torch.tensor([self.alpha0]))
        self.weight0 = rand.sample([h0,d]).view(h0,d)
        self.bias0 = rand.sample([h0,])
        self.alpha1 = 1/math.sqrt(h0)
        rand = Uniform(torch.tensor([-self.alpha1]), torch.tensor([self.alpha1]))
        self.weight1 = rand.sample([h1,h0]).view(h1,h0)
        self.bias1 = rand.sample([h1,])
        self.alpha2 = 1/math.sqrt(h1)
        rand = Uniform(torch.tensor([-self.alpha2]), torch.tensor([self.alpha2]))
        self.weight2 = rand.sample([k,h1]).view(k,h1)
        self.bias2 = rand.sample([k,])
        self.params = [self.weight0, self.weight1, self.weight2, self.bias0, self.bias1, self.bias2]
        for params in self.params:
            params.requires_grad = True
            params = Parameter(params)

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        inner0 = torch.add(torch.matmul(self.weight0, x.transpose(0,1)), self.bias0)
        inner1 = torch.add(torch.matmul(self.weight1, relu(inner0)), self.bias1)
        inner2 = torch.add(torch.matmul(self.weight2, relu(inner1)), self.bias2)
        return inner2.transpose(0,1)


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    # train_hist = dict.fromkeys(['train', 'val'])
    # accuracy = 0
    # epochs = 0

    # while accuracy < 0.99:
    #     loss_epoch = 0
    #     for batch in train_loader:
    #         data, labels = batch
    #         y_hat = model.forward(data)
    #         loss = cross_entropy(y_hat, labels)
    #         loss_epoch += loss.item()
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     train_hist['train'].append(loss_epoch/len(train_loader.dataset))
    #     accuracy = torch.mean(y_hat==labels)
    #     epochs += 1
    # return train_hist

    epochs = 32
    opt = optimizer(model.params, lr=5e-3)
    print("model params", len(model.params))
    losses = []
    for i in range(epochs):
        loss_epoch = 0.
        accuracy = 0
        for batch in train_loader:
            images, labels = batch
            images, labels = images, labels
            #images = images.view(-1, 784)
            opt.zero_grad()
            y_hat = model.forward(images)
            y_pred = torch.argmax(y_hat,1)
            # print("y_hat and labels [c x n] and [c]", y_hat.size(), labels.size())
            loss = cross_entropy(y_hat, labels)
            # print("loss", (loss), Parameter(loss, requires_grad=True))
            # loss = Parameter(loss, requires_grad=True)
            loss_epoch += loss.item()
            loss.backward()
            opt.step()
            accuracy += torch.sum(y_pred==labels).item()
        acc = accuracy/len(train_loader.dataset)
        print(i,(acc))
        if (acc)>0.99:
            break
        print("Loss:", loss_epoch / len(train_loader))
        losses.append(loss_epoch / len(train_loader))
        print(sum(p.numel() for p in model.params))
    return losses

def test(model: Module, optimizer: Adam, test_loader: DataLoader) -> Tuple:
    """Returns the loss and accuracy for model on test data
    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        test_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).

    Returns:
        (float, float): Tuple containing loss and accuracy for test set.
    """
    print("model", type(model))
    opt = optimizer(model.params, lr=5e-3)
    print("model params", len(model.params))
    accuracy=0
    for batch in test_loader:
        images, labels = batch
        images, labels = images, labels
        #images = images.view(-1, 784)
        opt.zero_grad()
        y_hat = model.forward(images)
        y_pred = torch.argmax(y_hat,1)
        # print("y_hat and labels [c x n] and [c]", y_hat.size(), labels.size())
        loss = cross_entropy(y_hat, labels)
        # print("loss", (loss), Parameter(loss, requires_grad=True))
        # loss = Parameter(loss, requires_grad=True)
        # loss_epoch += loss.item()
        loss.backward()
        opt.step()
        accuracy += torch.sum(y_pred==labels).item()
    acc = accuracy/len(test_loader.dataset)
    return (loss.item()/len(test_loader), acc)


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    h=64; h0 = 32; h1=32
    d = 784
    k = 10

    train_loader = DataLoader(
        TensorDataset(x, y),
        batch_size=32,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=32,
        shuffle=True,
    )

    # Show one batch of images. Each batch of images has shape [batch_size, 1, 28, 28],
    # where 1 is the "channels" dimension of the image.
    # for images,labels in train_loader:
    #     plt.figure(figsize=(10,10))
    #     grid_img = torchvision.utils.make_grid(images)
    #     plt.imshow(grid_img.permute(1, 2, 0))
    #     plt.title("A single batch of images")
    #     plt.savefig("batch_imgs.png")
    #     break
    f1 = F1(h,d,k)
    f2 = F2(h0,h1,d,k)

    model1 = train(f1,Adam,train_loader)
    model2 = train(f2, Adam, train_loader)

    # plot per epoch loss
    _, ax = plt.subplots(1,1,figsize=(7,3))
    ax.plot(range(len(model1)),model1, "x-", label="model1")
    ax.plot(range(len(model2)),model2, "x-", label="model2")
    ax.set_title("Model loss vs epoch")
    ax.legend()
    plt.savefig("F1_F2_epochs_loss.png")
    plt.close()

    # print accuracy and loss for test set
    test_loss1, test_acc1 = test(f1, Adam, test_loader)
    test_loss2, test_acc2 = test(f2, Adam, test_loader)
    print("Model 1: ", test_loss1, test_acc1)
    print("Model 2: ", test_loss2, test_acc2)



if __name__ == "__main__":
    main()

## Intro to PyTorch
In this problem you will explore PyTorch, and implement basic components of its nn module.
The problem is organized to mimic (though not exactly) structure of PyTorch.
You will need to implement various [layers](./layers), [losses](./losses) and [optimizers](./optimizers), and then finish with using them on a toy problem.

You should first complete layers, losses or optimizers (in any order).
**If you lint your code:** Do not run `inv lint --apply` before you finish all work, and pass all tests. It may remove important imports that are needed to finish the problem.

### Layers
This is the most basic building block of neural networks. You will implement a [linear](./layers/linear.py) layer, along with 2 activation functions: [ReLU](./layers/relu.py), [sigmoid](./layers/sigmoid.py) and a [softmax](./layers/softmax.py) function which we will need later.

These are all standalone, and tested so you can work on them and verify your work in any order.

### Losses
Another vital block of all of machine learning. As we've learned in class to train any ML algorithm we need an objective. This is also true for neural networks.
We will look at 2 loss functions: [mean squared error](./losses/MSE.py) and [cross entropy](./losses/CrossEntropy.py).

Mean squared error has been consistent theme throughout the class, and it is a really good for regression problems.
Cross entropy is a loss functions specialized for classification, where targets are integers indicating correct class.
It also requires predictions to be probabilities, for which we will use softmax function from layers section.

Because of different requirements for target shapes in the MSE and crossentropy, we will later use different dataloaders, and thus will need separate files for each loss function (see `crossentropy_search.py` and `mean_squared_error_search.py` functions).

Both of these are tested, and can be implemented in any order.

### Optimizers
Optimizers are crucial for performing gradient descent or other optimization algorithms.
Here we will only implement Stochastic Gradient Descent (SGD) algorithm.

Note that for each model parameter the gradients are already calculated, so you will only need to implement update rule (and figure out how to access parameters and gradients).

### Hyperparameter search
Finally you will use your implementations on a simple yet difficult problem.
Given a dataset representing XOR function (we treat positives as truth, negatives as false), you will try total of 10 different architectures (5 for each loss function) and determine which one performs the best.

To start look into `train` function in [train](./train.py) file.
Here you will build a training loop through the dataset.
At the end of each epoch, you will record running training loss, and validation loss, if validation loader has been provided.

Then see either [crossentropy_search.py](./crossentropy_search.py) or [mean_squared_error_search.py](mean_squared_error_search.py) files.
We recommend starting with `accuracy_score` functions, which should be a single for-loop through the dataloaders.

Then you should implement `crossentropy_parameter_search` (or MSE equivalent), which for each model architecture should call `train` function with correct optimizer, and record history of that model training (and the model itself).

Lastly, you will implement `main` function in each MSE and CrossEntropy files.
It should:

1. Call their respective `search` functions
2. Plot both training and validation per-epoch errors all on single plot (2 plots total; 1 plot per loss function; 10 lines on each plot).
3. Choose the best performing model. It should achieve the lowest validation loss at any point of the training.
4. Plot predictions of the best model on the test set (use `plot_model_guesses` function from [train file](./train.py))
5. Report accuracy of the best model on the test set

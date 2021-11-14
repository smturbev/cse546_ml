import torch
from torch import nn

from utils import problem


class SoftmaxLayer(nn.Module):
    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a softmax calculation.
        Given a matrix x (n, d) on each element performs:

        softmax(x) = exp(x_ij) / sum_k=0^d exp(x_ik)

        i.e. it first takes an exponential of each element,
            and that normalizes rows so that their L-1 norm is equal to 1.

        Args:
            x (torch.Tensor): More specifically a torch.FloatTensor, with shape (n, d).
                Input data.

        Returns:
            torch.Tensor: More specifically a torch.FloatTensor, also with shape (n, d).
                Each row has L-1 norm of 1, and each element is in [0, 1] (i.e. each row is a probability vector).
                Output data.

        Note:
            - For a numerically stable approach to softmax (needed for the problem),
                first subtract max of x from data (no need for dim argument, torch.max(x) suffices).
                This causes exponent to not blow up, and arrives to exactly the same answer.
            - YOU ARE NOT ALLOWED to use torch.nn.Softmax (or it's functional counterparts) in this class.
            - Make use of pytorch documentation: https://pytorch.org/docs/stable/index.html
        """
        raise NotImplementedError("Your Code Goes Here")

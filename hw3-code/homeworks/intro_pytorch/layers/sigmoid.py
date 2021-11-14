import torch
from torch import nn

from utils import problem


class SigmoidLayer(nn.Module):
    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a sigmoid calculation:
        Element-wise given x return 1 / (1 + e^(-x))

        Args:
            x (torch.Tensor): More specifically a torch.FloatTensor, with some shape.
                Input data.

        Returns:
            torch.Tensor: More specifically a torch.FloatTensor, with the same shape as x.
                Every negative element should be substituted with sigmoid of that element.
                Output data.

        Note:
            - YOU ARE NOT ALLOWED to use torch.nn.Sigmoid (or torch.nn.functional.sigmoid) in this class.
                YOU CAN however use other aliases of sigmoid function in PyTorch if you are able to find them in docs.
            - Make use of pytorch documentation: https://pytorch.org/docs/stable/index.html
        """
        raise NotImplementedError("Your Code Goes Here")

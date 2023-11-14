import torch
from typing import Optional, Callable
from pykeops.torch import LazyTensor


class GPUNadarayaWatson:

    def __init__(self, contexts: torch.Tensor, returns: torch.Tensor, lengthscale: float = None,
                 input_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> None:
        self.contexts = contexts if input_transform is None else input_transform(contexts)
        self.returns = returns
        self.input_transform = input_transform

        # KeOps allows to work with multi-dimensional outputs, i.e. (N, D). We hence broadcast one-dimensional
        # observations to the appropriate shape
        if len(self.returns.shape) == 1:
            self.returns = self.returns[:, None]

        if lengthscale is None:
            self.lengthscale = 2 * (torch.median(torch.nn.functional.pdist(contexts)) ** 2)
        else:
            self.lengthscale = 2 * (lengthscale ** 2)

    def predict_individual(self, x: torch.Tensor, with_gradient: bool = False):
        if len(x.shape) == 2 and x.shape[0] == 0:
            return torch.zeros((0, self.returns.shape[-1]), device=x.device)
        else:
            if self.input_transform is not None:
                x = self.input_transform(x)

            assert x.shape[-1] == self.contexts.shape[-1]

            # This allows for arbitrary batching dimensions at the beginning
            diffs = LazyTensor(x[..., None, :]) - LazyTensor(self.contexts[None, ...])
            reduce_dim = len(x.shape)
            log_activations = -(diffs ** 2).sum(dim=reduce_dim) / self.lengthscale
            try:
                prediction = log_activations.sumsoftmaxweight(LazyTensor(self.returns[None, ...]),
                                                              dim=len(log_activations.shape) - 1)
            except ValueError as e:
                prediction = log_activations.sumsoftmaxweight(LazyTensor(self.returns[None, ...]),
                                                              enable_chunks=False,
                                                              dim=len(log_activations.shape) - 1)

            if with_gradient:
                raise RuntimeError("not supported")
            else:
                return prediction

    def save(self, path):
        pass

    def load(self, path):
        pass

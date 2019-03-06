from mxnet.optimizer import Optimizer
from mxnet import ndarray
from mxnet.ndarray import zeros, sgd_update as upd, NDArray, zeros

class My_SGD(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_state(self, index, weight):
        return None

    def update(self,  indices, weights, grads, states):
        self._update_count(indices)
        lr = self._get_lr(indices)
        wd = self._get_wd(indices)
        kwargs = {'rescale_grad': self.rescale_grad}
        for weight, grad in zip(weights, grads):
            upd(weight, grad, out=weight, lr=lr, wd=wd, **kwargs)


class My_NAG(Optimizer):
    def __init__(self, momentum=0.0, **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum

    def create_state(self, index, weight):
        momentum = None
        if self.momentum != 0.0:
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype)
        return momentum

    def update(self, index, weight, grad, state):
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        grad = grad * self.rescale_grad
        state_2 = state
        state_2 *= self.momentum
        grad += wd * weight
        state_2 += grad
        grad += self.momentum * state_2
        weight += -lr * grad

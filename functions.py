from torch.autograd import Function

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x        # No Operation in the forward pass. Simply propagating x further

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: gradient so far as calculated by backprop, dL/dx
        output = - ctx.alpha * grad_output # Flipping the gradient by multiplying by alpha
        return output, None 
    

import torch
from torch.autograd import Function
from torch import nn


class BiMapFunction(Function):

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = (weight.double() @ input.double()) @ weight.double().t()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_variables
        grad_input = grad_weight = None
        input = input.double()
        if ctx.needs_input_grad[0]:
            grad_input = weight.t() @ grad_output @ weight

        if ctx.needs_input_grad[1]:
            e_grad = 2 * grad_output @ weight @ input
            e_grad = e_grad.sum(0)
            grad_weight = e_grad - e_grad @ weight.t() @ weight
        return grad_input, grad_weight



class BiMap(nn.Module):
    def __init__(self, input_features, output_features):
        super(BiMap, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        if output_features < input_features:
            self.weight = nn.Parameter(torch.Tensor(
                output_features, input_features))
            a = torch.rand(input_features, input_features)
            u, s, v = torch.svd(a@a.t())
            self.weight.data = u[:, :output_features].t()
            self.weight.data = torch.tensor(self.weight.data,dtype=torch.float64)
        else:
            self.weight = nn.Parameter(torch.Tensor(
                output_features, input_features))
            a = torch.rand(output_features, output_features)
            u, s, v = torch.svd(a @ a.t())
            self.weight.data = u[:, :input_features]
            self.weight.data = torch.tensor(self.weight.data, dtype=torch.float64)

    def forward(self, input):
        return BiMapFunction.apply(input, self.weight)


class BiMapgpuFunction(Function):

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = (weight.double() @ input.double()) @ weight.double().t()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_variables
        grad_input = grad_weight = None
        input = input.double()
        if ctx.needs_input_grad[0]:
            grad_input = weight.t() @ grad_output @ weight

        if ctx.needs_input_grad[1]:
            e_grad = 2 * grad_output @ weight @ input
            e_grad = e_grad.sum(0)
            grad_weight = e_grad - e_grad @ weight.t() @ weight
        return grad_input, grad_weight



class BiMapgpu(nn.Module):
    def __init__(self, input_features, output_features):
        super(BiMapgpu, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(
            output_features, input_features))
        a = torch.rand(input_features, input_features)
        u, s, v = torch.svd(a@a.t())
        self.weight.data = u[:, :output_features].t()
        self.weight.data = torch.tensor(self.weight.data,dtype=torch.float64,device ="cuda")
    def forward(self, input):
        return BiMapgpuFunction.apply(input, self.weight)


class BiMapSFunction(Function):

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = (weight.double() @ input.double()) @ weight.double().t()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_variables
        grad_input = grad_weight = None
        input = input.double()
        if ctx.needs_input_grad[0]:
            grad_input = weight.t() @ grad_output @ weight

        if ctx.needs_input_grad[1]:
            e_grad = 2 * grad_output @ weight @ input
            e_grad = e_grad.sum(0)
            grad_weight = e_grad - e_grad @ weight.t() @ weight
        return grad_input, grad_weight



class BiMapS(nn.Module):
    def __init__(self, input_features, output_features):
        super(BiMapS, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(
            output_features, input_features))
        a = torch.rand(input_features, input_features)
        u, s, v = torch.svd(a@a.t())
        self.weight.data = u[:, :output_features].t()
        self.weight.data = torch.tensor(self.weight.data,dtype=torch.float64)
    def forward(self, input):
        return BiMapSFunction.apply(input, self.weight)

class FrMap(nn.Module):
    def __init__(self, input_features, output_features):
        super(FrMap, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(
            output_features, input_features))
        a = torch.rand(input_features, input_features)
        u, s, v = torch.svd(a @ a.t())
        self.weight.data = u[:, :output_features].t()
        self.weight.data = torch.tensor(self.weight.data, dtype=torch.float64)

    def forward(self, input):
        return FrMapFunction.apply(input, self.weight)

class FrMapFunction(Function):

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = weight.double() @ input.double()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_variables
        grad_input = grad_weight = None
        input = input.double()
        if ctx.needs_input_grad[0]:
            grad_input = weight.t() @ grad_output
            grad_weight = grad_output @ torch.permute(input,[0,2,1])

        return grad_input, grad_weight

class FrMapmul(nn.Module):
    def __init__(self, input_features, output_features,inchannel):
        super(FrMapmul, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.inchannel = inchannel
        self.weight = nn.Parameter(torch.Tensor(inchannel,
            output_features, input_features))
        a = torch.rand(inchannel,input_features, input_features)
        u, s, v = torch.svd(a @ a.permute(0,2,1))
        self.weight.data = u[:, :,:output_features].permute(0,2,1)
        self.weight.data = torch.tensor(self.weight.data, dtype=torch.float64)

    def forward(self, input):
        return FrMapmulFunction.apply(input, self.weight)

class FrMapmulFunction(Function):

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        if len(input.size()) == 3:
            output = (weight.unsqueeze(0)@input.unsqueeze(1))
        else:
            output = (weight.unsqueeze(0)@input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_variables
        grad_input = grad_weight = None
        input = input.double()
        if ctx.needs_input_grad[0]:
            grad_input = weight.permute(0,2,1) @ grad_output
            grad_weight = grad_output @ input.permute(0,1,3,2)
        else:
            grad_input = weight.permute(0, 2, 1) @ grad_output
            grad_weight = grad_output @ input.unsqueeze(1).permute(0,1,3,2)
        grad_weight = grad_weight.mean(0)
        # grad_input = grad_input.sum(0)
        return grad_input, grad_weight




class BiMapmulFunction(Function):

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        if len(input.size()) == 3:
            weight = weight.unsqueeze(0)
            output = (weight @ input.unsqueeze(1)) @ weight.permute(0, 1, 3, 2)
        else:
            output = (weight.double() @ input.double()) @ weight.double().permute(0,2,1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_variables
        grad_input = grad_weight = None
        input = input.double()
        if ctx.needs_input_grad[0]:
            grad_input = weight.permute(0,2,1)@grad_output@weight

        if ctx.needs_input_grad[1]:
            if len(input.size()) == 3:
                e_grad = 2 * grad_output @ weight @ input.unsqueeze(1)
                e_grad = e_grad.sum(0)  # 因为梯度必须有一个，不能有batchsize个
                grad_weight = e_grad - e_grad @ weight.permute(0, 2, 1) @ weight
            else:
                e_grad = 2 * grad_output @ weight @ input
                e_grad = e_grad.sum(0)#因为梯度必须有一个，不能有batchsize个
                grad_weight = e_grad - e_grad @ weight.permute(0,2,1) @ weight
        return grad_input, grad_weight



class BiMapmul(nn.Module):
    def __init__(self, input_features, output_features,inchannel):
        super(BiMapmul, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(
            output_features, input_features,inchannel))

        a = torch.rand(inchannel,input_features, input_features)
        u, s, v = torch.svd(a@a.permute(0,2,1))
        self.weight.data = u[:, :,:output_features].permute(0,2,1)
        self.weight.data = torch.tensor(self.weight.data,dtype=torch.float64)
    def forward(self, input):
        return BiMapmulFunction.apply(input, self.weight)



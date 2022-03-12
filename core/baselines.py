import torch


# Conditional Regression Analysis for Recurrence Time Data
# conditional regression forest
# model: lambda_0j(t-t_j)exp(z1(t)beta + z2(t)gamma)
class Recur(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
    
    def forward():
        
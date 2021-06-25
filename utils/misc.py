import numpy as np
import torch

def identify_strange_input_values(input, threshold=1000000):

    mins, maxs = torch.min(input).item(), torch.max(input).item()
    isnan_identifier = torch.isnan(input)
    isinf_identifier = torch.isinf(input)

    assertion1 = abs(mins) > threshold
    assertion2 = abs(maxs) > threshold
    assertion3 = True in isnan_identifier
    assertion4 =  True in isinf_identifier

    if True in [assertion1, assertion2, assertion3, assertion4]:
        print('\n PROBLEM IN THE INPUTS')


def identify_strange_output_values(output):

    mins = torch.min(output, axis=-1).values.numpy()
    maxs = torch.max(output, axis=-1).values.numpy()

    print(' MINs {} \n and \n MAXs {}'.format(mins, maxs))


output = torch.randn((4, 8))
identify_strange_output_values(output)
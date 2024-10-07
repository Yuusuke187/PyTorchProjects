# We will use this file to create a simple Support Vector Machine model
# and train it

# An SVM is used to find an optimal line or hyperplane that maximizes the
# disstance between each class in an N-dimensional space

import torch
import torch.nn as nn
import torch.optim as optim

class SVM(nn.module):
    def __init__(self, input_size):
        # Have the class initialize itself
        super(SVM, self).__init__()
        
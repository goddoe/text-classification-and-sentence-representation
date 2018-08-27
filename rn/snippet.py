import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# prepare coord tensor


def cvt_coord(i):
    return [(i / 5 - 2) / 2., (i % 5 - 2) / 2.]


batch_size = 3

coord_tensor = torch.FloatTensor(batch_size, 25, 2)
coord_tensor = Variable(coord_tensor)
np_coord_tensor = np.zeros((batch_size, 25, 2))

for i in range(25):
    np_coord_tensor[:, i, :] = np.array(cvt_coord(i))
coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

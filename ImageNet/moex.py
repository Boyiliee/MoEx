import torch
import torch.nn.functional as F


def moex(x, ex_index, norm_type, epsilon=1e-5, positive_only=False):
    '''MoEx operation
    inputs:
        x: feature map of shape (batch_size, channels, height, width)
        ex_index: the indices of the examples that provide the new moments
        norm_type: normalization to compute the moments
        epsilon: small constant to stablize the computation of standard deviations
        positive_only: only compute the statistics across the moments
    output: new feature map with exchanged moments of shape (batch_size, channels, height, width)
    '''
    dtype = x.dtype
    x = x.float()

    B, C, H, W = x.shape
    if norm_type == 'bn':
        norm_dims = [0, 2, 3]
    elif norm_type == 'in':
        norm_dims = [2, 3]
    elif norm_type == 'ln':
        norm_dims = [1, 2, 3]
    elif norm_type == 'pono':
        norm_dims = [1]
    elif norm_type.startswith('gn'):
        if norm_type.startswith('gn-d'):
            # gn-d4 means GN where each group has 4 dims
            G_dim = int(norm_type[4:])
            G = C // G_dim
        else:
            # gn4 means GN with 4 groups
            G = int(norm_type[2:])
            G_dim = C // G
        x = x.view(B, G, G_dim, H, W)
        norm_dims = [2, 3, 4]
    elif norm_type.startswith('gpono'):
        if norm_type.startswith('gpono-d'):
            # gpono-d4 means GPONO where each group has 4 dims
            G_dim = int(norm_type[len('gpono-d'):])
            G = C // G_dim
        else:
            # gpono4 means GPONO with 4 groups
            G = int(norm_type[len('gpono'):])
            G_dim = C // G
        x = x.view(B, G, G_dim, H, W)
        norm_dims = [2]
    else:
        raise NotImplementedError(f'norm_type={norm_type}')
        
    if positive_only:
        x_pos = F.relu(x)
        s1 = x_pos.sum(dim=norm_dims, keepdim=True)
        s2 = x_pos.pow(2).sum(dim=norm_dims, keepdim=True)
        count = x_pos.gt(0).sum(dim=norm_dims, keepdim=True)
        count[count == 0] = 1 # deal with 0/0
        mean = s1 / count
        var = s2 / count - mean.pow(2)
        std = var.add(epsilon).sqrt()
    else:
        mean = x.mean(dim=norm_dims, keepdim=True)
        std = x.var(dim=norm_dims, keepdim=True).add(epsilon).sqrt()
    swap_mean = mean[ex_index]
    swap_std = std[ex_index]
    # output = (x - mean) / std * swap_std + swap_mean
    # equvalent but more efficient
    scale = swap_std / std
    shift = swap_mean - mean * scale
    output = x * scale + shift
    return output.view(B, C, H, W).to(dtype)
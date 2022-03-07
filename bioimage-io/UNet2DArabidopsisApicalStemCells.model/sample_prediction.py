import numpy as np
import torch
from unet import UNet2D

with torch.no_grad():
    state = torch.load('confocal_pnas_2d.pytorch')

    net = UNet2D(in_channels=1,
                 out_channels=1,
                 f_maps=32,
                 testing=True)

    net.load_state_dict(state)

    # load and normalize the input
    im = np.load('test_input.npy')
    im -= im.mean()
    im /= im.std()

    # forward pass
    inp = torch.from_numpy(im)
    out = net(inp)
    out = out.cpu().numpy()

    # compare with test_output
    test_out = np.load('test_output.npy')
    assert np.allclose(out, test_out)


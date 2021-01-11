import numpy as np
import torch
from unet import UNet3D

with torch.no_grad():
    state = torch.load('unet3d-lateral-root-lightsheet-ds1x.pytorch')

    net = UNet3D(in_channels=1,
                 out_channels=1,
                 f_maps=32,
                 testing=True)

    net.load_state_dict(state)

    # load and normalize the input
    im = np.load('test_input.npz')
    im = im['arr_0'].astype('float32')
    im -= im.mean()
    im /= im.std()

    # forward pass
    inp = torch.from_numpy(im)
    out = net(inp)
    out = out.cpu().numpy()

    # compare with test_output
    test_out = np.load('test_output.npz')
    test_out = test_out['arr_0']
    assert np.allclose(out, test_out)

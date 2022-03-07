import numpy as np
import torch
from unet import UNet3D

with torch.no_grad():
    state = torch.load('unet3d-lateral-root-nuclei-lightsheet-ds1x.pytorch')

    net = UNet3D(in_channels=1,
                 out_channels=2,
                 f_maps=32,
                 testing=True)

    net.load_state_dict(state)

    # with h5py.File('sample_raw.h5', 'r') as f:
    #     raw = f['raw'][10:110, :128, :128]
    #     im = raw[None, None, ...]
    #     im = im.astype('float32')
    #     np.save('test_input.npy', im)

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

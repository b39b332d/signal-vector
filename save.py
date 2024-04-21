import numpy as np
import torch
import model.model as module_arch
X=torch.tensor(np.zeros([1,36,256]), dtype=torch.float)
model = module_arch.ConvNet1D()

# model = module_arch.Net1D(
#     in_channels=24,
#     base_filters=100,
#     ratio=1.0,
#     filter_list = [64, 128, 128, 256,256,512],
#     m_blocks_list = [4, 4, 6, 6,8,8],
#     kernel_size=16,
#     stride=2,
#     groups_width=16,
#     verbose=False,
#     n_classes=24)
checkpoint = torch.load("model_save/model_20240419_120154_179_0.37475821375846863")
model.load_state_dict(checkpoint)
y=model(X)
model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('model_scripted2.pt') # Save
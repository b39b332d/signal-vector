import numpy as np
import torch
import model.model2 as module_arch
X=torch.tensor(np.zeros([1,39,256]), dtype=torch.float)
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
checkpoint = torch.load("/home/hexingyan/signal_vector/model_save/1t_28_202626_3_0.5644904375076294")
model.load_state_dict(checkpoint)
y=model(X)
model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('model_scripted_all.pt') # Save
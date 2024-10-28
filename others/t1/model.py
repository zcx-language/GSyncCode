import torch
from DHN import DHN
from SteganoGAN.steganogan.encoders import DenseEncoder
from config import CocoConfig
from decoder import Decoder
from mask_rcnn.model import MaskRCNN
import numpy as np
import torch.nn as nn
import time
from torchinfo import summary
from typing import List, Tuple


msg_len = 50
img_size = 400
class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_SHAPE = [3, 512, 512]
    
# encoder = DenseEncoder(msg_len, 64)

# masker = MaskRCNN(model_dir="./output", config=InferenceConfig()).cuda()


# msg = torch.randn(1, msg_len)
# x = torch.randn(1, 3, img_size, img_size)

# msg = msg.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, img_size, img_size)
# print(encoder(x,msg).shape)
# x = np.random.randn(400,400, 3)
# print(masker.detect([x]))


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DenseEncoder(msg_len, 64).cuda()
        self.masker = MaskRCNN(model_dir="./output", config=InferenceConfig()).cuda()
        self.dhn = DHN().cuda()
        self.decoder = Decoder().cuda()
        pass

    def forward(self, a):
        msg = torch.randn(1, msg_len).cuda()
        x = torch.randn(1, 3, img_size, img_size).cuda()
        msg = msg.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, img_size, img_size)
        self.encoder(x, msg)
        self.masker.detect([np.random.rand(400, 400, 3)])
        self.dhn(torch.randn(2, 2, 128, 128).cuda())
        self.decoder(x)
        pass
    pass


def measure_gpu_memory_and_time(model, input_tensor, device='cuda'):
    # Move the model and input tensor to GPU
    model.to(device)
    if isinstance(input_tensor, (List, Tuple)):
        input_tensor = tuple([t.to(device) for t in input_tensor])
    else:
        input_tensor = input_tensor.to(device)
    
    model_info = summary(model, input_data=input_tensor, device=device, mode='eval', verbose=0)
    total_bytes = model_info.to_megabytes(model_info.total_input + model_info.total_output_bytes + model_info.total_param_bytes)

    # Warm-up
    with torch.no_grad():
        if isinstance(input_tensor, (List, Tuple)):
            _ = model(*input_tensor)
        else:
            _ = model(input_tensor)
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        if isinstance(input_tensor, (List, Tuple)):
            _ = model(*input_tensor)
        else:
            _ = model(input_tensor)
    end_time = time.time()
    
    inference_time = (end_time - start_time) * 1000  # Convert to ms

    print(f"Total Memory Usage: {total_bytes:.2f} MB")
    print(f"Inference Time: {inference_time:.4f} ms")

if __name__ == '__main__':
    encoder = DenseEncoder(msg_len, 64).cuda()
    locator = MaskRCNN(model_dir="./output", config=InferenceConfig()).cuda()
    decoder = Decoder().cuda()
    dhn = DHN().cuda()

    msg = torch.randn(1, msg_len).cuda()
    x = torch.randn(1, 3, img_size, img_size).cuda()
    msg = msg.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, img_size, img_size)
    # self.encoder(x, msg)
    # self.masker.detect([np.random.rand(400, 400, 3)])
    # self.dhn(torch.randn(2, 2, 128, 128).cuda())
    # self.decoder(x)

    measure_gpu_memory_and_time(encoder, (torch.randn(1, 3, 400, 400), torch.randn(1, 50, 400, 400)))
    measure_gpu_memory_and_time(locator, torch.randn(1))
    measure_gpu_memory_and_time(dhn, torch.randn(2, 2, 128, 128).cuda())
    measure_gpu_memory_and_time(decoder, torch.randn(1, 3, 256, 256))
    # from torchvision.models import maskrcnn_resnet50_fpn


# -*- coding: utf-8 -*-
import torch
from torchvision import datasets, transforms
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=64, shuffle=False, num_workers=1, pin_memory=True)


G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
engine = trt.utils.load_engine(G_LOGGER, 'torch_mnist.engine')
runtime = trt.infer.create_infer_runtime(G_LOGGER)
context = engine.create_execution_context()


# run on one test sample
img, target = next(iter(test_loader))
img = img.numpy()[0]
target = target.numpy()[0]


# Allocate the memory on the GPU and allocate memory on the CPU to hold results after inference
# The size of the allocations is the size of the input and expected output * the batch size
h_input = img.ravel()
h_output = np.empty(10, dtype=np.float32)
# allocate device memory

d_input = cuda.mem_alloc(1 * h_input.size * h_input.dtype.itemsize)
d_output = cuda.mem_alloc(1 * h_output.size * h_output.dtype.itemsize)

# The engine needs bindings provided as pointers to the GPU memory
# PyCUDA lets us do this for memory allocations by casting those allocations to ints
bindings = [int(d_input), int(d_output)]

# Create a CUDA stream to run inference in
stream = cuda.Stream()

# Transfer the data to the GPU, run inference, and then copy the results back
# transfer input data to device
cuda.memcpy_htod_async(d_input, h_input, stream)
# execute model
context.enqueue(1, bindings, stream.handle, None)
# transfer predictions back
cuda.memcpy_dtoh_async(h_output, d_output, stream)
# synchronize threads
stream.synchronize()

# Now that you have the results, run ArgMax to get a prediction
print("True label: " + str(target))
print("Prediction: " + str(np.argmax(h_output)))
print(h_output)


context.destroy()
engine.destroy()
runtime.destroy()












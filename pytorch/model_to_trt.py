# -*- coding: utf-8 -*-
import torch
from torchvision import datasets, transforms
from net_definition import Net
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


model = Net()
model.cuda()

# load saved model
model = torch.load('../model/torch_000.model')
weights = model.state_dict()
print(weights.keys())
# print(weights['fc2.bias'])


# TensorRT expects weights in NCHW format, therefore, if your framework uses another format,
# you may need to pre-process your weights before flattening


# Start converting the model to TensorRT by first creating a builder and a logger for the build process
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
builder = trt.infer.create_infer_builder(G_LOGGER)
network = builder.create_network()


# Name for the input layer, data type, tuple for dimension
data = network.add_input("data", trt.infer.DataType.FLOAT, (1, 28, 28))
assert data

conv1_w = weights['conv1.weight'].cpu().numpy().reshape(-1)
conv1_b = weights['conv1.bias'].cpu().numpy().reshape(-1)
conv1 = network.add_convolution(data, 20, (5, 5), conv1_w, conv1_b)
assert conv1
conv1.set_stride((1, 1))

pool1 = network.add_pooling(conv1.get_output(0), trt.infer.PoolingType.MAX, (2, 2))
assert pool1
pool1.set_stride((2, 2))

conv2_w = weights['conv2.weight'].cpu().numpy().reshape(-1)
conv2_b = weights['conv2.bias'].cpu().numpy().reshape(-1)
conv2 = network.add_convolution(pool1.get_output(0), 50, (5, 5), conv2_w, conv2_b)
assert conv2
conv2.set_stride((1, 1))

pool2 = network.add_pooling(conv2.get_output(0), trt.infer.PoolingType.MAX, (2, 2))
assert pool2
pool2.set_stride((2, 2))

fc1_w = weights['fc1.weight'].cpu().numpy().reshape(-1)
fc1_b = weights['fc1.bias'].cpu().numpy().reshape(-1)
fc1 = network.add_fully_connected(pool2.get_output(0), 500, fc1_w, fc1_b)
assert fc1

relu1 = network.add_activation(fc1.get_output(0), trt.infer.ActivationType.RELU)
assert relu1

fc2_w = weights['fc2.weight'].cpu().numpy().reshape(-1)
fc2_b = weights['fc2.bias'].cpu().numpy().reshape(-1)
fc2 = network.add_fully_connected(relu1.get_output(0), 10, fc2_w, fc2_b)
assert fc2

# Mark you output layer
fc2.get_output(0).set_name("prob")
network.mark_output(fc2.get_output(0))


# Set the rest of the parameters for the network (max batch size and max workspace) and build the engine
builder.set_max_batch_size(1)
builder.set_max_workspace_size(1 << 20)

engine = builder.build_cuda_engine(network)
network.destroy()
builder.destroy()


# Create the engine runtime and generate a test case from the torch data loader
runtime = trt.infer.create_infer_runtime(G_LOGGER)
img, target = next(iter(test_loader))
img = img.numpy()[0]
target = target.numpy()[0]


# Create an execution context for the engine
context = engine.create_execution_context()


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


# We can also save our engine to a file to use later
trt.utils.write_engine_to_file('torch_mnist.engine', engine.serialize())


# Finally, clean up your context, engine, and runtime
context.destroy()
engine.destroy()
runtime.destroy()




















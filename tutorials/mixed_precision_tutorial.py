import torch, time, gc
import numpy as np

start_time = time
memory_usage = []

def start_timer():
    global start_time, memory_usage
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()
    memory_usage = []

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    avg_memory_usage = np.mean(memory_usage)
    print(local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))
    print("Average memory used by tensors = {:.0f} bytes".format(avg_memory_usage))
    print("\n")

def make_model(in_size, out_size, num_layers):
    layers = []
    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(in_size, out_size))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(in_size, out_size))
    model = torch.nn.Sequential(*tuple(layers)).cuda()
    return model

batch_size = 512
in_size = 4096
out_size = 4096
num_layers = 3
num_batches = 50
epochs = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)

data = [torch.randn(batch_size, in_size) for _ in range(num_batches)]
targets = [torch.randn(batch_size, out_size) for _ in range(num_batches)]

loss_fn = torch.nn.MSELoss().cuda()

net = make_model(in_size, out_size, num_layers)
net.to(device)
opt = torch.optim.SGD(net.parameters(), lr=0.001)

start_timer()
for epoch in range(epochs):
    for input, target in zip(data, targets):
        output = net(input)
        loss = loss_fn(output, target)
        loss.backward()
        opt.step()
        opt.zero_grad()
        memory_usage.append(torch.cuda.memory_allocated())
end_timer_and_print("Default precision: ")

# ----------------------------------------------------------

use_amp = True
net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

start_timer()
for epoch in range(epochs):
    for input, target in zip(data, targets):
        with torch.autocast(device_type=device, dtype=torch.float16):
            output = net(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)

        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        memory_usage.append(torch.cuda.memory_allocated())
end_timer_and_print("Mixed precision: ")




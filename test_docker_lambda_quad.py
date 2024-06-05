import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensor_cpu = torch.tensor([[1., 2.], [3., 4.]])
print(tensor_cpu)

tensor_gpu = tensor_cpu.to(device)

print(tensor_gpu)

tensor_gpu_2 = torch.tensor([[0, 2.], [7., 4.5]]).to(device)

result_gpu = torch.matmul(tensor_gpu, tensor_gpu_2)

result_cpu = result_gpu.to("cpu")
print(result_cpu)
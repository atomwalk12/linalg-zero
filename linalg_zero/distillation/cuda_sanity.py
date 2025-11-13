# cuda_sanity.py
import multiprocessing as mp

import torch


def worker():
    torch.cuda.set_device(0)
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("child OK:", torch.cuda.get_device_name(0), y.shape)


if __name__ == "__main__":
    print("cuda_available:", torch.cuda.is_available())
    p = mp.get_context("spawn").Process(target=worker)
    p.start()
    p.join()
    print("rc", p.exitcode)

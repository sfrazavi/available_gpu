import subprocess as sp
import numpy as np
import os

total_free = 11019
threshold = total_free // 2
# threshold = 500 # temporary 500M

def waiting():

    def available_memory():
        cmd = "nvidia-smi --query-gpu=memory.free --format=csv"
        output = sp.check_output(cmd.split())
        output = output.decode('ascii')
        output = output.split("\n")[1:-1]
        output = [int(x.split()[0]) for x in output]
        return output

    while (True):
        memory_free_values = available_memory()
        available = np.where(np.array(memory_free_values) >= threshold)
        if not available:
            pass
        else:
            return available[0]

def run(command):
    available = waiting()  # index of empty gpus
    print("available = ", available)
    os.system("CUDA_VISIBLE_DEVICES={} {}".format(available[0], command))

command = "python3 main.py"
run(command)


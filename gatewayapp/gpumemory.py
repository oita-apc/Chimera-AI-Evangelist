# Copyright (C) 2020 - 2022 APC, Inc.

import subprocess
import os

g_pid = str(os.getpid()).encode('utf8')

def get_gpu_memory_usage():
    """
    Grab nvidia-smi output and return a dictionary of the memory usage.
    """
    data = {}

    try:
        p = subprocess.Popen(['nvidia-smi -q'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # output, err = p.communicate()
        used_name = ''
        total_countLine = 0
        process_countLine = 0
        for line in p.stdout:
            if b"FB Memory Usage" in line:
                # print(line.decode('utf8'))
                total_countLine += 1
                continue
            if total_countLine > 0 and total_countLine <= 3:
                line = line.decode('utf8')
                tempArr = [x.strip() for x in line.split(':')]
                # print(tempArr)
                data[tempArr[0]] = tempArr[1]
                total_countLine += 1
                if(total_countLine == 3):
                    used_name = tempArr[0]
                continue
            if (b"Process ID" in line and g_pid in line) or (process_countLine > 0 and process_countLine <= 2):
                process_countLine += 1
                continue
            if process_countLine == 3:
                line = line.decode('utf8')
                tempArr = [x.strip() for x in line.split(':')]
                data[used_name] = tempArr[1]
                process_countLine += 1
            if (process_countLine > 3):
                break
            
    except (OSError, ValueError) as e:
        pass
    return data 

if __name__ == '__main__':
    import json
    gpuinfo = get_gpu_memory_usage()
    print("gpuinfo", json.dumps(gpuinfo))
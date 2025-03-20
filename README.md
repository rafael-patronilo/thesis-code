# How to run

1. Install docker
2. Clone the repository
3. Run the `./docker-build` bash script
4. Run `./docker-run IMG python src/__init__.py --help`.

Docker scripts are written in bash and expect a unix environment 
(they were run in a WSL2 Ubuntu environment). Both scripts
handle the removal of dangling images/containers.

The docker-run script mounts the appropriate directories. 
The script will attempt to reserve all available graphics cards for the docker container. 
This may lead to issue if you have multiple graphics cards and a few working on other tasks. 
There is currently not an option to configure this but it should be easy to edit the script.

# Repository structure

Directory `src/core` contains tools are agnostic to the dataset used.
Remaining code is mostly specific to the XTRAINS dataset. 

# Quality Disclaimer

Due to time constraints during development, 
I acknowledge that the code provided in this repository 
is bellow my personal quality standards. 

As I developed this project, I came to understand that 
many of my initial design decisions had made the 
implementation of certain features more difficult, and 
therefore severely impacted code quality.
I came to understand that I should have made use of more 
packages for deep learning, such as Keras, which would have 
provided more robust solutions to certain problems I approached.


While at some points I went back and corrected some design choices, 
I did not rewrite the whole codebase. 
This was mostly due mostly to time constraints but
also to preserve functionality of earlier experiments.

I provide this repository with the goal of facilitating the 
reproduction of the results in my thesis. I cannot in good faith 
recommend its direct use as it is for future projects, although you are free to 
extract any parts that you find useful.

The code is under-documented, and there are not as many unit tests as
I would desire. There is however a significant amount of descriptive logging which
I throuroughly examing during development to assert everything had
the expected behaviour.

# System details
The models were trained in a Windows 11 computer within docker containers.

Training epochs with the XTRAINS dataset took a bit under 15 minutes each.

## Hardware details

| **Device**        | **Details**                                                      |
|-------------------|------------------------------------------------------------------|
| **GPU**           |`NVIDIA GeForce GTX 1060 6GB`                                     |
| **CPU**           |`Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz` (16 threads, 8 cores) |
| **RAM**           | 32 GB                                                            |



 

## `nvidi-smi`
```
Thu Mar 20 09:38:10 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.02              Driver Version: 560.94         CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce GTX 1060 6GB    On  |   00000000:01:00.0  On |                  N/A |
|  0%   36C    P8              8W /  180W |     804MiB /   6144MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

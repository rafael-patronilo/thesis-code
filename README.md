# How to run

1. Install docker
2. Clone the repository
3. Run the `./docker-build` bash script
4. Run `./docker-run IMG python src/__init__.py --help`.

Docker scripts are written in bash and expect a unix environment 
(they were run in a WSL2 Ubuntu environment). Both scripts
handle the removal of dangling images/containers.

The docker-run script mounts the appropriate directories.

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
I did not rewrite the codebase due mostly to time constraints but
also to preserve functionality of earlier experiments.

I provide this repository with the goal of facilitating the 
reproduction of the results in my thesis. I cannot in good faith 
recommend its direct use for future projects, although you are free to 
extract any parts that you find useful.

The code is under-documented, and there are not as many unit tests as
I would desire. There is however a significant amount of logging which
I used to both debug the code during development and assert it had
the expected behaviour.
# Simulation codes and data for "Crossover effects and nontrivial scaling on the radial growth with competitive dynamics"

## Contents

There are three python codes files and text files of data of figures. Simulations and results of the paper can be obtained by these codes. Some infomations is as follow.

### Running environment

All of three codes work with Python 3.10 and library including numpy, scipy, matplotlib and tqdm. We integrate simulation and data processing into the one program and output results. Therefore, it only needs to be run once to obtain the data results.

The three code files represent the simulations of the radial EW-KPZ, radial RDSR-RSOS, and radial RDSR-BD, respectively. Detailed descriptions of the algorithms can be found in our paper.

### The radial EW-KPZ equation

The program "RKPZ.py" is used to simulate the radial EW-KPZ equation. It has seven parameters and two functions. The meanings of the parameters are shown in the table below.

| Parameter        | Meaning                                |
| ---------------- | -------------------------------------- |
| Interface_length | the length of interface (lattice size) |
| time             | the total time steps                   |
| D                | the diffusion coefficient              |
| F                | the nonlinear (lateral) coefficient    |
| R0               | the initial interface radius           |
| dtime            | the time difference interval           |
| repeat           | the repeat times                       |

The function "mainloop" will return the surface height for each time step by finite diference. Then the program will calculate the interface width *W*. Another function "Glt" is used to calculate the height correlation function *G*. After data processing, the results will be saved as a text file and show a preview figure.

### The radial  RDSR-RSOS and RDSR-BD models

In these programs (RDSR-RSOS.py and RDSR-BD.py), there are two functions and four parameters. The function "growth" is the main code of the growth process, and it will return the interface width *W* and the surface height for each time. It need three parameter "probability", "SL" and "Time", which means the probability of choosing growth method, the initial substrate length and the total time step, respectively. And another parameter "Repeat" means the repeat times of the program. The function "Glt" is used to calculate height correlation function *G*, and it need input surface height data and will return value of *G*. Finally, the program will save the data of *logW* and *logG*. And we also can use matplotlib to show the morphology of the model.

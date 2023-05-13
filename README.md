# PHYS230_Project
This repository contains material that I prepared for the computational course project in Phys 230 course that took place during Spring 2023 at UC Merced. There are few codes written in python by me as a beginner which explain the results I presented for my project on 'Continuum modelling: Liquid-liquid phase separation'.
There are two documents in the repository other than the codes:

1. Phys_230_final_presentation.pdf which holds my non-modified presentation, along with several additional slides at the end to showcase new trial results.

2. Phys_230_project_Arnab_Roy.pdf which is a combination of my HW 3 submission (without modifications) and a brief summary describing the state of the project at the end with possible future steps.

Each .py file in the repository has the following purpose/instructions:

(1) **FE_RK4_Numba_Combined_droplet_random_init.py**

A two component liquid-liquid phase separation model represented in Section 1 of the Phys_230_project.pdf file. It consists of two different time evolution methods: Forward Euler and 4th order Runge Kutta. Note: density array ranges from -1 to 1.

The instructions to run the code consist of: 

a. Preferred model parameters are specified on lines 21-30. Be careful deciding initial density and variance, if every component of the array is not within -1 or 1 the code will run infinitely.

c. Deciding if you want to keep track of simulation time throughout ('yes' keeps track, everything else doesn't); replace line 35 with tcls = 'yes' (steps between each calculation on line 37).

d. numba jit decorators are used at the beginning of every function object i.e. every short chunk or blocks starting with 'def' to speed up the computation.

e. You can pick which time evolution scheme you want, if line 115(116) is uncommented you are using Forward Euler(Runge Kutta). Can comment/uncomment respectively to change this.

f. Decide if you want to print out the maximum and minimum value of the final u array, that can be done by uncommenting lines 132-133.

(2) **Cahn_Hilliard_trial1_FE_RK4.py**

This is the very first code I wrote in this project. It consists of both Forward Euler and RK4 methods with random initialization. It is a much simpler code. I used np.roll command to create the 2D discrete laplacian for my model. Two key points here:

a. Lines 38-43 can be uncommented if you want Forward Euler time update. Currently RK4 time update is running here in lines 45-55.

b. Initial time counter starts at line 13 and final counter is performed at line 85. Hence the parameter t at line 86 is giving us the actual real time the simulation took to complete.

c. At first this code was written for only the cahn-hilliard dynamics without any reaction. If you want to include the reaction source term s, you have to uncomment lines 30 and 32 and don't forget to comment line 31 then.

d. Choose parameters for systems at lines 65-76.

N.B: I would suggest to start with this simplest code at first.

(3) **Cahn_Hilliard_single_droplet_trial1_with_reaction.py**

This code was written for single droplet initialization using interpolating tanh function (without reaction) and then also reaction term was added.

a. Single droplet initialization is actually happening in lines 67-70 but before that look at lines 60-64 where I am creating a circular region at a particular place on the grid with radius R (line 57) by numpy array operation. Then tanh function was used to interpolated between region of +1 inside the circular droplet to region of -1 outside via a thin interface.

b.Look carefully u_array is being initialized at line 84 by calling the 'initialize' function that we wrote earlier on lines 67-70. So, if line 84 remains commented you will repent. 

c. Choose parameters for systems at lines 71-81. Rest everything is similar to earlier code and add the reaction term say was as before.

(4) **time_comparison_code.py**

This simple short code is written to compare simulation time difference between Forward Euler and Runge-Kutta(4th order) methods.

Final Note: I have tried two other codes along with these, one for two droplet initialization with reactions, and another one for making animation or movie of the Cahn-Hilliard phase separation dynamics. But both of those are having some minor glitches. Hopefully I would be able to add them up later here.



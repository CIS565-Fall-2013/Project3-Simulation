-------------------------------------------------------------------------------
<center>CIS565: Project 3: Simulations
-------------------------------------------------------------------------------
<center>Fall 2013
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
<center>INTRODUCTION:
-------------------------------------------------------------------------------
This project is a two part project. The first project a 3D visualization of an N-Body system simulated using CUDA and OpenGL shaders. The second part was a simple flocking simulation based largely on this paper: http://www.red3d.com/cwr/boids/

-------------------------------------------------------------------------------
<center>PART I: N-BODY SIM
-------------------------------------------------------------------------------

This part of the project is rather straightforward. Here are a few screen shots.

<center>![planet sim](https://raw.github.com/josephto/Project3-Simulation/master/Part1/PlanetSim1.png "screen shot")

<center>![planet sim](https://raw.github.com/josephto/Project3-Simulation/master/Part1/PlanetSim2.png "screen shot")

-------------------------------------------------------------------------------
<center>PART II: FLOCKING SIM
-------------------------------------------------------------------------------

For this part of the project, I implemented a simple flocking simulation. I added in alignment, separation, cohesion, and a spherical boundary avoidance force that kept the particles within the frame of the window. Here are a few screenshots:

<center>![flocking sim](https://raw.github.com/josephto/Project3-Simulation/master/Part2/flocking1.png "screen shot")

<center>![flocking sim](https://raw.github.com/josephto/Project3-Simulation/master/Part2/flocking2.png "screen shot")

Here's a video of my flocking simulation in action. I apologize for the poor quality: http://www.youtube.com/watch?v=Ypkwl0yjSD8

-------------------------------------------------------------------------------
<center>PERFORMANCE REPORT:
-------------------------------------------------------------------------------

Here's a table with some performance analysis that I conducted on my code. I recorded how many seconds for one update/frame of the simulation to occur when I used naive force accumulation over all the planets in the system compared to using shared memory.

Number of Planets | Without Shared Memory | With Shared Memory
------------------|------------------------|---------------------
1000    | 0.001 sec  | 0.001 sec
2000    | 0.003 sec  | 0.002 sec
3000    | 0.005 sec  | 0.004 sec
4000    | 0.008 sec  | 0.008 sec
5000    | 0.012 sec  | 0.010 sec
6000    | 0.015 sec  | 0.013 sec
7000    | 0.019 sec  | 0.018 sec
8000    | 0.025 sec  | 0.022 sec
9000    | 0.042 sec  | 0.038 sec

As you can see, shared memory is faster but only slightly. 


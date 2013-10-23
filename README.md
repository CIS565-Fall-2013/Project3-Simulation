CIS565: Project 3: CUDA Simulation and GLSL Visualization
===
Fall 2013
---
---
INTRODUCTION
---
The goal of this project was to leverage the power of GPU in N-body simulations, especially those which require N^2 queries at every step of the simulation.
Inter-body forces usually require calculating contributions from each body on every other body. To this end, there are two parts of this project. 

Part1 simulates gravitational forces between planets, and Part2 simulates flocking-like group behaviors.

PART 1: CUDA NBody Simulation
===

---
FEATURES:
---

 *  Full N-body inter-body gravitational forces
 *  Simulation implemented with a naive and shared memory approach
 *  Vertex shader code to render a height field
 *  Fragment shader code to light that height field
 *  Geometry shader code to create screen facing billboards from rendered points
 *  Fragment shader code to render those billboards like colorful spheres with simple diffuse shading
 *  RK4 (Runge Kutta) and Euler numerical integration methods
 
---
BUILDING AND RUNNING
---
Change the following to run the code with different settings

```
//Control the number of bodies in the simulation
//main.cpp (line 7)
#define N_FOR_VIS 25
```

```
//Settings for shared memory and RK4
//kernel.h (line 17)
#define SHARED 1
#define RK4 0
```

---
SHARED MEMORY IMPLEMENTATION
---

Before I go into the shared memory approach, I would like to quickly state the naive way to do this,

    Launch a thread on the GPU for each body bi
        (In each thread)
        totalInteractionForce = 0
        for all N bodies
            calculate force fij from body bj on bi
            totalInteractionForce += fij

An important point to note is that at each time step, the calculation of all the forces is done based on the snapshot of the state from previous frame.
Hence, we can do the calculations in parallel,  where each body's calculation is independent of others.

Though the above approach is massively parallel, the memory access on the GPU is all over the place.
If we can access memory in a "good" manner, we can hope to get better performance.
One such technique is to use shared memory.

    Launch a thread on the GPU for each body bi
    Based on a pre-determined tile size, determine the number of tiles needed to cover the global memory array
    (in each block)
      Foreach tile in tiles
       load a tile from global memory into shared memory
       __syncthreads
       Foreach thread in the block
        Accumulate the forces on body bi from the current tile
       __syncthreads
       
      return accumulatedForce
        
The most important part is to remember to sync the threads, once after loading a tile into shared memory and once after the current tile has been utilized by all the threads in the block.
It is much faster to access data from shared memory than global memory, and since every thread walks down sequentially on a tile, we end up getting better performance.

---
SCREENSHOTS
---
25 planets orbiting around the center.

An interesting aspect of the render is that these planets are billboards and have been shaded to simulate spheres

![alt tag](https://raw.github.com/vimanyu/Project3-Simulation/master/Part1/resources/colored_planets.png)

PART 2: Flocking
===

---
FEATURES
---
- Simulation of two categories of group behaviours
  * N-body behaviors:  Arrival, Departure 
  * N^2 body behaviors: Alignment, Separation, Cohesion, Flocking

- The N-body behaviors are implemented simply with each thread responsible for a body and accepting a target position.

- The N^2 body behaviors are implemented either through the naive approach or the shared memory approach.

**Arrival**: The agents all move to the origin of the world. As they get closer to the target, their velocity decreases.

**Departure**: The agents get repelled away from the world origin. And they slow down as they get farther away

**Alignment**: Agents look for other agents in the neighbourhood and the group moves with an average velocity of the neighbourhood

**Separation**: Agents calculate the average departure velocity from the other agents in the neighbourhood

**Cohesion**: Agents move to the center of mass of all the agents in the neighbourhood

**Flocking**: Combination of alignment, cohesion and separation. Useful to simulate flocking of birds, shoals of fish swimming, etc.

---
KEYBOARD CONTROLS
---

We can trigger different group behaviors interactively

Key|Group Behavior
---|---
'a' | Arrival
'd' | Departure
'S' | Separation
'C' | Cohesion
'A' | Alignment
'F' | Flocking

---
VIDEOS
---
**100 bodies sim:**

[![ScreenShot](https://raw.github.com/vimanyu/Project3-Simulation/master/Part2/resources/flockingVideoScreenshot.png)](http://www.youtube.com/watch?v=Gj59Ote7p5A)

**1000 bodies sim:**

[![ScreenShot](https://raw.github.com/vimanyu/Project3-Simulation/master/Part2/resources/flockingVideo2Screenshot.png)](http://www.youtube.com/watch?v=mV6yPZRVU-U)

---
PERFORMANCE ANALYSIS
---

**Tested on a laptop with Intel Core2Duo T7100 and Nvidia 8600M GT**

Test 1: Comparison of Euler vs RK4 integration

![alt tag](https://raw.github.com/vimanyu/Project3-Simulation/master/Part1/resources/eulerVsRk4.png)

Test 2: Comparison of Naive implementation vs Shared memory implementation

![alt tag](https://raw.github.com/vimanyu/Project3-Simulation/master/Part1/resources/naiveVsSharedMem.png)

---
ACKNOWLEDGEMENTS
---
I adapted the geometry shader code from [this excellent tutorial on the subject](http://ogldev.atspace.co.uk/www/tutorial27/tutorial27.html)

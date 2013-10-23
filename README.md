CIS565: Project 3: CUDA Simulation and GLSL Visualization
===
Fall 2013
---
Due Sunday, 10/20/2013 by 11:59:59 pm
---

PART 1: CUDA NBody Simulation
===
This is the basic tutorial part. I'd done with this part, as well as the shared memory force calculation. You can find the performance evaluation below for this shared memory part.


PART 2: Your CUDA Simulation
===

---
VIDEO LINK
---

basic pyramid version
http://youtu.be/ffbJMcmOPHc

enhanced version
http://www.youtube.com/watch?v=EDuQCPeSilE&feature=youtu.be

Do feel free to watch my demo on the above link.

---
INTRODUCTION
---
I had implemented 2 simulations for this project: cloth simulation and flocking behavior simulation. 

In the cloth simulation, I simulate a cloth which is modeled as a 200*200 mass-spring system. Each knot of the cloth 
have 12 springs: 4 structure springs, 4 shear springs and 4 bend springs. Each spring is calculated upon its original length as the value when it is set up. Coeff values for all three kinds of springs can be modified 
easily in the kernel.h. I do not implement the collision detection part because of the the time is limited. Instead, I add a wind-force-field in this simulation so that it can wave as the wind blows.

In the flocking behaviors simulation, I used 12,000 agents in the demo video. The flocking behavior have 4 component behaviours: cohesion, which is used to group them together; separation, which guarantee the minimum 
distance between the agents; alignment, make sure that the neighbour agents shall go to the same direction; and wander, which add some noise to the individuals. Coeff values are also listed in the kernel.h. 

On the interaction side, I add the mouse interaction which enable the user to manipulate the camera with zoom in/out, move left/right, rotate left/ right. On the glsl side, I change the code so that it will render as pyramids
 for each point, with different colors.
 
 So all above are generally the work I'd done with. And I'll explain them in detail below.

---
CLOTH SIMULATION
---
In this simulation, I do not implement the collision detection part. Instead, I add a wind-force-field so that it can wave against the wind. The wind force is pretty much similar to the gravity field, and is much easier to implement. 
I would like to try to accomplish the obstacle collision detection after I have some free moments lol.

So in the cloth simulation, I set up the cloth as a 200*200 knots matrix(this number is flexible in the main.h). For each knot, I set up about 12 springs(the knots near to the boundary will have less springs) to 
simulate the cloth, which includes 4 structural springs, 4 shear springs and 4 bend springs. Each knot has its mass as 1.0f, so that the cloth is modeled into a large-scale mass-spring system. For each spring, its original length 
is the lenght when it is set up. 

The calculation of mass-spring system is pretty straight forward, which just follows the Hook's rule. However, I came across a serious problem that the spring would mistakenly overshoot. 
![screeshot](http://https://github.com/heguanyu/Project3-Simulation/tree/master/screenshots/spring_exploded_explain.jpg?raw=true)
So when the spring has a large enough mass to bounce it back ,it may bounce to the other side, and thus messed up the later calculation.

There is not a good way to solve this problem, which is the disadvantage of the mass-spring system, even with the RK4 integration method. Therefore, I invent my own method to alleviate this problem in some extends.
The idea is simple: shrink the DT to increase the precision of the simulation. However, directly modify the DT will cause the problem that the result video of the simulation is slower than the normal version. To handle this 
problem, I just divide one calculation step into several substeps, which where the DT for each step is the original DT/substeps num. In this case, the precision can be enhanced. As far as I'd tested, in the case of 200*200 knots,
 supporting as much as 16 substeps can still keep the fps in the real-time animation range(>24 fps). With this method, I can use more stiff springs, and make the calculation more accurate.
 
 
 I'd also attemped to change the PBO so that it can render the cloth as a cloth. However I just failed it, and cannot figure out where is going wrong. Therefore, I just transfer the position of knots into VBO and then rendering it 
 as separate knots. When the knots are crowded enough, it stlil looks like a cloth, just the normal is a little bit weird. 
 
 Below are the runtime screenshots.
![screeshot](http://https://github.com/heguanyu/Project3-Simulation/tree/master/screenshots/cloth.bmp?raw=true)
![screeshot](http://https://github.com/heguanyu/Project3-Simulation/tree/master/screenshots/cloth2.bmp?raw=true)


---
FLOCKING SIMULATION
---
Flocking is a million times easier than the cloth simulation, where the formula is much more obvious, no need to model it, and it really take slight jobs in implementing the code. 
So, my flocking behavior consists of 4 sub-behaviors. The cohesion, separation, alignment and wander. The weights for all 4 behaviors are flexible, and which can lead to different animation result. Also, the radius of the neighbourhood,
 which is another crucial parameter of the calculation, is also flexible, and it is interesting to see that when the radius is small, it is like everyone is moving randomly in the space.
 
 I transfer the velocity data into the geometry shader so that it will generate a sharp pyramid that points to the direction where it is moving towards. So if comparing the both video demos, you will find that the second one looks much
  better than the first one. 

Below are the runtime screenshots
![screeshot](http://https://github.com/heguanyu/Project3-Simulation/tree/master/screenshots/flocking1.bmp?raw=true)
![screeshot](http://https://github.com/heguanyu/Project3-Simulation/tree/master/screenshots/flocking2.bmp?raw=true)

---
SHARED MEMORY WALKTHROUGH
---
Shared memory is easy to implement. So for each thread x in the block, it will copy the data from the block i and offset x into the shared memory, where i loop on the number of the blocks. After all the thread 
finish copying, it fetch all the value from the shared memory and calculate the related force/acceleration/velocity. And after all the threads finish calculation, it will move on to the next loop i.



---
PERFORMANCE EVALUATION
---
As there is nothing good to evaluate for this assignment, I just evaluate the shared memory part for the flocking simulation.

Ironically, this shared memory method do not fit the flocking simulation. In flocking simulation, sharedMemory method is even slower than the naive calculation. The reason is that when calculate the value between two agents in the flock, 
if their distance is larger than the radius of the neighbourhood, it will not go into the calculation part and then go to the next agent. So some threads may finish calculation earlier and then help calculating the later data.

In the shared memory method, we have to align all the threads, which means that even one thread has finished its calculation, generally because its agenets have less neighbours than the others, it would have to wait for the other threads
 to finish their work(by __syncthreads()). So the shared memory is slower.
 
 However, in the N-Body case, it is faster than the naive calculation, where N body method do not check if a planet is too far away or not. Therefore, the threadds are always aligned, and shared memory method is thus faster.\
 
 The chart below shows  how much faster the naive method is upon the shared method.
![screeshot](http://https://github.com/heguanyu/Project3-Simulation/tree/master/performance_eval/flocking_comp_512.bmp?raw=true) 
 The chart below shows that the blocksize has very slight influence on the fps.
 ![screeshot](http://https://github.com/heguanyu/Project3-Simulation/tree/master/performance_eval/flocking_comp_blocknum.bmp?raw=true) 
 
 
---
THIRD PARTY CODE
---
None except for the basecode. Everything else are scratched up, including the mouse interaction


CIS565: Project 3: CUDA Simulation and GLSL Visualization
===
Fall 2013
---
Yingting Xiao
---

PART 1: CUDA NBody Simulation
===

![alt tag](https://raw.github.com/YingtingXiao/Project3-Simulation/master/screenshots/NBody.PNG)

---
Performance Analysis
---

![alt tag](https://raw.github.com/YingtingXiao/Project3-Simulation/master/performance/nbody.PNG)

The data is collected with N = 10000 and no visualization. We can see the time per frame decreases as block size increases, and stays approximately the same when block size >= 128. When block size <= 64, shared memory performance is slightly better than naive computation performance. However, when block size > 64, non-shared memory computation outperforms shared memory computation. I think this is because there isn't a lot of access to global/shared memory in calculateAcceleration, and __syncthreads in sharedMemAcc takes extra time.


PART 2: PBD (position based dynamics) Cloth Simulation
===

![alt tag](https://raw.github.com/YingtingXiao/Project3-Simulation/master/screenshots/Cloth1.PNG)

![alt tag](https://raw.github.com/YingtingXiao/Project3-Simulation/master/screenshots/Cloth2.PNG)

Video:

https://vimeo.com/77772426

---
Approach
---

My cloth simulation is based on this paper by Matthias Muller et al:

http://www.matthiasmueller.info/publications/posbaseddyn.pdf

I ported my pbd framework in CIS563 into CUDA. There are three types of internal constraints: fixed point constraint, stretch constraint and bend constraint. Fixed point constraint worked fine, but projecting stretch constraint doesn't work well in parallel. The reason is that when the solver is projecting one constraint, it's using vertex positions computed by projecting other constraints. In my cloth, the top vertices are fixed, so the stretch propagates from the top to the bottom. Therefore, to get a good looking cloth, I think we have to project the stretch constraints in a certain order. I haven't figured out how to do this in parallel, so my cloth looks kind of ugly. I also tried using mass-spring system for the stretch constraints, but since the rest of the solver is PBD, mass-spring doesn't work well. I think there exists some way to project the stretch constraints in parallel, so I'll keep experimenting.

---
Performance Analysis
---

![alt tag](https://raw.github.com/YingtingXiao/Project3-Simulation/master/performance/cloth.PNG)

The data is collected with 101 subdivisions in both x and z direction on the cloth, i.e. 101*101 vertices. The time used by runCuda is roughly linearly correlated to the number of solver iterations (number of iterations for resolving stretch and bend constraints). This indicates that resolving stretch and bend constraints is the bottleneck of this simulation.
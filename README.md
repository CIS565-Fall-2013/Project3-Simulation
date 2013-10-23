CUDA Simulation (N-Body &amp; Flocking)
=======================================

N-Body
------
![25](Part1/resources/25.png)

### CPU vs GPU
Is GPU really faster? I implemented the same program on both CPU and GPU to compare their performance.
GPU was indeed faster. It can also be seen that the CPU exhibits the characteristics of O(n<sup>2</sup>) time complexity for force calculation.
![Performance Comparison Results](Part1/resources/Performance Comparison.png)
![Performance Comparison Results Scaled](Part1/resources/Performance Comparison Scaled.png)
<sub><sup>Note: Above were calculated based only on time taken to compute the acceleration, velocity, and position of each planet.</sup></sub>

### Galaxy
![Galaxy](Part1/resources/3200_2.png)
&#8593; 3200 planets after 1000 frames.

Flocking
--------




<sub><sup>Note: This was for a class at UPenn: CIS565 Project 3 Fall 2013 (Due 10/22/2013)</sup></sub>


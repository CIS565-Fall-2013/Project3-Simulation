Features:
(Attempted to extend nbody in lieu of doing cloth/flocking):

0. Standard features for Nbody (visualization, forces, shared memory)
0.5. Standard performance analysis: http://lightspeedbanana.blogspot.com/2013/10/nbody-simulation-naive-acceleration.html
1. Collision detection (see http://youtu.be/4AkgVNNrA-A)
2. Use of "softening" term to avoid "slingshot" effect when particles are too close to each other (see http://lightspeedbanana.blogspot.com/2013/10/nbody-simulation-better-clumping.html)
3. Comparison of Forward Euler vs Symplectic Euler (see the second half of http://lightspeedbanana.blogspot.com/2013/10/nbody-simulation-better-clumping.html)
4. Prefetching and loop unrolling, analysis of prefecthing and loop unrolling, plus performance analysis to try to find "sweet spot" of kernel. I tested over 100 permutations of block size, tile size, prefetching, and loop unrolling. See the following post (http://lightspeedbanana.blogspot.com/2013/10/nbody-simulation-shared-memory-loop.html)

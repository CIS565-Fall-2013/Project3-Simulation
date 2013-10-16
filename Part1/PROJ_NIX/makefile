#All cuda paths are for v5.0+

NVCC     = nvcc -m64
CC       = gcc -m64
CPPC     = g++ -m64

CUDA_FLAGS = -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda/include

LFLAGS = -lglut -lGL -lGLEW 

all: 565simulator

kernel.o: ../src/kernel.cu
	$(NVCC) $(CUDA_FLAGS) -c ../src/kernel.cu

glslUtility.o: ../src/glslUtility.cpp
	$(CPPC) $(CUDA_FLAGS) ../src/glslUtility.cpp -c

utilities.o: ../src/utilities.cpp
	$(CPPC) $(CUDA_FLAGS) ../src/utilities.cpp -c

main.o: ../src/main.cpp ../src/glslUtility.h ../src/kernel.h ../src/utilities.h ../src/main.h
	$(CPPC) $(CUDA_FLAGS) ../src/main.cpp -c

565simulator: main.o kernel.o glslUtility.o utilities.o
	$(NVCC) $(LFLAGS) main.o kernel.o glslUtility.o utilities.o -o 565simulator

clean:
	rm *.o
	rm 565simulator

test: 565simulator
	./565simulator

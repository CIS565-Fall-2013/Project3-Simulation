/* Example N-Body simulation for CIS565 Fall 2013
* Author: Liam Boone
* main.cpp */

#include "main.h"

#define N_FOR_VIS 1500
#define DT 0.2
#define VISUALIZE 1

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv)
{
	// Launch CUDA/GL

	init(argc, argv);

	cudaGLSetGLDevice( compat_getMaxGflopsDeviceId() );
	cudaGLRegisterBufferObject( boidVBO );

#if VISUALIZE == 1 
	initCuda(N_FOR_VIS, mapDims);
#else
	initCuda(2*128);
#endif

	projection = glm::perspective(fovy, float(width)/float(height), zNear, zFar);
	view = glm::lookAt(cameraPosition, glm::vec3(0,0,mapDims.z/2), glm::vec3(0,0,1));


	GLuint passthroughProgram;
	initShaders(program);

	glUseProgram(program[BOIDS]);
	glActiveTexture(GL_TEXTURE0);
	glEnable(GL_DEPTH_TEST);


	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);

	glutMainLoop();

	return 0;
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda()
{
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

	float4 *dptr=NULL;
	float *dptrvert=NULL;
	cudaGLMapBufferObject((void**)&dptrvert, boidVBO);

	// execute the kernel
	cudaNBodyUpdateWrapper(DT);
#if VISUALIZE == 1
	cudaUpdateVBO(dptrvert);
#endif
	// unmap buffer object
	cudaGLUnmapBufferObject(boidVBO);
}

int timebase = 0;
int frame = 0;

void display()
{
	static float fps = 0;
	frame++;
	int time=glutGet(GLUT_ELAPSED_TIME);

	if (time - timebase > 1000) {
		fps = frame*1000.0f/(time-timebase);
		timebase = time;
		frame = 0;
	}
	runCuda();

	char title[100];
	sprintf( title, "565 NBody sim [%0.2f fps]", fps );
	glutSetWindowTitle(title);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   
#if VISUALIZE == 1
	// VAO, shader program, and texture already bound
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	//glDrawElements(GL_TRIANGLES, 6*field_width*field_height,  GL_UNSIGNED_INT, 0);

	//Draw BOIDS
	glUseProgram(program[BOIDS]);

	glEnableVertexAttribArray(positionLocation);
	glEnableVertexAttribArray(upLocation);
	glEnableVertexAttribArray(forwardLocation);
	glEnableVertexAttribArray(colorLocation);
	glEnableVertexAttribArray(shapeLocation);

	glBindBuffer(GL_ARRAY_BUFFER, boidVBO);
	//Setup interleaved buffer
	glVertexAttribPointer(positionLocation, 4, GL_FLOAT, GL_FALSE, boidVBOStride*sizeof(GLfloat), (void*)(boidVBO_PositionOffset*sizeof(GLfloat))); 
	glVertexAttribPointer(upLocation,       3, GL_FLOAT, GL_FALSE, boidVBOStride*sizeof(GLfloat), (void*)(boidVBO_UpOffset*sizeof(GLfloat))); 
	glVertexAttribPointer(forwardLocation,  3, GL_FLOAT, GL_FALSE, boidVBOStride*sizeof(GLfloat), (void*)(boidVBO_ForwardOffset*sizeof(GLfloat))); 
	glVertexAttribPointer(colorLocation,    3, GL_FLOAT, GL_FALSE, boidVBOStride*sizeof(GLfloat), (void*)(boidVBO_ColorOffset*sizeof(GLfloat))); 
	glVertexAttribPointer(shapeLocation,    4, GL_FLOAT, GL_FALSE, boidVBOStride*sizeof(GLfloat), (void*)(boidVBO_ShapeOffset*sizeof(GLfloat))); 

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boidIBO);
	
	glPointSize(5.0f); 
	glDrawElements(GL_POINTS, N_FOR_VIS, GL_UNSIGNED_INT, 0);
	glPointSize(1.0f); 

	glDisableVertexAttribArray(positionLocation);
	glDisableVertexAttribArray(upLocation);
	glDisableVertexAttribArray(forwardLocation);
	glDisableVertexAttribArray(colorLocation);
	glDisableVertexAttribArray(shapeLocation);

#endif
	glutPostRedisplay();
	glutSwapBuffers();
}

void keyboard(unsigned char key, int x, int y)
{
	std::cout << key << std::endl;
	switch (key) 
	{
	case(27):
		//Reset device to flush profiling data
		cudaDeviceReset(); 
		exit(0);
		break;
	}
}


//-------------------------------
//----------SETUP STUFF----------
//-------------------------------


void init(int argc, char* argv[])
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(width, height);
	glutCreateWindow("565 Flocking Sim");

	// Init GLEW
	glewInit();
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		/* Problem: glewInit failed, something is seriously wrong. */
		std::cout << "glewInit failed, aborting." << std::endl;
		exit (1);
	}

	initVAO();
}


//Setup Interleaved VBO for boids
void initVAO(void)
{

	GLfloat *bodies    = new GLfloat[boidVBOStride*(N_FOR_VIS)];
	GLuint *bindices   = new GLuint[N_FOR_VIS];

	//Initialize boids.
	for(int i = 0; i < N_FOR_VIS; i++)
	{
		//Position
		bodies[boidVBO_PositionOffset + boidVBOStride*i + 0] = 0.0f;
		bodies[boidVBO_PositionOffset + boidVBOStride*i + 1] = 0.0f;
		bodies[boidVBO_PositionOffset + boidVBOStride*i + 2] = 0.0f;
		bodies[boidVBO_PositionOffset + boidVBOStride*i + 3] = 1.0f;

		//Up
		bodies[boidVBO_UpOffset + boidVBOStride*i + 0] = 0.0f;
		bodies[boidVBO_UpOffset + boidVBOStride*i + 1] = 0.0f;
		bodies[boidVBO_UpOffset + boidVBOStride*i + 2] = 1.0f;

		//Forward
		bodies[boidVBO_ForwardOffset + boidVBOStride*i + 0] = 1.0f;
		bodies[boidVBO_ForwardOffset + boidVBOStride*i + 1] = 0.0f;
		bodies[boidVBO_ForwardOffset + boidVBOStride*i + 2] = 0.0f;

		//Color
		bodies[boidVBO_ColorOffset + boidVBOStride*i + 0] = 1.0f;
		bodies[boidVBO_ColorOffset + boidVBOStride*i + 1] = 1.0f;
		bodies[boidVBO_ColorOffset + boidVBOStride*i + 2] = 1.0f;

		//Shape
		bodies[boidVBO_ShapeOffset + boidVBOStride*i + 0] = 1.0f;
		bodies[boidVBO_ShapeOffset + boidVBOStride*i + 1] = 1.0f;
		bodies[boidVBO_ShapeOffset + boidVBOStride*i + 2] = 0.0f;
		bodies[boidVBO_ShapeOffset + boidVBOStride*i + 3] = 0.0f;
		bindices[i] = i;
	}

	glGenBuffers(1, &boidVBO);
	glGenBuffers(1, &boidIBO);

	glBindBuffer(GL_ARRAY_BUFFER, boidVBO);
	glBufferData(GL_ARRAY_BUFFER, boidVBOStride*(N_FOR_VIS)*sizeof(GLfloat), bodies, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boidIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, (N_FOR_VIS)*sizeof(GLuint), bindices, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	delete[] bodies;	
	delete[] bindices;
}

void initShaders(GLuint * program)
{
	GLint location;
	program[BOIDS] = glslUtility::createProgram("shaders/boidVS.glsl", "shaders/boidGS.glsl", "shaders/boidFS.glsl", attributeLocations, 1);
	glUseProgram(program[BOIDS]);

	//Initalize uniforms
	if ((location = glGetUniformLocation(program[BOIDS], "u_projMatrix")) != -1)
	{
		glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	}
	if ((location = glGetUniformLocation(program[BOIDS], "u_viewMatrix")) != -1)
	{
		glUniformMatrix4fv(location, 1, GL_FALSE, &view[0][0]);
	}
	if ((location = glGetUniformLocation(program[BOIDS], "u_lightPos")) != -1)
	{
		glUniform3fv(location, 1, &lightPosition[BOIDS]);
	}
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void deletePBO(GLuint* pbo)
{
	if (pbo) 
	{
		// unregister this buffer object with CUDA
		cudaGLUnregisterBufferObject(*pbo);

		glBindBuffer(GL_ARRAY_BUFFER, *pbo);
		glDeleteBuffers(1, pbo);

		*pbo = (GLuint)NULL;
	}
}

void deleteTexture(GLuint* tex)
{
	glDeleteTextures(1, tex);
	*tex = (GLuint)NULL;
}

void shut_down(int return_code)
{
	system("pause");
	exit(return_code);
}

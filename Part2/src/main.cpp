/* Example N-Body simulation for CIS565 Fall 2013
 * Author: Liam Boone
 * main.cpp */

#include "main.h"

#define N_FOR_VIS 9600
#define N_FOR_PREDATOR 10
#define DT 0.3
#define VISUALIZE 1
//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv)
{
    // Launch CUDA/GL

    init(argc, argv);

    cudaGLSetGLDevice( compat_getMaxGflopsDeviceId() );
    initPBO(&pbo);
    cudaGLRegisterBufferObject( droidVBO );
	cudaGLRegisterBufferObject( droidNBO );
	cudaGLRegisterBufferObject( predatorVBO );
    
#if VISUALIZE == 1 
	initCuda(N_FOR_VIS, N_FOR_PREDATOR);
#else
    initCuda(2*128);
#endif

    projection = glm::perspective(fovy, float(width)/float(height), zNear, zFar);
    view = glm::lookAt(cameraPosition, glm::vec3(0), glm::vec3(0,0,1));

    projection = projection * view;

    GLuint passthroughProgram;
    initShaders(program);

    glEnable(GL_DEPTH_TEST);


    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);

    glutMainLoop();

    return 0;
}
int timebase = 0;
int frame = 0;
//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda()
{
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    
	GLfloat *dptrvert=NULL;
	GLfloat *dptnormal=NULL;
	float *dptrpredvert=NULL;	
    //cudaGLMapBufferObject((void**)&dptr, pbo);
    cudaGLMapBufferObject((void**)&dptrvert, droidVBO);
	cudaGLMapBufferObject((void**)&dptrpredvert, predatorVBO);
	cudaGLMapBufferObject((void**)&dptnormal, droidNBO);
	  
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0);

    // execute the kernel
    cudaNBodyUpdateWrapper(DT, frame);
#if VISUALIZE == 1
    cudaUpdateVBO(dptrvert, dptnormal, field_width, field_height);
	cudaUpdateVBOPre(dptrpredvert, field_width, field_height);
#endif
    // unmap buffer object
    cudaGLUnmapBufferObject(droidVBO);
	cudaGLUnmapBufferObject(droidNBO);
	cudaGLUnmapBufferObject(predatorVBO);

	cudaEventRecord( stop, 0);
	cudaEventSynchronize( stop );

	float seconds = 0.0f;
	cudaEventElapsedTime( &seconds, start, stop);
  
	//printf("time %f \n", seconds);
}



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
    sprintf( title, "Flocking [%0.2f fps]", fps );
    glutSetWindowTitle(title);

    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   

	camera();
#if VISUALIZE == 1
   
    glUseProgram(program[DROID]);

    glEnableVertexAttribArray(positionLocation);
	glEnableVertexAttribArray(colorLocation);
	glEnableVertexAttribArray(normalLocation);

    glBindBuffer(GL_ARRAY_BUFFER, droidVBO);
    glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0); 

	glBindBuffer(GL_ARRAY_BUFFER, droidCBO);
	glVertexAttribPointer((GLuint)colorLocation, 3, GL_FLOAT, 0, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, droidNBO);
	glVertexAttribPointer((GLuint)normalLocation, 4, GL_FLOAT, 0, 0, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, droidIBO);
   
   	glDrawElements(GL_TRIANGLES, N_FOR_VIS * 12, GL_UNSIGNED_INT, 0);
    //glDrawRangeElements(GL_TRIANGLES, 0, N_FOR_VIS * 12, N_FOR_VIS * 12, GL_UNSIGNED_SHORT, 0);

	glDisableVertexAttribArray(positionLocation);
	glDisableVertexAttribArray(colorLocation);
	glDisableVertexAttribArray(normalLocation);



	glUseProgram(program[PREDATOR]);

    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, predatorVBO);
    glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0); 

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, predatorIBO);
   
    //glPointSize(4.0f); 
    glDrawElements(GL_POINTS, N_FOR_PREDATOR, GL_UNSIGNED_INT, 0);

    glPointSize(1.0f);

    glDisableVertexAttribArray(positionLocation);

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
            exit(1);
            break;
    }
}


void mouse(int button, int state, int x, int y)
{
	if(button ==  GLUT_LEFT_BUTTON)
	{
		glutMotionFunc(mouseLeft);
	}
	else if(button == GLUT_RIGHT_BUTTON)
	{
		glutMotionFunc(mouseRight);
	}
	else
	{
		glutMotionFunc(NULL);
	}
	
	mouse_old_x = x;
    mouse_old_y = y;
}


void mouseLeft(int x, int y)
{
	float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

	head += dy * 0.2f;
    pitch += dx * 0.2f;

	mouse_old_x = x;
    mouse_old_y = y;
}

void mouseMiddle(int x, int y)
{
	float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

	glm::vec3 vdir(lookat - cameraPosition);
    glm::vec3 u(glm::normalize(glm::cross(vdir, glm::vec3(0,0,1))));
    glm::vec3 v(glm::normalize(glm::cross(u, vdir)));

    lookat += 0.01f * (dy * v - dx * u);

	mouse_old_x = x;
    mouse_old_y = y;
}


void mouseRight(int x, int y)
{
	float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

	eye_distance -= dy * 0.01f;

	mouse_old_x = x;
    mouse_old_y = y;
}

void camera()
{
	float r_head = glm::radians(head), r_pitch = glm::radians(pitch);
    cameraPosition.x = lookat.x + eye_distance * glm::cos(r_head) * glm::cos(r_pitch);
    cameraPosition.y = lookat.y + eye_distance * glm::sin(r_head);
    cameraPosition.z = lookat.z + eye_distance * glm::cos(r_head) * glm::sin(r_pitch);

    glMatrixMode(GL_MODELVIEW);
    glm::vec3 up = glm::vec3(0.0f, (glm::cos(r_head) > 0.0f) ? 1.0f : -1.0f, 0.0f);
    modelview = glm::lookAt(cameraPosition, lookat, up);
    glLoadMatrixf(&modelview[0][0]);
    
    glMatrixMode(GL_PROJECTION);
    projection = glm::perspective(60.0f, static_cast<float>(field_width) / static_cast<float>(field_height), 0.1f, 100.0f);
    glLoadMatrixf(&projection[0][0]);

	GLint location;
	glUseProgram(program[DROID]);
	
    if ((location = glGetUniformLocation(program[0], "u_projMatrix")) != -1)
    {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
	if ((location = glGetUniformLocation(program[0], "u_modelviewMatrix")) != -1)
    {
        glUniformMatrix4fv(location, 1, GL_FALSE, &modelview[0][0]);
    }
	 if ((location = glGetUniformLocation(program[0], "u_cameraPos")) != -1)
    {
        glUniform3fv(location, 1, &cameraPosition[0]);
    }
	if ((location = glGetUniformLocation(program[0], "u_lightPos")) != -1)
    {
		glUniform3fv(location, 1, &lightPosition[0]);
    }
	if ((location = glGetUniformLocation(program[0], "u_lightColor")) != -1)
    {
		glUniform3fv(location, 1, &lightColor[0]);
    }	    

	glUseProgram(program[PREDATOR]);
	if ((location = glGetUniformLocation(program[1], "u_projMatrix")) != -1)
    {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
	 if ((location = glGetUniformLocation(program[1], "u_cameraPos")) != -1)
    {
        glUniform3fv(location, 1, &cameraPosition[0]);
    }
	if ((location = glGetUniformLocation(program[1], "u_modelviewMatrix")) != -1)
    {
        glUniformMatrix4fv(location, 1, GL_FALSE, &modelview[0][0]);
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
    glutCreateWindow("565 NBody sim");

	srand((unsigned)time(0));
    // Init GLEW
    glewInit();
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        /* Problem: glewInit failed, something is seriously wrong. */
        std::cout << "glewInit failed, aborting." << std::endl;
        exit (1);
    }

	initDroid();
	initPredator();
}

void initPBO(GLuint* pbo)
{
    if (pbo) 
    {
        // set up vertex data parameter
        int num_texels = field_width*field_height;
        int num_values = num_texels * 4;
        int size_tex_data = sizeof(GLfloat) * num_values;

        // Generate a buffer ID called a PBO (Pixel Buffer Object)
        glGenBuffers(1,pbo);
        // Make this the current UNPACK buffer (OpenGL is state-based)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
        // Allocate data for the buffer. 4-channel 8-bit image
        glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
        cudaGLRegisterBufferObject( *pbo );
    }
}

//void initDroid()
//{
//	glGenBuffers(1, &droidVBO);
//    glGenBuffers(1, &droidIBO);
//	glGenBuffers(1, &droidNBO);
//	glGenBuffers(1, &droidCBO);
//
//	float scale = 0.2f;
//
//	float* vertices = new float[48 * N_FOR_VIS];
//
//	for(int i = 0; i < 12 * N_FOR_VIS; i += 12)
//	{
//		float var = 0.0f;
//		vertices[i + 0] = var-scale; vertices[i + 1] = var+scale; vertices[i + 2] = var+scale; vertices[i + 3] = 1.0f;
//		vertices[i + 4] = var-scale; vertices[i + 5] = var-scale; vertices[i + 6] = var+scale; vertices[i + 7] = 1.0f;
//		vertices[i + 8] = var+scale; vertices[i + 9] = var-scale; vertices[i + 10] = var+scale; vertices[i + 11] = 1.0f;	
//		vertices[i + 15] = 1.0f; vertices[i + 19] = 1.0f; vertices[i + 23] = 1.0f; vertices[i + 27] = 1.0f;
//		vertices[i + 31] = 1.0f; vertices[i + 35] = 1.0f; vertices[i + 39] = 1.0f; vertices[i + 43] = 1.0f;
//		vertices[i + 47] = 1.0f;
//	}
//		
//	glBindBuffer(GL_ARRAY_BUFFER, droidVBO);
//	glBufferData(GL_ARRAY_BUFFER, 12 * N_FOR_VIS* sizeof(float), vertices, GL_DYNAMIC_DRAW);
//		
//	delete [] vertices;
//
//	//again with colors
//	float* colors = new float[36 * N_FOR_VIS];
//
//	for(int i = 0; i < N_FOR_VIS; i++)
//	{
//		float color[3] = {(float)rand()/(float)RAND_MAX,(float)rand()/(float)RAND_MAX,(float)rand()/(float)RAND_MAX};
//		
//		for(int j = 0; j < 9; j += 3)
//		{
//			colors[i*9+j] = color[0];
//			colors[i*9+j+1] = color[1];
//			colors[i*9+j+2] = color[2];
//		}
//	}
//	
//	glBindBuffer(GL_ARRAY_BUFFER, droidCBO);
//	
//	glBufferData(GL_ARRAY_BUFFER, 9 * N_FOR_VIS * sizeof(float), colors, GL_STREAM_DRAW);
//																				
//	delete [] colors;
//
//	//once more, this time with normals
//	float* normals = new float[12 * N_FOR_VIS];
//	
//	for(int i = 0; i < 12 * N_FOR_VIS; i += 12)
//	{
//		normals[i + 0] = 0; normals[i + 1] = 0; normals[i + 2] = 1; normals[i + 3] = 0;
//		normals[i + 4] = 0; normals[i + 5] = 0; normals[i + 6] = 1; normals[i + 7] = 0;
//		normals[i + 8] = 0; normals[i + 9] = 0; normals[i + 10] = 1; normals[i + 11] = 0;		
//	}
//
//	glBindBuffer(GL_ARRAY_BUFFER, droidNBO);
//	glBufferData(GL_ARRAY_BUFFER, 12 * N_FOR_VIS * sizeof(float), normals, GL_STATIC_DRAW); 
//																				 
//	delete [] normals;
//
//	unsigned short* indices = new unsigned short[3 * N_FOR_VIS];
//
//	for(int i = 0; i < 3 * N_FOR_VIS; i += 3)
//	{
//		indices[i + 0] = i + 0; indices[i + 1] = i + 1; indices[i + 2] = i + 2;		
//	}
//
//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, droidIBO);
//	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * N_FOR_VIS * sizeof(unsigned short), indices, GL_STATIC_DRAW);
//
//	glBindBuffer(GL_ARRAY_BUFFER, 0);
//    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
//
//	delete [] indices;
//}
void initDroid()
{
	glGenBuffers(1, &droidVBO);
    glGenBuffers(1, &droidIBO);
	glGenBuffers(1, &droidNBO);
	glGenBuffers(1, &droidCBO);

	float scale = 0.2f;

	GLfloat* vertices = new GLfloat[48 * N_FOR_VIS];

		
	glBindBuffer(GL_ARRAY_BUFFER, droidVBO);
	glBufferData(GL_ARRAY_BUFFER, 48 * N_FOR_VIS* sizeof(GLfloat), vertices, GL_DYNAMIC_DRAW);
		
	delete [] vertices;

	//again with colors
	GLfloat* colors = new GLfloat[36 * N_FOR_VIS];

	for(int i = 0; i < N_FOR_VIS; i++)
	{
		float color[3] = {(float)rand()/(float)RAND_MAX,(float)rand()/(float)RAND_MAX,(float)rand()/(float)RAND_MAX};
		
		for(int j = 0; j < 36; j += 3)
		{
			colors[i*36+j] = color[0];
			colors[i*36+j+1] = color[1];
			colors[i*36+j+2] = color[2];
		}
	}
	
	glBindBuffer(GL_ARRAY_BUFFER, droidCBO);	
	glBufferData(GL_ARRAY_BUFFER, 36 * N_FOR_VIS * sizeof(GLfloat), colors, GL_STREAM_DRAW);
																				
	delete [] colors;

	//once more, this time with normals
	GLfloat* normals = new GLfloat[48 * N_FOR_VIS];	
	

	glBindBuffer(GL_ARRAY_BUFFER, droidNBO);
	glBufferData(GL_ARRAY_BUFFER, 48 * N_FOR_VIS * sizeof(GLfloat), normals, GL_STATIC_DRAW); 
																				 
	delete [] normals;

	GLuint * indices = new GLuint [12 * N_FOR_VIS];

	for(int i = 0; i < 12 * N_FOR_VIS; i ++)
	{
		indices[i] = i;
	}

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, droidIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 12 * N_FOR_VIS * sizeof(GLuint), indices, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	delete [] indices;
}

void initPredator()
{
	 GLfloat *bodies    = new GLfloat[4*(N_FOR_PREDATOR)];
	 GLuint *bindices   = new GLuint[N_FOR_PREDATOR];

	  for(int i = 0; i < N_FOR_PREDATOR; i++)
    {
        bodies[4*i+0] = 0.0f;
        bodies[4*i+1] = 0.0f;
        bodies[4*i+2] = 0.0f;
        bodies[4*i+3] = 1.0f;
        bindices[i] = i;
    }

	glGenBuffers(1, &predatorVBO);
    glGenBuffers(1, &predatorIBO);

	glBindBuffer(GL_ARRAY_BUFFER, predatorVBO);
    glBufferData(GL_ARRAY_BUFFER, 4*(N_FOR_PREDATOR)*sizeof(GLfloat), bodies, GL_DYNAMIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, predatorIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (N_FOR_PREDATOR)*sizeof(GLuint), bindices, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

   
    delete[] bodies;
    delete[] bindices;
}

void initShaders(GLuint * program)
{
    program[0] = glslUtility::createProgram("shaders/droidVS.glsl", "shaders/droidFS.glsl", attributeLocations, 3);

	program[1] = glslUtility::createProgram("shaders/predatorVS.glsl", "shaders/predatorGS.glsl", "shaders/predatorFS.glsl", attributeLocations, 1);   
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
    exit(return_code);
}

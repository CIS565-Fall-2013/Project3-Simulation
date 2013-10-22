/* Example N-Body simulation for CIS565 Fall 2013
 * Author: Liam Boone
 * main.cpp */

#include "main.h"

#define DT 0.5
#define VISUALIZE 1
//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv)
{
    // Launch CUDA/GL
    init(argc, argv);
    cudaGLSetGLDevice( compat_getMaxGflopsDeviceId() );
    cudaGLRegisterBufferObject( planetVBO );
	cudaGLRegisterBufferObject( velocityVBO );
    
#if VISUALIZE == 1 
    initCuda(N_FOR_VIS);
#else
    initCuda(2*128);
#endif

    projection = glm::perspective(fovy, float(width)/float(height), zNear, zFar);
    view = glm::lookAt(cameraPosition, glm::vec3(0.0, 0.0, 0), glm::vec3(0,1,0));
    projection = projection * view;

    initShaders(program);

    glEnable(GL_DEPTH_TEST);

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
	glutMotionFunc(mouseMotion);

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

    float *dptrvert=NULL;
	float *velptr=NULL;
    cudaGLMapBufferObject((void**)&dptrvert, planetVBO);
	cudaGLMapBufferObject((void**)&velptr, velocityVBO);

    // execute the kernel
    cudaFlockingUpdateWrapper(DT, seekTarget);
#if VISUALIZE == 1
    cudaUpdateVBO(dptrvert, velptr);
#endif
    // unmap buffer object
    cudaGLUnmapBufferObject(planetVBO);
	cudaGLUnmapBufferObject(velocityVBO);
}
/*
float ORG[3] = {0,0,0};


float XP[3] = {1,0,0}, XN[3] = {-1,0,0},
	YP[3] = {0,1,0}, YN[3] = {0,-1,0},
	ZP[3] = {0,0,1}, ZN[3] = {0,0,-1};


float w, h, tip = 0, turn = 0;
void Draw_Axes (void)
{
	glPushMatrix ();

	glTranslatef (-2.4, -1.5, -5);
	glRotatef (tip , 1,0,0);
	glRotatef (turn, 0,1,0);
	glScalef (0.25, 0.25, 0.25);

	glLineWidth (2.0);

	glBegin (GL_LINES);
	glColor3f (1,0,0); // X axis is red.
	glVertex3fv (ORG);
	glVertex3fv (XP ); 
	glColor3f (0,1,0); // Y axis is green.
	glVertex3fv (ORG);
	glVertex3fv (YP );
	glColor3f (0,0,1); // z axis is blue.
	glVertex3fv (ORG);
	glVertex3fv (ZP ); 
	glEnd();

	glPopMatrix ();
}

void drawAxes(void)
{
	glPushAttrib(GL_LIGHTING_BIT | GL_LINE_BIT);
	glDisable(GL_LIGHTING);
	//draw axis.
	GLfloat modelview[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, modelview);
	GLint viewport[4];
	glGetIntegerv(GL_VIEWPORT, &viewport[0]);
	GLint width = viewport[2] / 16;
	GLint height = viewport[3] / 16;
	glViewport(0, 0, width, height);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	//get the camera position and up vector from the modelview matrix.
	double campos[3] = {0.0 + 2.0f * modelview[2], 0.0 + 2.0f * modelview[6], 0.0 + 2.0f * modelview[10]};
	double up[3] = {modelview[1], modelview[5], modelview[9]};
	//set up the view matrix.
	gluLookAt(campos[0], campos[1], campos[2], 
		0.0, 0.0, 0.0,
		up[0], up[1], up[2]);

	glBegin(GL_LINES);
	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(1.0f, 0.0f, 0.0f);

	glColor3f(0.0f, 1.0f, 0.0f);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);

	glColor3f(0.0f, 0.0f, 1.0f);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 0.0f, 1.0f);
	glEnd();

	glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
	glPopMatrix();
	glPopAttrib();
}*/

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
	float executionTime = glutGet(GLUT_ELAPSED_TIME) - timeSinceLastFrame;
	timeSinceLastFrame = glutGet(GLUT_ELAPSED_TIME);
    runCuda();

	char title[100];
	sprintf( title, "Flocking Simulation [%0.2f fps] [%0.2fms] ", fps, executionTime);
    glutSetWindowTitle(title);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   
#if VISUALIZE == 1

    glUseProgram(program[PASS_THROUGH]);

    glEnableVertexAttribArray(positionLocation);
    glBindBuffer(GL_ARRAY_BUFFER, planetVBO);
    glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0); 

	glEnableVertexAttribArray(velocityLocation);
	glBindBuffer(GL_ARRAY_BUFFER, velocityVBO);
	glVertexAttribPointer((GLuint)velocityLocation, 3, GL_FLOAT, GL_FALSE, 0, 0); 

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planetIBO);
   
    glPointSize(4.0f); 
    glDrawElements(GL_POINTS, N_FOR_VIS, GL_UNSIGNED_INT, 0);

    glDisableVertexAttribArray(positionLocation);
//	Draw_Axes();
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

void mouseMotion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

/*
	if (button_mask & 0x01) 
	{// left button*/
		viewPhi   += 0.005f*dx;
		viewTheta += 0.005f*dy;
		seekTarget.x = 400.0f*sin(viewTheta)*sin(viewPhi);
		seekTarget.y = 400.0f*cos(viewTheta);
		seekTarget.z = 400.0f*sin(viewTheta)*cos(viewPhi);
/*
		renderCam->views[0].x = sin(viewTheta)*sin(viewPhi);
		renderCam->views[0].y = cos(viewTheta);
		renderCam->views[0].z = sin(viewTheta)*cos(viewPhi);*/

//	} 
/*
	if (button_mask & 0x02) 
	{// middle button
		renderCam->positions[0] += 0.02f*dx*glm::cross(renderCam->views[0], renderCam->ups[0]);
		renderCam->positions[0] += 0.02f*dy*glm::cross(glm::cross(renderCam->views[0], renderCam->ups[0]), renderCam->views[0]);
	}
	if (button_mask & 0x04)
	{// right button
		renderCam->positions[0] -= 0.02f*dy*renderCam->views[0];
	}

	iterations = 0;
	for(int i=0; i<renderCam->resolution.x*renderCam->resolution.y; i++){
		renderCam->image[i] = glm::vec3(0,0,0);
	}*/

	mouse_old_x = x;
	mouse_old_y = y;
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

	timeSinceLastFrame = glutGet(GLUT_ELAPSED_TIME);
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


void initVAO(void)
{

    GLfloat *bodies     = new GLfloat[4*(N_FOR_VIS)];
	GLfloat *velocities = new GLfloat[3*(N_FOR_VIS)];
    GLuint *bindices    = new GLuint [N_FOR_VIS];

    for(int i = 0; i < N_FOR_VIS; i++)
    {
        bodies[4*i+0] = 0.0f;
        bodies[4*i+1] = 0.0f;
        bodies[4*i+2] = 0.0f;
        bodies[4*i+3] = 1.0f;

		velocities[3*i+0] = 0.0f;
		velocities[3*i+1] = 0.0f;
		velocities[3*i+2] = 0.0f;

        bindices[i] = i;
    }

    glGenBuffers(1, &planetVBO);
	glGenBuffers(1, &velocityVBO);
    glGenBuffers(1, &planetIBO);

    glBindBuffer(GL_ARRAY_BUFFER, planetVBO);
    glBufferData(GL_ARRAY_BUFFER, 4*(N_FOR_VIS)*sizeof(GLfloat), bodies, GL_DYNAMIC_DRAW);
    
	glBindBuffer(GL_ARRAY_BUFFER, velocityVBO);
	glBufferData(GL_ARRAY_BUFFER, 3*(N_FOR_VIS)*sizeof(GLfloat), velocities, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planetIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (N_FOR_VIS)*sizeof(GLuint), bindices, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    delete[] bodies;
    delete[] bindices;
	delete[] velocities;
}

void initShaders(GLuint * program)
{
    GLint location;
    
    program[1] = glslUtility::createProgram("shaders/planetVS.glsl", "shaders/planetGS.glsl", "shaders/planetFS.glsl", attributeLocations, 1);
    glUseProgram(program[1]);
    
    if ((location = glGetUniformLocation(program[1], "u_projMatrix")) != -1)
    {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
    if ((location = glGetUniformLocation(program[1], "u_cameraPos")) != -1)
    {
        glUniform3fv(location, 1, &cameraPosition[0]);
    }
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void shut_down(int return_code)
{
    exit(return_code);
}

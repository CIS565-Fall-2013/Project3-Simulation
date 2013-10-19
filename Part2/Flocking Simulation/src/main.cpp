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
    initPBO(&pbo);
    cudaGLRegisterBufferObject( planetVBO );
    
#if VISUALIZE == 1 
    initCuda(N_FOR_VIS);
#else
    initCuda(2*128);
#endif

    projection = glm::perspective(fovy, float(width)/float(height), zNear, zFar);
    view = glm::lookAt(cameraPosition, glm::vec3(0), glm::vec3(0,0,1));

    projection = projection * view;

    GLuint passthroughProgram;
    initShaders(program);

    glUseProgram(program[HEIGHT_FIELD]);
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
    cudaGLMapBufferObject((void**)&dptr, pbo);
    cudaGLMapBufferObject((void**)&dptrvert, planetVBO);

    // execute the kernel
    cudaFlockingUpdateWrapper(DT);
#if VISUALIZE == 1
//    cudaUpdatePBO(dptr, field_width, field_height);
    cudaUpdateVBO(dptrvert, field_width, field_height);
#endif
    // unmap buffer object
    cudaGLUnmapBufferObject(planetVBO);
    cudaGLUnmapBufferObject(pbo);
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

    /*char title[100];
    sprintf( title, "565 NBody sim [%0.2f fps]", fps );*/
	char title[100];
	sprintf( title, "Flocking Simulation [%0.2f fps] [%0.2fms] ", fps, executionTime);
    glutSetWindowTitle(title);

    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, field_width, field_height, 
            GL_RGBA, GL_FLOAT, NULL);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   
#if VISUALIZE == 1
    // VAO, shader program, and texture already bound
    //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    //glDrawElements(GL_TRIANGLES, 6*field_width*field_height,  GL_UNSIGNED_INT, 0);

    glUseProgram(program[HEIGHT_FIELD]);

    glEnableVertexAttribArray(positionLocation);
    glEnableVertexAttribArray(texcoordsLocation);
    
    glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0); 

    glBindBuffer(GL_ARRAY_BUFFER, planeTBO);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planeIBO);

    glDrawElements(GL_TRIANGLES, 6*field_width*field_height,  GL_UNSIGNED_INT, 0);

    glDisableVertexAttribArray(positionLocation);
    glDisableVertexAttribArray(texcoordsLocation);

    glUseProgram(program[PASS_THROUGH]);

    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, planetVBO);
    glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0); 

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planetIBO);
   
    glPointSize(4.0f); 
    glDrawElements(GL_POINTS, N_FOR_VIS+1, GL_UNSIGNED_INT, 0);

    glPointSize(1.0f);

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
    initTextures();
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

void initTextures()
{
    glGenTextures(1,&displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, field_width, field_height, 0, GL_RGBA, GL_FLOAT, NULL);
}

void initVAO(void)
{
    const int fw_1 = field_width-1;
    const int fh_1 = field_height-1;

    int num_verts = field_width*field_height;
    int num_faces = fw_1*fh_1;

    GLfloat *vertices  = new GLfloat[2*num_verts];
    GLfloat *texcoords = new GLfloat[2*num_verts]; 
    GLfloat *bodies    = new GLfloat[4*(N_FOR_VIS+1)];
    GLuint *indices    = new GLuint[6*num_faces];
    GLuint *bindices   = new GLuint[N_FOR_VIS+1];

    glm::vec4 ul(-5.0,-5.0,5.0,5.0);
    glm::vec4 lr(1.0,1.0,0.0,0.0);

    for(int i = 0; i < field_width; ++i)
    {
        for(int j = 0; j < field_height; ++j)
        {
            float alpha = float(i) / float(fw_1);
            float beta = float(j) / float(fh_1);
            vertices[(j*field_width + i)*2  ] = alpha*lr.x + (1-alpha)*ul.x;
            vertices[(j*field_width + i)*2+1] = beta*lr.y + (1-beta)*ul.y;
            texcoords[(j*field_width + i)*2  ] = alpha*lr.z + (1-alpha)*ul.z;
            texcoords[(j*field_width + i)*2+1] = beta*lr.w + (1-beta)*ul.w;
        }
    }

    for(int i = 0; i < fw_1; ++i)
    {
        for(int j = 0; j < fh_1; ++j)
        {
            indices[6*(i+(j*fw_1))    ] = field_width*j + i;
            indices[6*(i+(j*fw_1)) + 1] = field_width*j + i + 1;
            indices[6*(i+(j*fw_1)) + 2] = field_width*(j+1) + i;
            indices[6*(i+(j*fw_1)) + 3] = field_width*(j+1) + i;
            indices[6*(i+(j*fw_1)) + 4] = field_width*(j+1) + i + 1;
            indices[6*(i+(j*fw_1)) + 5] = field_width*j + i + 1;
        }
    }

    for(int i = 0; i < N_FOR_VIS+1; i++)
    {
        bodies[4*i+0] = 0.0f;
        bodies[4*i+1] = 0.0f;
        bodies[4*i+2] = 0.0f;
        bodies[4*i+3] = 1.0f;
        bindices[i] = i;
    }

    glGenBuffers(1, &planeVBO);
    glGenBuffers(1, &planeTBO);
    glGenBuffers(1, &planeIBO);
    glGenBuffers(1, &planetVBO);
    glGenBuffers(1, &planetIBO);
    
    glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
    glBufferData(GL_ARRAY_BUFFER, 2*num_verts*sizeof(GLfloat), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, planeTBO);
    glBufferData(GL_ARRAY_BUFFER, 2*num_verts*sizeof(GLfloat), texcoords, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planeIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6*num_faces*sizeof(GLuint), indices, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, planetVBO);
    glBufferData(GL_ARRAY_BUFFER, 4*(N_FOR_VIS+1)*sizeof(GLfloat), bodies, GL_DYNAMIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planetIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (N_FOR_VIS+1)*sizeof(GLuint), bindices, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    delete[] vertices;
    delete[] texcoords;
    delete[] bodies;
    delete[] indices;
    delete[] bindices;
}

void initShaders(GLuint * program)
{
    GLint location;
    program[0] = glslUtility::createProgram("shaders/heightVS.glsl", "shaders/heightFS.glsl", attributeLocations, 2);
    glUseProgram(program[0]);
    
    if ((location = glGetUniformLocation(program[0], "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }
    if ((location = glGetUniformLocation(program[0], "u_projMatrix")) != -1)
    {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
    if ((location = glGetUniformLocation(program[0], "u_height")) != -1)
    {
        glUniform1i(location, 0);
    }
    
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

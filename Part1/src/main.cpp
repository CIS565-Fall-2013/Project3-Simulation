/* Example N-Body simulation for CIS565 Fall 2013
 * Author: Liam Boone
 * main.cpp */

#include "main.h"
#include "cam.h"

#define N_FOR_VIS 25
#define DT 0.2
#define VISUALIZE 1

cam::cam()
{
	rad=glm::length(cameraPosition);
	theta=45.0;
	phi=45.0;
	pos = cameraPosition;
	view = glm::lookAt(pos, glm::vec3(0), glm::vec3(0,0,1));
	projectionView = projection * view;
}
void cam::reset()
{
	rad=glm::length(cameraPosition);
	theta=45.0;
	phi=45.0;
	pos = cameraPosition;
	view = glm::lookAt(pos, glm::vec3(0), glm::vec3(0,0,1));
	projectionView = projection * view;
}
void cam::setFrame()
{	
	pos.x=rad*sin(3.14*theta/180)*sin(3.14*phi/180);
	pos.y=rad*sin(3.14*theta/180)*cos(3.14*phi/180);
	pos.z=rad*cos(3.14*theta/180);

	view = glm::lookAt(pos, glm::vec3(0), glm::vec3(0,0,1));
	projectionView = projection * view;
}

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

    projectionView = projection * view;

    GLuint passthroughProgram;
    initShaders(program);

    glUseProgram(program[HEIGHT_FIELD]);
    glActiveTexture(GL_TEXTURE0);

    glEnable(GL_DEPTH_TEST);


    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
	glutMouseFunc(mouseClick); //check for mouse click
	glutMotionFunc(mouseMovement); //check for mouse movement
	glutPassiveMotionFunc(mouseMovementUpdate); //check for mouse movement
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
	if(integration == EULER)
		    cudaNBodyUpdateWrapper(DT);
	else if(integration == VELVERLET)
		cudaNBodyUpdateVelocityVerletWrapper(DT);


#if VISUALIZE == 1
    cudaUpdatePBO(dptr, field_width_pbo, field_height_pbo);
    cudaUpdateVBO(dptrvert, field_width, field_height);
#endif
    // unmap buffer object
    cudaGLUnmapBufferObject(planetVBO);
    cudaGLUnmapBufferObject(pbo);
}

int timebase = 0;
int frame = 0;

void display()
{
	mouseCam.setFrame();
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
	char integMethod[30];
	switch(integration)
	{
		case EULER:
			sprintf(integMethod,"Euler");
			break;
		case VELVERLET:
			sprintf(integMethod,"Velocity Verlet");
			break;
		case RK4:
			sprintf(integMethod,"Runge-Kutta 4");
			break;
	}
    sprintf( title, "565 NBody sim : %s : [%0.2f fps]", integMethod, fps );
    glutSetWindowTitle(title);

    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, field_width_pbo, field_height_pbo, 
            GL_RGBA, GL_FLOAT, NULL);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   
#if VISUALIZE == 1
    // VAO, shader program, and texture already bound
    //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    //glDrawElements(GL_TRIANGLES, 6*field_width*field_height,  GL_UNSIGNED_INT, 0);

    glUseProgram(program[HEIGHT_FIELD]);

	GLuint location;
    if ((location = glGetUniformLocation(program[HEIGHT_FIELD], "u_projMatrix")) != -1)
    {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projectionView[0][0]);
    }
	if ((location = glGetUniformLocation(program[HEIGHT_FIELD], "u_cameraPos")) != -1)
    {
		glUniform3fv(location, 1, &mouseCam.pos[0]);
    }

    glEnableVertexAttribArray(positionLocation);
    glEnableVertexAttribArray(texcoordsLocation);
    
    glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0); 

    glBindBuffer(GL_ARRAY_BUFFER, planeTBO);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planeIBO);
	
	glPatchParameteri(GL_PATCH_VERTICES, 3);
    glDrawElements(GL_PATCHES, 6*field_width*field_height,  GL_UNSIGNED_INT, 0);

    glDisableVertexAttribArray(positionLocation);
    glDisableVertexAttribArray(texcoordsLocation);

    glUseProgram(program[PASS_THROUGH]);

	location;
    if ((location = glGetUniformLocation(program[PASS_THROUGH], "u_projMatrix")) != -1)
    {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projectionView[0][0]);
    }
	if ((location = glGetUniformLocation(program[PASS_THROUGH], "u_cameraPos")) != -1)
    {
		glUniform3fv(location, 1, &mouseCam.pos[0]);
    }

    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, planetVBO);
    glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0); 

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planetIBO);
   
    glPointSize(4.0f); 
    glDrawElements(GL_POINTS, N_FOR_VIS+1, GL_UNSIGNED_INT, 0);

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
		case 'r':
			mouseCam.reset();
			break;
		case 'i':
			integration = (integration + 1 ) % NUMBEROFINTEGRATIONS;
			break;
		case 'I':
			integration = (integration - 1 ) % NUMBEROFINTEGRATIONS;
			break;
    }
}


void mouseMovement(int x, int y) {
	float diffx=motion*(x-lastx); //check the difference between the current x and the last x position
	float  diffy=motion*(y-lasty); //check the difference between the  current y and the last y position
	lastx=x; //set lastx to the current x position
	lasty=y; //set lasty to the current y position
	if(LMB)
	{
		mouseCam.theta -= diffy;
		if (mouseCam.theta > 180)
			mouseCam.theta = 179.9999;
		if (mouseCam.theta < 0)
			mouseCam.theta=0.0001;
		mouseCam.phi += diffx;
	}
	if(RMB)
	{
		mouseCam.rad -= (motion*diffy);
		if(mouseCam.rad < 1)
			mouseCam.rad = 1;
	}
}

void mouseMovementUpdate(int x, int y) {
	lastx=x; //set lastx to the current x position
	lasty=y; //set lasty to the current y position
}

void mouseClick(int button, int state, int x, int y) {

	if(button == GLUT_LEFT_BUTTON)
		LMB = (state==GLUT_DOWN);
	else if(button == GLUT_MIDDLE_BUTTON)
		MMB = (state==GLUT_DOWN);
	else if(button = GLUT_RIGHT_BUTTON)
		RMB = (state==GLUT_DOWN);
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
        int num_texels = field_width_pbo*field_height_pbo;
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
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, field_width_pbo, field_height_pbo, 0, GL_RGBA, GL_FLOAT, NULL);
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

    glm::vec4 ul(-1.0,-1.0,1.0,1.0);
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
    program[0] = glslUtility::createProgram("shaders/heightVS.glsl", "shaders/heightTCS.glsl", "shaders/heightTES.glsl", "shaders/heightGS.glsl", "shaders/heightFS.glsl", attributeLocations, 2);
    glUseProgram(program[0]);
    
    if ((location = glGetUniformLocation(program[0], "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }
    if ((location = glGetUniformLocation(program[0], "u_projMatrix")) != -1)
    {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projectionView[0][0]);
    }
    if ((location = glGetUniformLocation(program[0], "u_height")) != -1)
    {
        glUniform1i(location, 0);
    }
	if ((location = glGetUniformLocation(program[0], "windowDim")) != -1)
    {
        glUniform2f(location, width, height);
    }
    
    program[1] = glslUtility::createProgram("shaders/planetVS.glsl", "shaders/planetGS.glsl", "shaders/planetFS.glsl", attributeLocations, 1);
    glUseProgram(program[1]);
    
    if ((location = glGetUniformLocation(program[1], "u_projMatrix")) != -1)
    {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projectionView[0][0]);
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

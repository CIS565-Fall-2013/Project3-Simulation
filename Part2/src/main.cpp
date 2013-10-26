/* Example N-Body simulation for CIS565 Fall 2013
 * Author: Liam Boone
 * main.cpp */

#include "main.h"

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv)
{
    // Launch CUDA/GL

    init(argc, argv);

    cudaGLSetGLDevice( compat_getMaxGflopsDeviceId() );
	cudaGLRegisterBufferObject(planeVBO);
	cudaGLRegisterBufferObject(planeNBO);
    
	dx = width / (xdim-1);
	dz = height / (zdim-1);
	initCuda(xdim, zdim, dx, dz, initialY, mass);

    projection = glm::perspective(fovy, float(width)/float(height), zNear, zFar);
	view = glm::lookAt(cameraPosition, cameraRef, glm::vec3(0,1,0));

    projection = projection * view;

    initShaders(program);

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

    float *dptrvert=NULL;
	float *dptrnorm=NULL;
	cudaGLMapBufferObject((void**)&dptrvert, planeVBO);
	cudaGLMapBufferObject((void**)&dptrnorm, planeNBO);

    // execute the kernel
    update(dt);
	cudaUpdateVAO(dptrvert, dptrnorm);

    // unmap buffer object
	cudaGLUnmapBufferObject(planeVBO);
	cudaGLUnmapBufferObject(planeNBO);
}

int timebase = 0;
int frame = 0;
int time = 0;
int timePerFrame = 0;

void display()
{
 //   static float fps = 0;
 //   frame++;
	//

 //   if (time - timebase > 1000) {
 //       fps = frame*1000.0f/(time-timebase);
 //       timebase = time;
 //       frame = 0;
 //   }
	//timebase = time;

	timebase=glutGet(GLUT_ELAPSED_TIME);
    runCuda();
	time=glutGet(GLUT_ELAPSED_TIME);
	timePerFrame = time - timebase;
	timebase = time;

    char title[100];
    //sprintf( title, "GPU Cloth Sim [%0.2f fps]", fps );
	sprintf( title, "GPU Cloth Sim [%4d ms]", timePerFrame );
    glutSetWindowTitle(title);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnableVertexAttribArray(positionLocation);
	glEnableVertexAttribArray(normalLocation);
	glEnableVertexAttribArray(texcoordsLocation);
    
    glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
    glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0); 

	glBindBuffer(GL_ARRAY_BUFFER, planeNBO);
    glVertexAttribPointer((GLuint)normalLocation, 4, GL_FLOAT, GL_FALSE, 0, 0); 

    glBindBuffer(GL_ARRAY_BUFFER, planeTBO);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planeIBO);

    glDrawElements(GL_TRIANGLES, 6*(xdim-1)*(zdim-1),  GL_UNSIGNED_INT, 0);

    glDisableVertexAttribArray(positionLocation);
	glDisableVertexAttribArray(normalLocation);
	glDisableVertexAttribArray(texcoordsLocation);

    glutPostRedisplay();
    glutSwapBuffers();
}

void keyboard(unsigned char key, int x, int y)
{
    std::cout << key << std::endl;
    switch (key) 
    {
        case(27):
			shut_down(1);
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
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("GPU Cloth Sim");

    // Init GLEW
    glewInit();
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        /* Problem: glewInit failed, something is seriously wrong. */
        std::cout << "glewInit failed, aborting." << std::endl;
        shut_down(1);
    }

    initVAO();
    initTextures();
}

void initTextures()
{
    glGenTextures(1,&displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, xdim, zdim, 0, GL_RGBA, GL_FLOAT, NULL);
}

void initVAO(void)
{
    int num_verts = xdim * zdim;
    int num_faces = (xdim-1) * (zdim-1);

    GLfloat *vertices  = new GLfloat[4*num_verts];
	GLfloat *normals = new GLfloat[4*num_verts];
    GLfloat *texcoords = new GLfloat[2*num_verts]; 
    GLuint *indices    = new GLuint[6*num_faces];

	glm::vec4 ul(0,0,1.0,1.0);
    glm::vec4 lr(width,height,0.0,0.0);

    for(int i = 0; i < xdim; ++i)
    {
        for(int j = 0; j < zdim; ++j)
        {
            float alpha = float(i) / float(xdim-1);
            float beta = float(j) / float(zdim-1);

			vertices[(i*zdim + j)*4  ] = alpha*lr.x + (1-alpha)*ul.x;
			vertices[(i*zdim + j)*4+1] = initialY;
			vertices[(i*zdim + j)*4+2] = beta*lr.y + (1-beta)*ul.y;
			vertices[(i*zdim + j)*4+3] = 1;

			normals[(i*zdim +  j)*4  ] = 0;
			normals[(i*zdim +  j)*4+1] = 1;
			normals[(i*zdim +  j)*4+2] = 0;
			normals[(i*zdim +  j)*4+3] = 0;

			texcoords[(i*zdim+ j)*2  ] = alpha*lr.z + (1-alpha)*ul.z;
			texcoords[(i*zdim+ j)*2+1] = beta*lr.w + (1-beta)*ul.w;
        }
    }

    for(int i = 0; i < xdim-1; ++i)
    {
        for(int j = 0; j < zdim-1; ++j)
        {
			indices[6*(j+(i*(zdim-1)))    ] = zdim*i + j;
			indices[6*(j+(i*(zdim-1))) + 1] = zdim*i + j + 1;
			indices[6*(j+(i*(zdim-1))) + 2] = zdim*(i+1) + j;
			indices[6*(j+(i*(zdim-1))) + 3] = zdim*(i+1) + j;
			indices[6*(j+(i*(zdim-1))) + 4] = zdim*i + j + 1;
			indices[6*(j+(i*(zdim-1))) + 5] = zdim*(i+1) + j + 1;
        }
    }

    glGenBuffers(1, &planeVBO);
	glGenBuffers(1, &planeNBO);
    glGenBuffers(1, &planeTBO);
    glGenBuffers(1, &planeIBO);
    
    glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
    glBufferData(GL_ARRAY_BUFFER, 4*num_verts*sizeof(GLfloat), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, planeNBO);
    glBufferData(GL_ARRAY_BUFFER, 4*num_verts*sizeof(GLfloat), normals, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, planeTBO);
    glBufferData(GL_ARRAY_BUFFER, 2*num_verts*sizeof(GLfloat), texcoords, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planeIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6*num_faces*sizeof(GLuint), indices, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    delete[] vertices;
	delete[] normals;
    delete[] texcoords;
    delete[] indices;
}

void initShaders(GLuint program)
{
    GLint location;
    program = glslUtility::createProgram("shaders/planeVS.glsl", "shaders/planeFS.glsl", attributeLocations, 2);
    glUseProgram(program);
    
    if ((location = glGetUniformLocation(program, "u_projMatrix")) != -1)
    {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void shut_down(int return_code)
{
	freeCuda();
    exit(return_code);
}

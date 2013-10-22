/* Example N-Body simulation for CIS565 Fall 2013
 * Author: Liam Boone
 * main.cpp */

#include "main.h"

#if SIMMODE==CLOTHSIM
	#define N_FOR_VIS cloth_width*cloth_height
#elif SIMMODE==BASICSIM
	#define N_FOR_VIS 12
#elif SIMMODE==FLOCKSIM
	#define N_FOR_VIS 12000
#endif
#define DT 0.16
#define VISUALIZE 1

#define RESETFRAME 200

int mousex,mousey;

int mouseStatus;	//0: left hold, 1: mid hold, 2: right hold, -1 nothing

int totalframes;

bool paused;
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
	cudaGLRegisterBufferObject( planetVelBO );
    

#if SIMMODE==CLOTHSIM
	initCudaCloth(cloth_width,cloth_height);
#elif SIMMODE==BASICSIM
	initCuda(N_FOR_VIS);
#elif SIMMODE==FLOCKSIM
	initCudaFlock(N_FOR_VIS);
	targetPosition=glm::vec3(0,0,0);
#endif
	
	paused=false;
	srand(time(NULL));
	totalframes=0;
	cameraDir=-glm::normalize(cameraPosition);
    c_perspective = glm::perspective(fovy, float(width)/float(height), zNear, zFar);
    view = glm::lookAt(cameraPosition, cameraDir+cameraPosition, glm::vec3(0,0,1));
    projection = c_perspective * view;
	
	glm::vec3 right=glm::normalize(glm::cross(cameraDir,glm::vec3(0,0,1)));
	realup=glm::normalize(glm::cross(right,cameraDir));

    GLuint passthroughProgram;
    initShaders(program);

    glUseProgram(program[HEIGHT_FIELD]);
    glActiveTexture(GL_TEXTURE0);

    glEnable(GL_DEPTH_TEST);
	

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
	glutMotionFunc(getmousePos);
	glutMouseFunc(mouseAction);
	

    glutMainLoop();

    return 0;
}

//////////////////////////MOUSE ACTIONS/////////////////////
void getmousePos(int x, int y)
{
	
	if(mouseStatus==0)
	{
		if(y-mousey<-5) cameraPosition+=cameraDir*0.1f;
		else if(y-mousey>5)cameraPosition-=cameraDir*0.1f;
	}
	if(mouseStatus==1)
	{
		glm::vec3 right=glm::normalize(glm::cross(cameraDir,realup));
		if(x-mousex<-5) cameraPosition-=right*0.1f;
		else if(x-mousex>5) cameraPosition+=right*0.1f;
	}

	if(mouseStatus==2)
	{
		glm::vec3 centerpt=cameraPosition+cameraDir*5.0f;
		glm::vec3 right=glm::normalize(glm::cross(cameraDir,realup));
		if(x-mousex<-5) cameraPosition-=right*0.15f;
		else if(x-mousex>5) cameraPosition+=right*0.15f;
		cameraDir=glm::normalize(centerpt-cameraPosition);
		cameraPosition=centerpt-5.0f*cameraDir;
	}

	mousex=x;
	mousey=y;
	printf("%d %d\n",x,y);
}
void mouseAction(int button, int dir, int x, int y)
{
	//string statusname[4]={"nothing","left","middle","right"};
//	printf("%d %d\n", button, dir);
	if(dir==1) mouseStatus=-1;
	else
	{
		mouseStatus=button;
		mousex=x;
		mousey=y;
	}
	//printf("%d\n",mouseStatus);
	//printf("%s\n",statusname[mouseStatus+1]);
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
	float *dptrvel = NULL;
    cudaGLMapBufferObject((void**)&dptr, pbo);
    cudaGLMapBufferObject((void**)&dptrvert, planetVBO);
	cudaGLMapBufferObject((void**)&dptrvel, planetVelBO);

    // execute the kernel
#if SIMMODE==CLOTHSIM
    cudaClothUpdateWrapper(DT);
#elif SIMMODE==BASICSIM
	cudaNBodyUpdateWrapper(DT);
#elif SIMMODE==FLOCKSIM
	cudaFlockUpdateWrapper(DT,targetPosition);
#endif
#if VISUALIZE == 1
    cudaUpdatePBO(dptr, field_width, field_height);
    cudaUpdateVBO(dptrvert, field_width, field_height,targetPosition);
	cudaUpdateVelBO(dptrvel);
	//targetPosition=glm::vec3(dptrvert[4*N_FOR_VIS],dptrvert[4*N_FOR_VIS+1],0);
#endif
    // unmap buffer object
    cudaGLUnmapBufferObject(planetVBO);
	cudaGLUnmapBufferObject(planetVelBO);
    cudaGLUnmapBufferObject(pbo);
}

int timebase = 0;
int frame = 0;

void display()
{
	view = glm::lookAt(cameraPosition, cameraPosition+cameraDir, glm::vec3(0,0,1));
	projection = c_perspective * view;

	
    static float fps = 0;
    frame++;
	totalframes++;
    int time=glutGet(GLUT_ELAPSED_TIME);

    if (time - timebase > 1000) {
        fps = frame*1000.0f/(time-timebase);
        timebase = time;
        frame = 0;
    }

	GLuint location;

	if(SIMMODE==FLOCKSIM && totalframes%RESETFRAME==0)
	{
		targetPosition=glm::vec3(rand()%200,rand()%200,0);
//		targetPosition*=200.0f*(1/200.0f);	
		targetPosition-=glm::vec3(100.0f,100.0f,0.0f);
		targetPosition*=2.0f;
		printf("regenerate target position as (%f,%f,%f)\n",targetPosition.x,targetPosition.y,targetPosition.z);
	}
if(!paused)  runCuda();

    char title[100];
    sprintf( title, "565 NBody sim [%0.2f fps]", fps );
    glutSetWindowTitle(title);

    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, field_width, field_height, 
            GL_RGBA, GL_FLOAT, NULL);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   
#if VISUALIZE == 1
	//VAO, shader program, and texture already bound
	//	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	//glDrawElements(GL_TRIANGLES, 6*field_width*field_height,  GL_UNSIGNED_INT, 0);

#if SIMMODE==BASICSIM
	glUseProgram(program[HEIGHT_FIELD]);

	glEnableVertexAttribArray(positionLocation);
	glEnableVertexAttribArray(texcoordsLocation);

	glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
	glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0); 

	glBindBuffer(GL_ARRAY_BUFFER, planeTBO);
	glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planeIBO);
	if ((location = glGetUniformLocation(program[0], "u_projMatrix")) != -1)
	{
		glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	}


	glDrawElements(GL_TRIANGLES, 6*field_width*field_height,  GL_UNSIGNED_INT, 0);

	glDisableVertexAttribArray(positionLocation);
	glDisableVertexAttribArray(texcoordsLocation);
#endif
	glUseProgram(program[PASS_THROUGH]);

	glEnableVertexAttribArray(positionLocation);


	glBindBuffer(GL_ARRAY_BUFFER, planetVBO);
	glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0); 

#if SIMMODE==FLOCKSIM
	glEnableVertexAttribArray(velocityLocation);
	glBindBuffer(GL_ARRAY_BUFFER, planetVelBO);
	glVertexAttribPointer((GLuint)velocityLocation, 4, GL_FLOAT, GL_FALSE, 0, 0); 
#endif

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planetIBO);

	glPointSize(4.0f); 

	//GLuint location;
	//if ((location = glGetUniformLocation(program[1], "u_targetposition")) != -1)
	//{
	//	glUniform3fv(location, 1, &targetPosition[0]);
	//}
	if ((location = glGetUniformLocation(program[1], "u_cameraPos")) != -1)
	{
		glUniform3fv(location, 1, &cameraPosition[0]);
	}
	if ((location = glGetUniformLocation(program[1], "u_projMatrix")) != -1)
	{
		glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	}
#if SIMMODE==BASICSIM
	glDrawElements(GL_POINTS, N_FOR_VIS+1, GL_UNSIGNED_INT, 0);
#else
	glDrawElements(GL_POINTS, N_FOR_VIS, GL_UNSIGNED_INT, 0);
#endif
	glPointSize(1.0f);

	glDisableVertexAttribArray(positionLocation);
#if SIMMODE==FLOCKSIM
	glDisableVertexAttribArray(velocityLocation);
#endif

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
		case(' '):
			paused=!paused;
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
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, field_width, field_height, 0, GL_RGBA, GL_FLOAT, NULL);
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
            vertices[(j*field_width + i)*2  ] = (alpha*lr.x + (1-alpha)*ul.x)*2;
            vertices[(j*field_width + i)*2+1] = (beta*lr.y + (1-beta)*ul.y)*2;
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
	glGenBuffers(1, &planetVelBO);
    glGenBuffers(1, &planetIBO);
    
    glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
    glBufferData(GL_ARRAY_BUFFER, 2*num_verts*sizeof(GLfloat), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, planeTBO);
    glBufferData(GL_ARRAY_BUFFER, 2*num_verts*sizeof(GLfloat), texcoords, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planeIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6*num_faces*sizeof(GLuint), indices, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, planetVBO);
    glBufferData(GL_ARRAY_BUFFER, 4*(N_FOR_VIS+1)*sizeof(GLfloat), bodies, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, planetVelBO);
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
#if SIMMODE==CLOTHSIM
    program[1] = glslUtility::createProgram("shaders/planetVS.glsl", "shaders/planetGS_cloth.glsl", "shaders/planetFS.glsl", attributeLocations, 3);
#elif SIMMODE==BASICSIM
	program[1] = glslUtility::createProgram("shaders/planetVS.glsl", "shaders/planetGS_cloth.glsl", "shaders/planetFS.glsl", attributeLocations, 3);
#elif SIMMODE==FLOCKSIM
	program[1] = glslUtility::createProgram("shaders/planetVS.glsl", "shaders/planetGS_flock.glsl", "shaders/planetFS.glsl", attributeLocations, 3);
#endif
    glUseProgram(program[1]);
    
    if ((location = glGetUniformLocation(program[1], "u_projMatrix")) != -1)
    {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
    if ((location = glGetUniformLocation(program[1], "u_cameraPos")) != -1)
    {
        glUniform3fv(location, 1, &cameraPosition[0]);
    }
	if ((location = glGetUniformLocation(program[1], "u_targetposition")) != -1)
	{
		glUniform3fv(location, 1, &targetPosition[0]);
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

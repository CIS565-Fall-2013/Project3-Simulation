/* Example N-Body simulation for CIS565 Fall 2013
 * Author: Liam Boone
 * main.cpp */

#include "main.h"

#define N_FOR_VIS 300
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
    initPBO(&pbo);

	// Note the below function is deprecated as of CUDA 3.0
	// Thus, TODO: use cudaGraphicsGLRegisterBuffer instead!
	// Goal for this part is to let CUDA kernels be able to write information into the pbo and interact with OpenGL
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
    glActiveTexture(GL_TEXTURE0 + 0); // this uses displayImage as the active texture, so that the sampler in the VS will be able to sample this

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
	//////////////////////
	// Timing cuda call //
	//////////////////////
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
    float4 *dptr=NULL;
    float *dptrvert=NULL;
    cudaGLMapBufferObject((void**)&dptr, pbo);
    cudaGLMapBufferObject((void**)&dptrvert, planetVBO);

    // execute the kernel
    cudaNBodyUpdateWrapper(DT, target);
#if VISUALIZE == 1
    cudaUpdatePBO(dptr, field_width, field_height);
    cudaUpdateVBO(dptrvert, field_width, field_height);
#endif
    // unmap buffer object
    cudaGLUnmapBufferObject(planetVBO);
    cudaGLUnmapBufferObject(pbo);

	//////////////////////
	// Timing cuda call //
	//////////////////////
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("runCuda runtime: %3.1f ms \n", time);
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

	// call functions defined in kernel.cu
	if (!isPaused)
		runCuda();

    char title[100];
    sprintf( title, "565 NBody sim [%0.2f fps]", fps );
    glutSetWindowTitle(title);

	// If a non-zero named buffer object is bound to the GL_PIXEL_UNPACK_BUFFER target (see glBindBuffer) while a texture image is specified, 
	// data (last parameter of glTextSubImage2D) is treated as a byte offset into the buffer object's data store.
	// In other words, pbo currently contains the new set of height values computed by the runCuda call above. displayImage is told to access
	// pbo for values with no offset. Therefore, the calls below to heightVS will allow heightVS to sample the updated texture (displayImage)
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, field_width, field_height, 
            GL_RGBA, GL_FLOAT, NULL);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   
#if VISUALIZE == 1
    //VAO, shader program, and texture already bound
    //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    //glDrawElements(GL_TRIANGLES, 6*field_width*field_height,  GL_UNSIGNED_INT, 0);

    glUseProgram(program[HEIGHT_FIELD]); // "shaders/heightVS.glsl", "shaders/heightFS.glsl"

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

    glUseProgram(program[PASS_THROUGH]); // "shaders/planetVS.glsl", "shaders/planetGS.glsl", "shaders/planetFS.glsl"

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

// keyboard call back function
void keyboard(unsigned char key, int x, int y)
{
    std::cout << key << std::endl;

	float targetSpeed = 10.0f;

    switch (key) 
    {
        case(27):
            exit(1);
            break;
		case 'p':
		case 'P':
			isPaused = !isPaused;
			break;
		// move target in +z
		case 'w':
		case 'W':
			target = vec3(target.x, target.y, target.z+targetSpeed);
			break;
		// move target in -x
		case 'a':
		case 'A':
			target = vec3(target.x-targetSpeed, target.y, target.z);
			break;
		// move target in -z
		case 's':
		case 'S':
			target = vec3(target.x, target.y, target.z-targetSpeed);
			break;
		// move target in +x
		case 'd':
		case 'D':
			target = vec3(target.x+targetSpeed, target.y, target.z);
			break;
		// move target in -y
		case 'q':
		case 'Q':
			target = vec3(target.x, target.y-targetSpeed, target.z);
			break;
		// move target in +y
		case 'e':
		case 'E':
			target = vec3(target.x, target.y+targetSpeed, target.z);
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
        
		// Note the below function is deprecated as of CUDA 3.0
		// Thus, TODO: use cudaGraphicsGLRegisterBuffer instead!
		// Goal for this part is to let CUDA kernels be able to write information into the pbo and interact with OpenGL
		cudaGLRegisterBufferObject( *pbo );
    }
}

// set up displayImage as a texture
// GL_TEXTURE_2D​: Images in this texture all are 2-dimensional. They have width and height, but no depth.
void initTextures()
{
    glGenTextures(1,&displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage); // binds a type of texture to the texture name generated by glGenTextures

	// See here for a description of the parameters for glTextParameteri http://www.opengl.org/wiki/GLAPI/glTexParameter
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, field_width, field_height, 0, GL_RGBA, GL_FLOAT, NULL);
}

// Initialize vertex array object (VAO), an OpenGL object that encapsulates all the state needed to specify vertex data.
// As with OpenGL objects, VAOs have the usual creation, destruction, and binding functions: glGenVertexArrays, glDeleteVertexArrays,
// and glBindVertexArray.
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

	// changing the size of the plane.
	//float scale = 2.0f;
	//ul.x = scale * ul.x;
	//ul.y = scale * ul.y;
	//lr.x = scale * lr.x;
	//lr.y = scale * lr.y;

	// linearly interpolate between the the start of the plane and the max resolution of the plane
	// assuming the upper left corner of the plane is positioned at ul and lower right corner is positioned at lr.
    for(int i = 0; i < field_width; ++i)
    {
        for(int j = 0; j < field_height; ++j)
        {
            float alpha = float(i) / float(fw_1);
            float beta = float(j) / float(fh_1);

			// vertices go from -1 to 1 and -1 to 1 for a 2D plane
			vertices[(j*field_width + i)*2  ] = alpha*lr.x + (1-alpha)*ul.x;
            vertices[(j*field_width + i)*2+1] = beta*lr.y + (1-beta)*ul.y;

			// texcoords go from 0 to 1 and 0 to 1
            texcoords[(j*field_width + i)*2  ] = alpha*lr.z + (1-alpha)*ul.z;
            texcoords[(j*field_width + i)*2+1] = beta*lr.w + (1-beta)*ul.w;
        }
    }

	// indexing 2 triangles at a time for the plane drawing.
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

	// initializing position and indices for each body.
    for(int i = 0; i < N_FOR_VIS+1; i++)
    {
        bodies[4*i+0] = 0.0f;
        bodies[4*i+1] = 0.0f;
        bodies[4*i+2] = 0.0f;
        bodies[4*i+3] = 1.0f;
        bindices[i] = i;
    }

	// generate buffer objects for planeVBO, TBO, IBO and planetVBO and IBO
    glGenBuffers(1, &planeVBO);
    glGenBuffers(1, &planeTBO);
    glGenBuffers(1, &planeIBO);
    glGenBuffers(1, &planetVBO);
    glGenBuffers(1, &planetIBO);
    
    glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
    glBufferData(GL_ARRAY_BUFFER, 2*num_verts*sizeof(GLfloat), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, planeTBO);
    glBufferData(GL_ARRAY_BUFFER, 2*num_verts*sizeof(GLfloat), texcoords, GL_STATIC_DRAW);

	// plane is drawn using indices, thus use GL_ELEMENT_ARRAY_BUFFER
	// see display() for the glDrawElements call
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planeIBO); 
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6*num_faces*sizeof(GLuint), indices, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, planetVBO);
    glBufferData(GL_ARRAY_BUFFER, 4*(N_FOR_VIS+1)*sizeof(GLfloat), bodies, GL_DYNAMIC_DRAW);
    
	// planets are drawn using indices, thus use GL_ELEMENT_ARRAY_BUFFER
	// see display() for the glDrawElements call
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
	// binding textures to samplers
    if ((location = glGetUniformLocation(program[0], "u_height")) != -1)
    {
		// // Texture unit 0 is for location
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

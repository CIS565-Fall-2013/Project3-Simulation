#include <GL/glew.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "glRoutine.h"
#include "variables.h"
#include "util.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "kernel.h"


using namespace std;

typedef struct {
    GLuint bo;
    GLuint typeSize;
    struct cudaGraphicsResource *cudaResource;
} mappedBuffer_t;

//float vertexData[] = { 
//    //vertex position
//    -1.0f, -1.0f, 0.0f,
//    1.0f, -1.0f, 0.0f,
//    1.0f, 1.0f, 0.0f,
//    -1.0f, -1.0f, 0.0f,
//    1.0f, 1.0f, 0.0f,
//    -1.0f,1.0f, 0.0f,
//
//    //texture coordinates
//    0.0f, 1.0f,
//    1.0f, 1.0f,
//    1.0f, 0.0f,
//    0.0f, 1.0f,
//    1.0f, 0.0f,
//    0.0f,0.0f};

float* pVertexData = 0;
int nRow = 10;
int nCol = 10;
int vertexNum = 0;
unsigned int* pVertexIdxData = 0;
int vertexIdxNum = 0;
float2 restLength;
float meshDim = 1.0f;

float dt = 0.01f;

glm::vec3 camera( 2.0f, 2.0f, 2.0f );
glm::vec3 center( 0.0f, 0.0f, 0.0f );
glm::vec3 up( 0.0f, 1.0f, 0.0f );
glm::mat4 projection;
glm::mat4 modelview;
glm::mat3 normalMat;

//GLSL shader related variables
GLuint fragShader;
GLuint vertShader;
GLuint shaderProg;
GLint texLoc;
GLint modelviewMatLoc;
GLint projectionMatLoc;
GLint normalMatLoc;

//OpenGL buffer objects & Cuda-OpenGL interop objects
cudaGraphicsResource* pboResource = 0;
cudaGraphicsResource* vboResource = 0;
GLuint pbo = 0;  //handle to pixel buffer object
GLuint vbo = 0;  //handle to vertex buffer object
GLuint ibo = 0;  //handle to index buffer
GLuint vao = 0;  //handle to vertex array object
GLuint texID = 0;



void glut_display()
{
    updateVelWrapper( vboResource, restLength, nCol+1, nRow+1 );
    updatePosWrapper( vboResource, dt, nCol+1, nRow+1 );

    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
    glClearColor( 0, 0, 0, 0 );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    modelview = glm::lookAt( camera, center, up );
    normalMat = glm::inverse( glm::mat3(modelview) );

    glUniformMatrix4fv( projectionMatLoc, 1, GL_FALSE, &projection[0][0] );
    glUniformMatrix4fv( modelviewMatLoc, 1, GL_FALSE, &modelview[0][0] );
    glUniformMatrix3fv( normalMatLoc, 1, GL_TRUE, &normalMat[0][0] );

    //glActiveTexture( GL_TEXTURE0 );
    //glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo );
    //glBindTexture( GL_TEXTURE_2D, texID );
    //glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, win_w, win_h, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
   // glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, win_width, win_height, GL_RGBA, GL_FLOAT, 0 );
    //glBindBuffer( GL_PIXEL_UNPACK_BUFFER,0 );
    
    glBindVertexArray( vao );
    //glDrawArrays( GL_TRIANGLES, 0, 6 );
    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, ibo );
    glDrawElements( GL_TRIANGLES, vertexIdxNum, GL_UNSIGNED_INT, (void*)0 );


    glBindVertexArray( 0 );
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);
    //glBindTexture(GL_TEXTURE_2D, 0);
    glutSwapBuffers();
}

void glut_idle()
{
    glutSetWindow(win_id);
    glutPostRedisplay();
}

void glut_reshape( int w, int h )
{
    win_height = h;
    win_width = w;
    //rebuild the pixel buffer object
    initPBO();

    //re-calculate the dimensions of grids
    glViewport( 0, 0, w, h );
    projection = glm::perspective( 60.0f, (float)win_width/(float)win_height, 0.1f, 10.0f ); 

    glUniformMatrix4fv( projectionMatLoc, 1, GL_FALSE, &projection[0][0] );
}

void glut_keyboard( unsigned char key, int x, int y)
{

}

int initPBO()
{
    float clearData[] = {0.0f, 0.0f,0.0f,0.0f};
    if( pbo ) 
    {
        //ungister from CUDA context
        cudaGraphicsUnregisterResource( pboResource);
        //destroy the existing pbo 
        glDeleteBuffers( 1, &pbo ); pbo = 0;
        glDeleteTextures( 1, &texID ); texID = 0;
    }

    //create a PBO
    glGenBuffers(1, &pbo);
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo );
    //glBufferData( GL_PIXEL_UNPACK_BUFFER, sizeof( GLubyte) * win_w * win_h * 4, NULL, GL_STREAM_DRAW );
    glBufferData( GL_PIXEL_UNPACK_BUFFER, sizeof( GLfloat) * win_width * win_height * 4, NULL, GL_STREAM_DRAW );
    //glClearBufferData( GL_PIXEL_UNPACK_BUFFER, GL_RGBA32F, GL_BGRA, GL_FLOAT, clearData );
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );

    //register with CUAD context
    cudaGraphicsGLRegisterBuffer( &pboResource, pbo, cudaGraphicsMapFlagsWriteDiscard );

    //create texture for displaying the rendering result
    glActiveTexture( GL_TEXTURE0);
    glGenTextures( 1, &texID );
    glBindTexture( GL_TEXTURE_2D, texID );
    //glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, win_w, win_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL );
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, win_width, win_height, 0, GL_BGRA, GL_FLOAT, NULL );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glBindTexture( GL_TEXTURE_2D, 0 );

    return 0;
}


int initVertexData()
{
    createGridMesh( nCol, nRow );
    restLength.x = meshDim / nCol;
    restLength.y = meshDim / nRow;

    if( vbo )
    {
        cudaGraphicsUnregisterResource( vboResource );
        glDeleteBuffers( 1, &vbo );
    }
    
    glGenBuffers( 1, &vbo );
    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glBufferData( GL_ARRAY_BUFFER, sizeof(float)*vertexNum*4, pVertexData, GL_DYNAMIC_DRAW );
    cudaGraphicsGLRegisterBuffer( &vboResource, vbo, cudaGraphicsMapFlagsNone );

    if( ibo )
        glDeleteBuffers( 1, &ibo);
    glGenBuffers( 1, &ibo );
    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, ibo );
    glBufferData( GL_ELEMENT_ARRAY_BUFFER, vertexIdxNum * sizeof( unsigned int ), pVertexIdxData, GL_STATIC_DRAW );
    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
    //create and setup the vertex array object
    glGenVertexArrays( 1, &vao );
    glBindVertexArray( vao );

    glEnableVertexAttribArray(0);
    glVertexAttribPointer( 0, 4, GL_FLOAT, GL_FALSE, 0, (GLubyte*)NULL );

    //glEnableVertexAttribArray(1);
    //glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, 0, (GLvoid*)(sizeof( float) * 18) );

    glBindBuffer( GL_ARRAY_BUFFER,0 );
    glBindVertexArray(0);

    return 0;
}

int createGridMesh( unsigned int nCol, unsigned int nRow )
{
    size_t elmSize= 4;
    float xoffset = meshDim / nCol;
    float yoffset = meshDim / nRow;

    vertexNum = (nCol+1)*(nRow+1);
    vertexIdxNum = 6 * nCol * nRow;

    pVertexData = new float[vertexNum*elmSize*sizeof(float)];
    pVertexIdxData = new unsigned int[ vertexIdxNum ];

    for( int h = 0; h <= nRow ; ++h )
        for( int w = 0; w <= nCol; ++w )
        {
            pVertexData[ ( h*(nCol+1) + w )*elmSize     ] = xoffset*w;
            pVertexData[ ( h*(nCol+1) + w )*elmSize + 1 ] = 0.0f;
            pVertexData[ ( h*(nCol+1) + w )*elmSize + 2 ] = yoffset*(nRow-h);
            pVertexData[ ( h*(nCol+1) + w )*elmSize + 3 ] = 1.0f;
        }

    for( int h = 0; h < nRow;  ++h )
    {
        for( int w = 0; w < nCol; ++w )
        {
            pVertexIdxData[ (h*nCol + w)*6   ] = h*(nCol+1)     + w;
            pVertexIdxData[ (h*nCol + w)*6 +1 ] = (h+1)*(nCol+1) + w;
            pVertexIdxData[ (h*nCol + w)*6 +2 ] = h*(nCol+1)     + w+1;

            pVertexIdxData[ (h*nCol + w)*6 +3 ] = h*(nCol+1)     + w+1;
            pVertexIdxData[ (h*nCol + w)*6 +4 ] = (h+1)*(nCol+1) + w;
            pVertexIdxData[ (h*nCol + w)*6 +5 ] = (h+1)*(nCol+1) + w+1;
        }
    }

    return 0;
}

char* readFromFile( const char* filename, int* len )
{
	std::ifstream file;
	file.open( filename, std::ios::binary );
	if( !file.is_open() )
	{
        cerr<<"Read shader source failed!\n";
		return NULL;
	}

	(*len)=0;
	file.seekg( 0, std::ios::end );
	(*len) = file.tellg();
	file.seekg( 0, std::ios::beg );

	if( (*len) == 0 )
	{
		cerr<<"Shader source zero length!\n";
		return NULL;
	}

	char* buf = new char[(*len)+1];
	file.read( buf, *len );
    buf[(*len)] = '\0';
	return buf;
} 

int checkAndShowShaderStatus( const char* obj_name, GLuint obj, int check_mode )
{
	int err_code;
	int max_infolen;
	if( check_mode == 1 )
	{
		glGetShaderiv( obj, GL_COMPILE_STATUS, &err_code );
		glGetShaderiv( obj, GL_INFO_LOG_LENGTH, &max_infolen ); 
	}
	else 
	{	
		glGetProgramiv( obj, GL_LINK_STATUS, &err_code );
		glGetProgramiv( obj, GL_INFO_LOG_LENGTH, &max_infolen ); 
	}
	if( err_code != GL_TRUE )
	{
		int infolen;
		char *infobuf = new char[max_infolen+1];
		glGetShaderInfoLog( obj,max_infolen,&infolen, infobuf);
        cerr<<"ERROR("<<obj_name<<"):";
        cerr<<infobuf<<endl;
		delete [] infobuf;
		return -1;

	}
	else
		return 0;

}


GLuint initShader( GLenum shaderType, const char* shaderSourceFile )
{
    int src_len;
    GLuint shader = glCreateShader( shaderType );
 
    const char* source = readFromFile( shaderSourceFile, &src_len );
    if( source == NULL )
        return 0;

    glShaderSource( shader, 1, &source, NULL );
    glCompileShader( shader );

    delete [] source;

    if( checkAndShowShaderStatus( shaderSourceFile, shader, 1 ) != 0 )
        return 0;

    return shader;

}

GLuint initShaderProg( GLuint vertShader, GLuint fragShader )
{
    GLuint prog = glCreateProgram();

    glAttachShader( prog, vertShader );
    glAttachShader( prog, fragShader );

    glLinkProgram( prog );
    if( checkAndShowShaderStatus( "Shader Program", prog, 2 ) != 0 )
        return 0;

    //Obtain locations of shader variables
    texLoc = glGetUniformLocation( prog, "SpriteTex" );
    if( texLoc >= 0)
    {
        cerr<<"Uniform variable text1 unavailable!\n";
    }
    else
        glUniform1i( texLoc, 0 ); //set the sampler location to texture unit 0

    modelviewMatLoc = glGetUniformLocation(prog, "ModelViewMatrix" );
    projectionMatLoc = glGetUniformLocation(prog, "ProjectionMatrix" );
    normalMatLoc = glGetUniformLocation( prog, "NormalMatrix" );

    return prog;
}

int initGL()
{
    //init shader
    fragShader = initShader( GL_FRAGMENT_SHADER, "cloth.frag" );
    vertShader = initShader( GL_VERTEX_SHADER, "cloth.vert" );
    shaderProg = initShaderProg( vertShader, fragShader );

    if( fragShader < 1 || vertShader < 1  || shaderProg < 1 )
        return -1;

    glUseProgram(shaderProg);

    //init vbo
    if( initVertexData() )
        return -1;

    //init pbo
    //if( initPBO() )
    //    return -1;
    initCUDABuffer( vboResource, nCol+1, nRow+1 );

    return 0;
}

void cameraEventHandler( int id )
{
}


void cleanUpGL()
{
    if( pbo )
    {
        cudaGraphicsUnregisterResource( pboResource );
        glDeleteBuffers( 1, &pbo );
        pbo = 0;
    }
    if( vbo )
    {
        cudaGraphicsUnregisterResource( vboResource );
        glDeleteBuffers( 1, &vbo );
        vbo = 0;
    }
    if( ibo )
    {
        glDeleteBuffers( 1, &ibo );
        ibo = 0;
    }
    if( texID )
    {
        glDeleteTextures( 1, &texID );
        texID = 0;
    }

    glDeleteVertexArrays( 1, &vao );
    vao = 0;

    if( pVertexData )
        delete [] pVertexData ;
    pVertexData = 0;

    if( pVertexIdxData )
        delete [] pVertexIdxData;
    pVertexIdxData = 0;

    cleanupCUDA();
}
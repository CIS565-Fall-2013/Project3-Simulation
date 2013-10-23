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
#include "glm.h"
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

//Cloth mesh data
float* pVertexData = 0;
float* pVertexNormal = 0;
int nRow = 50;
int nCol = 50;
int vertexNum = 0;
unsigned int* pVertexIdxData = 0;
int vertexIdxNum = 0;
float2 restLength;
float meshDimX = 3.0f;
float meshDimY = 3.0f;

glm::vec4 Kd( 1.0f, 1.0f, 1.0f, 1.0f );
glm::vec4 Ks( 1.0f, 1.0f, 1.0f, 1.0f );
float shininess = 32.0f;

float dt = 0.01f;
float elapse = 0.0f;
float windFactor = 0.0f;

glm::vec3 camera( 0.0f, 1.0f, 4.0f );
glm::vec3 center( 0.0f, 0.0f, 0.0f );
glm::vec3 up( 0.0f, 1.0f, 0.0f );
glm::mat4 projection;
glm::mat4 modelview;
glm::mat3 normalMat;

//sphere data
float* pSphereVertex = 0;
unsigned int* pSphereVertexIdx = 0;
float* pSphereNormal = 0;
int sphereVertexNum = 0;
int sphereVertexIdxNum = 0;
float radius = 0.2f;
bool bSphereMove = false;
glm::vec3 sphereTranslate(0.0f,0.0f,1.0f);
glm::vec4 ballKd( 0.5255f, 0.4314f, 0.0314f, 1.0f );
glm::vec4 ballKs( 0.35f, 0.35f, 0.35f, 1.0f );
float ballShininess = 32.0f;

//light data
glm::vec4 lightPos( 0.0f, 11.0f, 11.0f, 1.0f );

//GLSL shader related variables
GLuint fragShader;
GLuint vertShader;
GLuint shaderProg;
GLint texLoc;
GLint modelviewMatLoc;
GLint projectionMatLoc;
GLint normalMatLoc;
GLint kdLoc;
GLint ksLoc;
GLint shLoc;
GLint keLoc;
GLint lightPosLoc;

//OpenGL buffer objects & Cuda-OpenGL interop objects
cudaGraphicsResource* pboResource = 0;
cudaGraphicsResource* vboResource = 0;
cudaGraphicsResource* vboNResource = 0;
GLuint pbo = 0;  //handle to pixel buffer object
GLuint vbo = 0;  //handle to vertex buffer object
GLuint vboN = 0;
GLuint vao = 0;  //handle to vertex array object
GLuint texID = 0;

GLuint vbo_sphere = 0;
GLuint vbo_sphereN = 0;
GLuint ibo_sphere = 0;

GLuint ibo = 0;  //handle to index buffer

void glut_display()
{
    updateVelWrapper( vboResource, restLength, nCol+1, nRow+1 );
    updatePosWrapper( vboResource, vboNResource, dt, elapse, windFactor, restLength, nCol+1, nRow+1, sphereTranslate, 1.0f);
    updateNormalWrapper( vboNResource, nCol+1, nRow+1 );
    elapse += dt;
    
    //glPolygonMode( GL_FRONT, GL_LINE );
    //glEnable( GL_CULL_FACE );
    glEnable( GL_DEPTH_TEST );
    glCullFace( GL_BACK );
    glClearColor( 135.0f/255.0f, 206.0f/255.0f, 250.0f/255.0f, 0.0f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        //calcuate light in eyeSpace
    glm::vec4 lightInEyeSpace = modelview * lightPos;
    glUniform4fv( lightPosLoc,1, &lightInEyeSpace[0] );

    modelview = glm::lookAt( camera, center, up );
    normalMat = glm::inverse( glm::mat3(modelview) );

    glUseProgram( shaderProg );
    glUniformMatrix4fv( projectionMatLoc, 1, GL_FALSE, &projection[0][0] );
    glUniformMatrix4fv( modelviewMatLoc, 1, GL_FALSE, &modelview[0][0] );
    glUniformMatrix3fv( normalMatLoc, 1, GL_TRUE, &normalMat[0][0] );
    glUniform4fv( kdLoc, 1, &Kd[0] );
    glUniform4fv( ksLoc, 1, &Ks[0] );
    glUniform1f( shLoc, shininess );

    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glBindVertexArray( vao );

    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glVertexAttribPointer( 0, 4, GL_FLOAT, GL_FALSE, 0, (GLubyte*)NULL );
    glEnableVertexAttribArray(0);
    
    glBindBuffer( GL_ARRAY_BUFFER, vboN );
    glVertexAttribPointer( 1, 3, GL_FLOAT, GL_FALSE, 0, (GLubyte*) NULL );
    glEnableVertexAttribArray(1);

    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, ibo );
    glDrawElements( GL_TRIANGLES, vertexIdxNum, GL_UNSIGNED_INT, (void*)0 );


    modelview = glm::translate( modelview, sphereTranslate );
    normalMat = glm::transpose( glm::inverse( glm::mat3(modelview) ) );

    glUniformMatrix4fv( modelviewMatLoc, 1, GL_FALSE, &modelview[0][0] );
    glUniformMatrix3fv( normalMatLoc, 1, GL_FALSE, &normalMat[0][0] );
    glUniform4fv( kdLoc, 1, &ballKd[0] );
    glUniform4fv( ksLoc, 1, &ballKs[0] );
    glUniform1f( shLoc, ballShininess );
    
    
    glBindBuffer( GL_ARRAY_BUFFER, vbo_sphere);
    glVertexAttribPointer( 0, 4, GL_FLOAT, GL_FALSE, 0, (GLubyte*)NULL );
    glEnableVertexAttribArray(0);
    
    glBindBuffer( GL_ARRAY_BUFFER, vbo_sphereN );
    glVertexAttribPointer( 1, 3, GL_FLOAT, GL_FALSE, 0, (GLubyte*) NULL );
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_sphere );
    glDrawElements( GL_TRIANGLES, sphereVertexIdxNum, GL_UNSIGNED_INT, (void*)0 );

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    glBindVertexArray( 0 );
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
    projection = glm::perspective( 75.0f, (float)win_width/(float)win_height, 1.0f, 20.0f ); 

    glUniformMatrix4fv( projectionMatLoc, 1, GL_FALSE, &projection[0][0] );
}

void glut_keyboard( unsigned char key, int x, int y)
{

}

int start_x, start_y;
void glut_mouse(int button, int state, int x, int y )
{
    if( button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN )
    {
        windFactor = 1.0f;
    }
    else if( button == GLUT_RIGHT_BUTTON && state == GLUT_UP )
    {
        windFactor = 0.0f;
    }
    else if( button == GLUT_LEFT_BUTTON && state == GLUT_DOWN )
    {
        bSphereMove = true;
        start_x = x;
        start_y = y;
    }
    else if( button == GLUT_LEFT_BUTTON && state == GLUT_UP )
    {
        bSphereMove = false;
    }
}

void glut_mousemotion( int x, int y )
{
    int offset_x, offset_y;
    if( bSphereMove )
    {
        offset_x = x - start_x;
        offset_y = y - start_y;
        if( abs(offset_x) > abs(offset_y) )
            sphereTranslate.x += 0.01f * offset_x;
        else
            sphereTranslate.z += 0.01f * offset_y;

        start_x = x;
        start_y = y;
    }
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
    restLength.x = meshDimX / nCol;
    restLength.y = meshDimY / nRow;

    if( vbo )
    {
        cudaGraphicsUnregisterResource( vboResource );
        glDeleteBuffers( 1, &vbo );
    }
    
    glGenBuffers( 1, &vbo );
    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glBufferData( GL_ARRAY_BUFFER, sizeof(float)*vertexNum*4, pVertexData, GL_DYNAMIC_DRAW );
    cudaGraphicsGLRegisterBuffer( &vboResource, vbo, cudaGraphicsMapFlagsNone );

    if( vboN )
    {
        cudaGraphicsUnregisterResource( vboNResource );
        glDeleteBuffers( 1, &vboN );
    }
    
    glGenBuffers( 1, &vboN );
    glBindBuffer( GL_ARRAY_BUFFER, vboN );
    glBufferData( GL_ARRAY_BUFFER, sizeof(float)*vertexNum*3, pVertexNormal, GL_DYNAMIC_DRAW );
    cudaGraphicsGLRegisterBuffer( &vboNResource, vboN, cudaGraphicsMapFlagsNone );

    if( ibo )
        glDeleteBuffers( 1, &ibo);
    glGenBuffers( 1, &ibo );
    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, ibo );
    glBufferData( GL_ELEMENT_ARRAY_BUFFER, vertexIdxNum * sizeof( unsigned int ), pVertexIdxData, GL_STATIC_DRAW );
    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );

    //create and setup the vertex array object
    glGenVertexArrays( 1, &vao );
    glBindVertexArray( vao );

    glBindBuffer( GL_ARRAY_BUFFER,0 );

    return 0;
}

int createGridMesh( unsigned int nCol, unsigned int nRow )
{
    size_t elmSize= 4;
    float xoffset = meshDimX / nCol;
    float yoffset = meshDimY / nRow;

    vertexNum = (nCol+1)*(nRow+1);
    vertexIdxNum = 6 * nCol * nRow;

    pVertexData = new float[vertexNum*elmSize*sizeof(float)];
    pVertexIdxData = new unsigned int[ vertexIdxNum ];
    pVertexNormal = new float[vertexNum*3*sizeof(float)];

    for( int h = 0; h <= nRow ; ++h )
        for( int w = 0; w <= nCol; ++w )
        {
            pVertexData[ ( h*(nCol+1) + w )*elmSize     ] = xoffset*w - 0.5*meshDimX;
            pVertexData[ ( h*(nCol+1) + w )*elmSize + 1 ] = yoffset*(nRow-h);
            pVertexData[ ( h*(nCol+1) + w )*elmSize + 2 ] = 0.0f;
            pVertexData[ ( h*(nCol+1) + w )*elmSize + 3 ] = 1.0f;

            pVertexNormal[ ( h*(nCol+1) + w )*3  ] = .0f;
            pVertexNormal[ ( h*(nCol+1) + w )*3+1] = .0f;
            pVertexNormal[ ( h*(nCol+1) + w )*3+2] = 1.0f;
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

int initSphereVertexData()
{
    GLMmodel* model = glmReadOBJ( "sphere.obj" );
    if( model == NULL )
        return -1;

    sphereVertexNum = model->numvertices;
    sphereVertexIdxNum = model->numtriangles * 3;
    pSphereVertex = new float[ sphereVertexNum * 4 + model->numnormals * 3];
    pSphereNormal = new float[ model->numnormals * 3];
    pSphereVertexIdx = new unsigned int[ sphereVertexIdxNum ];

    GLMgroup* group = model->groups;
    while(group)
    {
        GLMtriangle* tri;
        for( int i=0; i < group->numtriangles; ++i )
        {
            tri = &model->triangles[group->triangles[i]];
            pSphereVertexIdx[3*i] = tri->vindices[0]-1;
            pSphereVertexIdx[3*i+1] = tri->vindices[1]-1;
            pSphereVertexIdx[3*i+2] = tri->vindices[2]-1;
            memcpy( &pSphereNormal[3*pSphereVertexIdx[3*i] ], &model->normals[ 3 * (tri->vindices[0]) ], 3*sizeof(float) );
            memcpy( &pSphereNormal[3*pSphereVertexIdx[3*i+1] ], &model->normals[ 3 * (tri->nindices[1]) ], 3*sizeof(float) );
            memcpy( &pSphereNormal[3*pSphereVertexIdx[3*i+2] ], &model->normals[ 3 * (tri->nindices[2]) ], 3*sizeof(float) );
        }
        group = group->next;
    }
    for( int i = 0; i < sphereVertexNum; ++i )
    {
        pSphereVertex[4*i] = model->vertices[3*(i+1)];
        pSphereVertex[4*i+1] = model->vertices[3*(i+1)+1];
        pSphereVertex[4*i+2] = model->vertices[3*(i+1)+2];
        pSphereVertex[4*i+3] = 1.0f;
    }

    //memcpy( pSphereVertex + 4*sphereVertexNum, pSphereNormal, 3*sphereVertexNum );

    if( vbo_sphere )
        glDeleteBuffers( 1, &vbo_sphere );
    
    glGenBuffers( 1, &vbo_sphere );
    glBindBuffer( GL_ARRAY_BUFFER, vbo_sphere );
    glBufferData( GL_ARRAY_BUFFER, sizeof(float)*sphereVertexNum*4, pSphereVertex, GL_STATIC_DRAW );

    if(vbo_sphereN)
        glDeleteBuffers( 1, &vbo_sphereN );
    glGenBuffers( 1, &vbo_sphereN );
    glBindBuffer( GL_ARRAY_BUFFER, vbo_sphereN );
    glBufferData( GL_ARRAY_BUFFER, sizeof(float)*sphereVertexNum*3, pSphereNormal, GL_STATIC_DRAW );

    if( ibo_sphere )
        glDeleteBuffers( 1, &ibo_sphere );

    glGenBuffers( 1, &ibo_sphere );
    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, ibo_sphere );
    glBufferData( GL_ELEMENT_ARRAY_BUFFER, sphereVertexIdxNum * sizeof( unsigned int ), pSphereVertexIdx, GL_STATIC_DRAW );

    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
    glBindBuffer( GL_ARRAY_BUFFER,0 );

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
    ksLoc = glGetUniformLocation( prog, "ks" );
    kdLoc = glGetUniformLocation( prog, "kd" );
    shLoc = glGetUniformLocation( prog, "shininess" );
    lightPosLoc = glGetUniformLocation( prog, "lightPos" );

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

    if( initSphereVertexData() )
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

    if( pSphereVertex )
        delete [] pSphereVertex;
    pSphereVertex = 0;

    if( pSphereVertexIdx )
        delete [] pSphereVertexIdx;
    pSphereVertexIdx = 0;

    cleanupCUDA();
}
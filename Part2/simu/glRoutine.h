#ifndef _GLROUTINE_H
#define _GLROUTINE_H

#include <GL/glew.h>
#include <GL/glut.h>


void glut_display();

void glut_idle();

void glut_reshape( int w, int h );

void glut_keyboard( unsigned char key, int x, int y);
void glut_mouse( int button, int state, int x, int y );
void glut_mousemotion( int x, int y );
int initPBO();
int createGridMesh( unsigned int nCol, unsigned int nRow );
int intVertexData();

GLuint initShader();
GLuint initShaderProg();

int initGL();

void cleanUpGL();

#endif
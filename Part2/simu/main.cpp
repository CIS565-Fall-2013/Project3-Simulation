#include <iostream>
#include <gl/glew.h>
#include <GL/freeglut.h>
#include "glRoutine.h"

using namespace std;

int win_width = 800;
int win_height = 600;
int win_id;

int main( int argc, char* argv[] )
{
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_STENCIL );

    glutInitContextVersion( 4,0 );
    glutInitContextFlags( GLUT_FORWARD_COMPATIBLE );
    glutInitContextProfile( GLUT_COMPATIBILITY_PROFILE );

    glutInitWindowSize( win_width, win_height );
    win_id = glutCreateWindow( "Cloth Simulation" );

    GLenum errCode = glewInit();
    if( errCode != GLEW_OK )
    {
        cerr<<"Error: "<<glewGetErrorString(errCode)<<endl;
        return 1;
    }

    if( initGL() != 0 )
    {
        cout<<"OpenGL initialization faild!\n";
        return 1;
    }

  
    glutDisplayFunc( glut_display );
    glutReshapeFunc( glut_reshape);
    glutKeyboardFunc( glut_keyboard );
    glutMouseFunc( glut_mouse );
    glutMotionFunc( glut_mousemotion );
    glutIdleFunc( glut_idle );
    glutMainLoop();

    cleanUpGL();
    return 0;
}
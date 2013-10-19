uniform mat4 u_projMatrix;
uniform sampler2D u_height;

attribute vec4 Position;
attribute vec2 Texcoords;

varying vec2 v_Texcoords;
varying float f_height;

void main(void)
{
    v_Texcoords = Texcoords;
    vec4 pos = Position;
    f_height = texture2D(u_height, Texcoords).w;
    pos.z = -0.01-clamp(f_height,0.0,2.0); // pos.z moves the plane up and down & the clamp parameters decides the max/min depth of the hole
    pos = u_projMatrix * pos;
    gl_Position = pos;
}

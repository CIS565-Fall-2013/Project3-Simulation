#version 330

in vec4 Position;
in vec2 Texcoords;
in vec4 Velocity;

out vec3 v_Velocity;

void main(void)
{
	gl_Position = Position;
	v_Velocity=normalize(Velocity.xyz)*1.0f;
	//gl_Position=vec4(v_Velocity,1.0f);
}

#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;
uniform vec3 u_lightPos;

in vec4 vs_position;
in vec3 vs_up;
in vec3 vs_forward;
in vec3 vs_color;
in vec4 vs_shape; //Length, Wingspan, Delta, Deflection Angle

out VertexData{
	vec3 EyeNormal;
	vec3 EyeForward;
	vec3 Color;
	float Length;
	float HalfWingSpan;
	float DeltaSweep;
	float HalfBodyHeight;
	float WingDeflection;
}vertexData;

//Transform each vertex to eye space
void main(void)
{
	gl_Position = u_viewMatrix*vs_position;

	//Warning, Will only work for view matricies with uniform scaling
	//Use Normal matrix otherwise (inverse(transpose(u_viewMatrix)
	vertexData.EyeNormal = vec3(u_viewMatrix*vec4(vs_up,0.0));
	vertexData.EyeForward = vec3(u_viewMatrix*vec4(vs_forward,0.0));


	vertexData.Color = vs_color;
	vertexData.Length         = 0.03;//vs_shape.x;
	vertexData.HalfWingSpan   = 0.05;//vs_shape.y/2;
	vertexData.DeltaSweep     = 0.025;//vs_shape.z;
	vertexData.HalfBodyHeight = vertexData.HalfWingSpan*0.25;	  
	vertexData.WingDeflection = radians(0.0);//vs_shape.w;

}

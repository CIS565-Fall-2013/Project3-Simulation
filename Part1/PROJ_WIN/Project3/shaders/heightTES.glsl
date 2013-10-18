#version 400

uniform mat4 u_projMatrix;
uniform sampler2D u_height;

layout (triangles, equal_spacing, ccw) in;

in vec4 tc_Position[];
in vec2 tc_Texcoords[];

out vec2 te_Texcoords;
out float te_height;
out vec3 te_PatchDistance;


vec2 interpolate2D(vec2 v0, vec2 v1, vec2 v2)
{
   	 return vec2(gl_TessCoord.x) * v0 + vec2(gl_TessCoord.y) * v1 + vec2(gl_TessCoord.z) * v2;
}
vec4 interpolate4D(vec4 v0, vec4 v1, vec4 v2)
{
   	 return vec4(gl_TessCoord.x) * v0 + vec4(gl_TessCoord.y) * v1 + vec4(gl_TessCoord.z) * v2;
}

void main()
{
	te_PatchDistance = gl_TessCoord;

	te_Texcoords = interpolate2D(tc_Texcoords[0], tc_Texcoords[1], tc_Texcoords[2]);
	vec4 Position = interpolate4D(tc_Position[0], tc_Position[1], tc_Position[2]);

	te_height = texture2D(u_height, te_Texcoords).w;
    Position.z = -0.01-clamp(te_height,0.0,2.0);

	gl_Position = u_projMatrix * Position;
}
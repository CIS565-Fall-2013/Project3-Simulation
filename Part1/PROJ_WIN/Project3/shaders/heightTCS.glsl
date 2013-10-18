#version 400

layout(vertices = 3) out;

uniform vec3 u_cameraPos;

in vec4 v_Position[];
in vec2 v_Texcoords[];

out vec4 tc_Position[];
out vec2 tc_Texcoords[];
 
uniform float tessLevelInner = 6.0;
uniform float tessLevelOuter = 6.0;
 
float GetTessLevel(float Distance0, float Distance1)
{
    float AvgDistance = (Distance0 + Distance1) / 2.0;

    if (AvgDistance <= 2.4) {
        return 6.0;
    }
    else if (AvgDistance <= 3.25) {
        return 4.0;
    }
    else {
        return 2.0;
    }
}	

void main () {
	tc_Position[gl_InvocationID] = v_Position[gl_InvocationID];
	tc_Texcoords[gl_InvocationID] = v_Texcoords[gl_InvocationID];

// Calculate the distance from the camera to the three control points
    float d0 = distance(u_cameraPos, v_Position[0].xyz);
    float d1 = distance(u_cameraPos, v_Position[1].xyz);
    float d2 = distance(u_cameraPos, v_Position[2].xyz);

	  // Calculate the tessellation levels
    gl_TessLevelOuter[0] = GetTessLevel(d1, d2);
    gl_TessLevelOuter[1] = GetTessLevel(d2, d0);
    gl_TessLevelOuter[2] = GetTessLevel(d0, d1);
    gl_TessLevelInner[0] = gl_TessLevelOuter[2];


	/*
	// Calculate the tessellation levels
	gl_TessLevelInner[0] = tessLevelInner; // number of nested primitives to generate
	gl_TessLevelOuter[0] = tessLevelOuter; // times to subdivide first side
	gl_TessLevelOuter[1] = tessLevelOuter; // times to subdivide second side
	gl_TessLevelOuter[2] = tessLevelOuter; // times to subdivide third side
	*/
}
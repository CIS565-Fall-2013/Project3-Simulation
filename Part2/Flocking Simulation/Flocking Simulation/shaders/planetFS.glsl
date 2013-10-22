#version 330

in vec3 worldCoord;
in vec3 worldNormal;
in vec3 fVelocity;

out vec4 FragColor;

void main()
{	
    vec3 L = normalize(vec3(2,2,2)-worldCoord);
    float light = 0.1 + 0.9*clamp(dot(worldNormal,L),0.0, 1.0);
    vec3 color = fVelocity;
	FragColor = vec4(light*color,1.0);
//	FragColor = vec4(light*color, 1.0);
} 

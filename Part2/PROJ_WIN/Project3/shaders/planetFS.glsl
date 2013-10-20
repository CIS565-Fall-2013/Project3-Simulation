#version 330

in vec3 WorldCoord;
in vec3 WorldVec;
in vec3 ToCam;
in vec3 Up;
in vec3 Right;
in vec3 TexCoord;
in vec4 tag;

out vec4 FragColor;

void main()
{
	vec3 N = Right * TexCoord.x + Up * TexCoord.y + WorldVec * TexCoord.z;
    float dist = length(WorldCoord);
    vec3 L = normalize(-WorldCoord);
    float light = (0.3 + 0.9*clamp(dot(N,L),0.0, 1.0)*exp(-dist));

    // Set the four points with different color
	vec3 color = vec3(0.0);
    if( tag.x < 0.0001 ) color = vec3(0.8, 0.2, 0.1);
    if( tag.y < 0.0001 ) color = vec3(0.2, 0.8, 0.1);
    if( tag.z < 0.0001 ) color = vec3(0.8, 0.9, 0.1);
    if( tag.w < 0.0001 ) color = vec3(0.6, 0.2, 0.6);
	FragColor = vec4(color*light,1.0);
} 
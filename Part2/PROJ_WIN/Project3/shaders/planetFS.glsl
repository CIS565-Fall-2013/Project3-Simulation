#version 330

in vec3 WorldCoord;
in vec3 ToCam;
in vec3 Up;
in vec3 Right;
in vec2 TexCoord;
out vec4 FragColor;

void main()
{
    float dist = length(WorldCoord);
    if(dist <= 0.01)
    {
        FragColor = vec4(1.0);
        return;
    }
    vec3 color = vec3(0.5, 0.1, 0.1);
    FragColor = vec4(color,1.0);
} 
#version 330


in vec3 fs_position;
in vec3 fs_normal;
in vec3 fs_color;

in vec3 fs_camera_pos;
in vec3 fs_light_position;
in vec3 fs_light_color;

out vec4 out_Color;

void main() {


	vec4 diffuseColor = vec4(fs_color, 1.0);

    vec3 normal = normalize(fs_normal);
    vec3 light = normalize(fs_light_position);

    float diffuseTerm = clamp(dot(normalize(fs_normal), normalize(fs_light_position)), 0.0, 1.0);

    vec3 ReflectionVector = 2*diffuseTerm*normal - light;
	///dot(light,normal)
	vec3 ViewDirection =  fs_camera_pos - fs_position;
    
    vec4 ambientLight = vec4(0.3, 0.3, 0.3, 1.0);

    vec4 specularLight = vec4(fs_light_color,1.0);
   
   float tmp;
	if(dot(normalize(ReflectionVector),normalize(ViewDirection))<0.0)
	 	tmp = 0.0;
	else
		tmp = dot(normalize(ReflectionVector),normalize(ViewDirection));

    
    out_Color = 0.1*2*specularLight * pow(tmp, 5) + 0.6*2*diffuseColor * diffuseTerm*specularLight  + 0.8*ambientLight*diffuseColor; 

	//out_Color = fs_color;
}
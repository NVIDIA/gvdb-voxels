#version 440


layout(location = 0)   in vec3 vertex;
layout(location = 1)   in vec3 normal;
layout(location = 2)   in vec3 texcoord;

out vec3 vs_WorldNormal;

uniform mat4 uW;
uniform vec3 uTexRes;

void main()
{	
	vs_WorldNormal = normalize( normal ); 

	gl_Position = uW * vec4( vertex, 1.0 ) - vec4(1,1,1,0);
}

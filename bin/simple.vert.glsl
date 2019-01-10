
#version 440 core
#extension GL_NV_gpu_shader5 : require

layout(location = 0)   in vec3 vertex;   // Sends vertex data from models here
layout(location = 1)   in vec3 normal;   // Sends normal data (if any) from models here

uniform mat4	 uModel;
uniform mat4	 uView; 
uniform mat4	 uProj; 

out vec3		WorldNormal; 
out vec3		WorldPos;

void main( void )
{  
	// World-space position & normal
	WorldPos = (uModel * vec4( vertex, 1.0f )).xyz;
	WorldNormal = (uModel * vec4( normal, 1.0f )).xyz;

	// Eye-space position & normal

    // Transform vertex into clip space
	gl_Position = uProj * uView * vec4( WorldPos, 1.0f );	
}
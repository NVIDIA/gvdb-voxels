#version 440

#extension GL_ARB_shader_image_size : enable

layout(triangles, invocations = 1) in;
layout(triangle_strip, max_vertices = 3) out;

uniform vec3 uTexRes;

in vec3 vs_WorldNormal[]; //need normal to pass to fp for lighting

out vec3 gs_WorldNormal; //pass normal to fp for lighting
out vec3 gs_Color; //color depending on dominant axes
out mat3 gs_SwizzleMatrixInv; //inverse of swizzle matrix needed in fs to unswizzle back

void main()
{
	//calculate normal    
	vec3 eyeSpaceNormal = abs(cross(normalize(gl_in[1].gl_Position.xyz - gl_in[0].gl_Position.xyz),
									normalize(gl_in[2].gl_Position.xyz - gl_in[0].gl_Position.xyz)));
	float dominantAxis = max(eyeSpaceNormal.x, max(eyeSpaceNormal.y, eyeSpaceNormal.z));	
	mat3 swizzleMatrix;

	if (dominantAxis == eyeSpaceNormal.x) {		
		swizzleMatrix = mat3(vec3(0.0, 0.0, 1.0),
							 vec3(0.0, 1.0, 0.0),
							 vec3(1.0, 0.0, 0.0));		
    } else if (dominantAxis == eyeSpaceNormal.y) {		
		swizzleMatrix = mat3(vec3(1.0, 0.0, 0.0),
						 	 vec3(0.0, 0.0, 1.0),
							 vec3(0.0, 1.0, 0.0));		
    } else if (dominantAxis == eyeSpaceNormal.z) {		
		swizzleMatrix = mat3(vec3(1.0, 0.0, 0.0),
							 vec3(0.0, 1.0, 0.0),
							 vec3(0.0, 0.0, 1.0));	
    }
    // Pass inverse of swizzle matrix to fragment shader.
    gs_SwizzleMatrixInv = inverse(swizzleMatrix);	

	// Send world position and normal, needed for lighting in the fragment shader
	gs_WorldNormal = vs_WorldNormal[0];
	gl_Position = vec4(swizzleMatrix * gl_in[0].gl_Position.xyz , 1.0);
	EmitVertex();
	gs_WorldNormal = vs_WorldNormal[1];
	gl_Position = vec4(swizzleMatrix * gl_in[1].gl_Position.xyz , 1.0);
	EmitVertex();
	gs_WorldNormal = vs_WorldNormal[2];
	gl_Position = vec4(swizzleMatrix * gl_in[2].gl_Position.xyz , 1.0);
	EmitVertex();

	EndPrimitive();
}

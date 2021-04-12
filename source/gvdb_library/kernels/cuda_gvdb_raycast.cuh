//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2017 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
//-----------------------------------------------------------------------------
// File: cuda_gvdb_raycast.cuh
//
// CUDA Raycasting 
// - Trilinear & tricubic interpolation
// - Gradients
// - Small-step Trilinear/cubic Raycast
// - GVDB rayDeep     - deep volume sampling
// - GVDB raySurface  - surface hit raycast
// - GVDB rayLevelSet - level set raycast
// - GVDB rayShadow   - shadow raycast
//-----------------------------------------------

// gvdbBrickFunc ( gvdb, channel, nodeid, t, pos, dir, hit, norm, clr )
typedef void(*gvdbBrickFunc_t)( VDBInfo*, uchar, int, float3, float3, float3, float3&, float3&, float4& );

static const int MAXLEV = 5;
static const int MAX_ITER = 256;

// Gets the value of a given floating-point channel at a point inside a brick.
// `gvdb` is the volume's `VDBInfo` object.
// `chan` is the channel to sample.
// `p` is the location of the point within the brick.
// `offs` is the coordinate of the minimum corner of the brick in the atlas.
// TODO: Turn `offs` into an integer vector.
inline __device__ float getTricubic ( VDBInfo* gvdb, uchar chan, float3 p, float3 offs )
{ 
	static const float MID = 1.0;
	static const float HI = 2.0;

	// find bottom-left corner of local 3x3x3 group		
	float3  q = floor3(p + offs) - MID;				// move to bottom-left corner	
	
	// evaluate tri-cubic
	float3 tb = frac3(p) * 0.5 + 0.25;
	float3 ta = (1.0-tb);
	float3 ta2 = ta*ta;
	float3 tb2 = tb*tb;
	float3 tab = ta*tb*2.0;

	// lookup 3x3x3 local neighborhood
	float tv[9];
	tv[0] = tex3D<float>( gvdb->volIn[chan], q.x,		q.y,		q.z );
	tv[1] = tex3D<float>( gvdb->volIn[chan], q.x+MID,	q.y,		q.z );
	tv[2] = tex3D<float>( gvdb->volIn[chan], q.x+HI,	q.y,		q.z );
	tv[3] = tex3D<float>( gvdb->volIn[chan] , q.x,		q.y+MID,	q.z );
	tv[4] = tex3D<float>( gvdb->volIn[chan], q.x+MID,	q.y+MID,	q.z );
	tv[5] = tex3D<float>( gvdb->volIn[chan], q.x+HI,	q.y+MID,	q.z );
	tv[6] = tex3D<float>( gvdb->volIn[chan], q.x,		q.y+HI,		q.z );
	tv[7] = tex3D<float>( gvdb->volIn[chan], q.x+MID,	q.y+HI,		q.z );
	tv[8] = tex3D<float>( gvdb->volIn[chan], q.x+HI,	q.y+HI,		q.z );

	float3 abc = make_float3 (	tv[0]*ta2.x + tv[1]*tab.x + tv[2]*tb2.x, 
								tv[3]*ta2.x + tv[4]*tab.x + tv[5]*tb2.x,
								tv[6]*ta2.x + tv[7]*tab.x + tv[8]*tb2.x );

	tv[0] = tex3D<float>( gvdb->volIn[chan], q.x,		q.y,		q.z+MID );
	tv[1] = tex3D<float>( gvdb->volIn[chan], q.x+MID,	q.y,		q.z+MID );
	tv[2] = tex3D<float>( gvdb->volIn[chan], q.x+HI,	q.y,		q.z+MID );
	tv[3] = tex3D<float>( gvdb->volIn[chan], q.x,		q.y+MID,	q.z+MID );
	tv[4] = tex3D<float>( gvdb->volIn[chan], q.x+MID,	q.y+MID,	q.z+MID );
	tv[5] = tex3D<float>( gvdb->volIn[chan], q.x+HI,	q.y+MID,	q.z+MID );
	tv[6] = tex3D<float>( gvdb->volIn[chan], q.x,		q.y+HI,		q.z+MID );
	tv[7] = tex3D<float>( gvdb->volIn[chan], q.x+MID,	q.y+HI,		q.z+MID );
	tv[8] = tex3D<float>( gvdb->volIn[chan], q.x+HI,	q.y+HI,		q.z+MID );

	float3 def = make_float3 (	tv[0]*ta2.x + tv[1]*tab.x + tv[2]*tb2.x, 
								tv[3]*ta2.x + tv[4]*tab.x + tv[5]*tb2.x,
								tv[6]*ta2.x + tv[7]*tab.x + tv[8]*tb2.x );

	tv[0] = tex3D<float>( gvdb->volIn[chan], q.x,		q.y,		q.z+HI );
	tv[1] = tex3D<float>( gvdb->volIn[chan], q.x+MID,	q.y,		q.z+HI );
	tv[2] = tex3D<float>( gvdb->volIn[chan], q.x+HI,	q.y,		q.z+HI );
	tv[3] = tex3D<float>( gvdb->volIn[chan], q.x,		q.y+MID,	q.z+HI );
	tv[4] = tex3D<float>( gvdb->volIn[chan], q.x+MID,	q.y+MID,	q.z+HI );
	tv[5] = tex3D<float>( gvdb->volIn[chan], q.x+HI,	q.y+MID,	q.z+HI );
	tv[6] = tex3D<float>( gvdb->volIn[chan], q.x,		q.y+HI,		q.z+HI );
	tv[7] = tex3D<float>( gvdb->volIn[chan], q.x+MID,	q.y+HI,		q.z+HI );
	tv[8] = tex3D<float>( gvdb->volIn[chan], q.x+HI,	q.y+HI,		q.z+HI );

	float3 ghi = make_float3 (	tv[0]*ta2.x + tv[1]*tab.x + tv[2]*tb2.x, 
								tv[3]*ta2.x + tv[4]*tab.x + tv[5]*tb2.x,
								tv[6]*ta2.x + tv[7]*tab.x + tv[8]*tb2.x );
	
	float3 jkl = make_float3 (	abc.x*ta2.y + abc.y*tab.y + abc.z*tb2.y, 
								def.x*ta2.y + def.y*tab.y + def.z*tb2.y,
								ghi.x*ta2.y + ghi.y*tab.y + ghi.z*tb2.y );

	return jkl.x*ta2.z + jkl.y*tab.z + jkl.z*tb2.z;
}

// Gets the value of a given floating-point channel at a point inside a brick.
// `gvdb` is the volume's `VDBInfo` object.
// `chan` is the channel to sample.
// `wp` is the point to sample in index-space (not atlas-space!)
// `offs` is the minimum vertex of the brick's bounding box in atlas space.
// `vmin` is the minimum vertex of the brick's bounding box in index-space.
inline __device__ float getTrilinear (VDBInfo* gvdb, uchar chan, float3 wp, float3 offs, float3 vmin)
{
	float3 p = offs + (wp-vmin);		// sample point in index coords		
	return tex3D<float> ( gvdb->volIn[chan], p.x, p.y, p.z );
}

#ifdef CUDA_PATHWAY
	inline __device__ unsigned char getVolSampleC ( VDBInfo* gvdb, uchar chan, float3 wpos )
	{
		float3 offs, vmin; uint64 nid;
		VDBNode* node = getNodeAtPoint ( gvdb, wpos, &offs, &vmin, &nid );				// find vdb node at point
		if ( node == 0x0 ) return 0;
		float3 p = 	offs + (wpos-vmin);
		return tex3D<uchar> ( gvdb->volIn[chan], p.x, p.y, p.z );
	}
	inline __device__ float getVolSampleF ( VDBInfo* gvdb, uchar chan, float3 wpos )
	{
		float3 offs, vmin; uint64 nid;
		VDBNode* node = getNodeAtPoint ( gvdb, wpos, &offs, &vmin, &nid );				// find vdb node at point
		if ( node == 0x0 ) return 0;
		float3 p = 	offs + (wpos-vmin);
		return tex3D<float> ( gvdb->volIn[chan], p.x, p.y, p.z );
	}
#endif

// Gets the negative of the gradient of the floating-point channel with index `chan` at the atlas-space position `p`
// using default filtering.
// This will point away from higher-density regions in a density field, and into a level set/signed distance field.
inline __device__ float3 getGradient ( VDBInfo* gvdb, uchar chan, float3 p )
{
	float3 g;
	// note: must use +/- 0.5 since apron may only be 1 voxel wide (cannot go beyond brick)
	g.x = 1.0* (tex3D<float>( gvdb->volIn[chan], p.x-.5, p.y,   p.z  ) - tex3D<float>( gvdb->volIn[chan], p.x+.5, p.y, p.z ));
	g.y = 1.0* (tex3D<float>( gvdb->volIn[chan], p.x,   p.y-.5, p.z  ) - tex3D<float>( gvdb->volIn[chan], p.x, p.y+.5, p.z ));
	g.z = 1.0* (tex3D<float>( gvdb->volIn[chan], p.x,   p.y,   p.z-.5) - tex3D<float>( gvdb->volIn[chan], p.x, p.y, p.z+.5 ));
	g = normalize ( g );
	return g;
}

// Gets the gradient of the floating-point channel with index `chan` at the atlas-space position `p` using
// default filtering.
// This will approximate the normal of a level set/signed distance field, and point away from higher-density regions
// in a density field.
inline __device__ float3 getGradientLevelSet ( VDBInfo* gvdb, uchar chan, float3 p )
{
	// tri-linear filtered gradient 
	// (assumes atlas has linear hardware filtering on)	
	float3 g;
	g.x = 1.0* (tex3D<float>( gvdb->volIn[chan], p.x+.5, p.y,   p.z  ) - tex3D<float>( gvdb->volIn[chan], p.x-.5, p.y, p.z ));
	g.y = 1.0* (tex3D<float>( gvdb->volIn[chan], p.x,   p.y+.5, p.z  ) - tex3D<float>( gvdb->volIn[chan], p.x, p.y-.5, p.z ));
	g.z = 1.0* (tex3D<float>( gvdb->volIn[chan], p.x,   p.y,   p.z+.5) - tex3D<float>( gvdb->volIn[chan], p.x, p.y, p.z-.5 ));
	g = normalize ( g );
	return g;
}

// Gets the negative of the gradient of the floating-point channel with index `chan` at the atlas-space position `p`
// using tricubic interpolation.
inline __device__ float3 getGradientTricubic ( VDBInfo* gvdb, uchar chan, float3 p, float3 offs )
{
	// tri-cubic filtered gradient
	const float vs = 0.5;
	float3 g;
	g.x = (getTricubic (gvdb, chan, p+make_float3(-vs,0,0), offs)	- getTricubic (gvdb, chan, p+make_float3(vs,0,0), offs))/(2*vs);
	g.y = (getTricubic (gvdb, chan, p+make_float3(0,-vs,0), offs)	- getTricubic (gvdb, chan, p+make_float3(0,vs,0), offs))/(2*vs);
	g.z = (getTricubic (gvdb, chan, p+make_float3(0,0,-vs), offs)	- getTricubic (gvdb, chan, p+make_float3(0,0,vs), offs))/(2*vs);
	g = normalize ( g );
	return g;
}

// Marches along the ray p + o + rdir*t in atlas space in steps of SCN_FINESTEP, using default interpolation.
// If it samples a point less than SCN_THRESH, halts and returns:
//	 `p`: The atlas-space coordinate of the intersection relative to `o`
//   returned value: The index-space coordinate of the intersection
// Otherwise, returns (NOHIT, NOHIT, NOHIT).
// Inputs:
//   `gvdb`: The volume's `VDBInfo` object
//   `chan`: The channel to sample
//   `p`: The origin of the ray in atlas-space relative to `o`
//   `o`: The minimum AABB vertex of the brick to start sampling from
//   `rpos`: Unused
//   `rdir`: The direction of the ray in atlas-space
//   `vmin`: The minimum AABB vertex of the brick in index-space
__device__ float3 rayLevelSet ( VDBInfo* gvdb, uchar chan, float3& p, float3 o, float3 rpos, float3 rdir, float3 vmin )
{
	float dt = SCN_FINESTEP;
	float3 pt = dt*rdir;
	
	for ( int i=0; i < 512; i++ ) {
		if ( tex3D<float>( gvdb->volIn[chan], p.x+o.x, p.y+o.y, p.z+o.z ) < SCN_THRESH )	// trilinear test
			return p + vmin;		
		p += pt;				
	}
	return make_float3(NOHIT, NOHIT, NOHIT);
}

// Samples channel `chan` at atlas-space position `p`, returning a `uchar4` color.
inline __device__ uchar4 getColor ( VDBInfo* gvdb, uchar chan, float3 p )
{
	return tex3D<uchar4> ( gvdb->volIn[chan], (int) p.x, (int) p.y, (int) p.z );
}

// Samples channel `chan` at atlas-space position `p`, obtaining a `uchar4` color and casting it to a `float4`.
inline __device__ float4 getColorF ( VDBInfo* gvdb, uchar chan, float3 p )
{
	return make_float4 (tex3D<uchar4> ( gvdb->volIn[chan], (int) p.x, (int) p.y, (int) p.z ) );
}

//----------- RAY CASTING

// Traces a brick, rendering voxels as cubes.
// To find a surface intersection, this steps through each voxel once using DDA and stops when it finds a voxel with
// value greater than or equal to SCN_THRESH.
// Inputs:
//   `gvdb`: The volume's `VDBInfo` object
//   `chan`: The channel to render
//   `nodeid`: The index of the node at level 0
//   `t`: The current parameter of the ray
//   `pos`: The origin of the ray
//   `dir`: The direction of the ray
// Outputs:
//   `hit`: If hit.z == NOHIT, no intersection; otherwise, the coordinates of the intersection
//   `norm`: The normal at the intersection
//   `hclr`: The color of the color channel at the intersection point.
__device__ void raySurfaceVoxelBrick ( VDBInfo* gvdb, uchar chan, int nodeid, float3 t, float3 pos, float3 dir, float3& hit, float3& norm, float4& hclr )
{
	float3 vmin;
	VDBNode* node	= getNode ( gvdb, 0, nodeid, &vmin );				// Get the VDB leaf node
	float3  o = make_float3( node->mValue ) ;	// Atlas sub-volume to trace	

	HDDAState dda;
	dda.SetFromRay(pos, dir, t);
	dda.PrepareLeaf(vmin);
	
	for (int iter=0; iter < MAX_ITER
		&& dda.p.x >=0 && dda.p.y >=0 && dda.p.z >=0
		&& dda.p.x < gvdb->res[0] && dda.p.y < gvdb->res[0] && dda.p.z < gvdb->res[0]; iter++)
	{
		if ( tex3D<float> ( gvdb->volIn[chan], dda.p.x+o.x+.5, dda.p.y+o.y+.5, dda.p.z+o.z+.5 ) > SCN_THRESH) {		// test texture atlas
			vmin += make_float3(dda.p);		// voxel location in world
			dda.t = rayBoxIntersect ( pos, dir, vmin, vmin + 1 );
			if (dda.t.z == NOHIT) {
				hit.z = NOHIT;
				continue;
			}
			hit = getRayPoint ( pos, dir, dda.t.x );

			// Compute the normal of the voxel [vmin, vmin+gvdb->voxelsize] at the hit point
			// Note: This is not normalized when the ray hits an edge of the voxel exactly
			float3 fromVoxelCenter = (hit - vmin) - 0.5f; // in [-1/2, 1/2]
			fromVoxelCenter -= 0.01 * dir; // Bias the sample point slightly towards the camera
			const float maxCoordinate = fmaxf(fmaxf(fabsf(fromVoxelCenter.x), fabsf(fromVoxelCenter.y)), fabsf(fromVoxelCenter.z));
			norm.x = (fabsf(fromVoxelCenter.x) == maxCoordinate ? copysignf(1.0f, fromVoxelCenter.x) : 0.0f);
			norm.y = (fabsf(fromVoxelCenter.y) == maxCoordinate ? copysignf(1.0f, fromVoxelCenter.y) : 0.0f);
			norm.z = (fabsf(fromVoxelCenter.z) == maxCoordinate ? copysignf(1.0f, fromVoxelCenter.z) : 0.0f);

			if ( gvdb->clr_chan != CHAN_UNDEF ) hclr = getColorF ( gvdb, gvdb->clr_chan, make_float3(dda.p)+o );
			return;	
		}
		dda.Next();
		dda.Step();
	}
}

// Traces a brick, rendering the surface with trilinear interpolation.
// To find an intersection, this samples using increments of `SCN_PSTEP` in `t` and stops when it finds a point greater
// than or equal to SCN_THRESH.
// Inputs:
//   `gvdb`: The volume's `VDBInfo` object
//   `chan`: The channel to render
//   `nodeid`: The index of the node at level 0
//   `t`: The current parameter of the ray
//   `pos`: The origin of the ray
//   `dir`: The direction of the ray
// Outputs:
//   `hit`: If hit.z == NOHIT, no intersection; otherwise, the coordinates of the intersection
//   `norm`: The normal at the intersection
//   `hclr`: The color of the color channel at the intersection point.
__device__ void raySurfaceTrilinearBrick ( VDBInfo* gvdb, uchar chan, int nodeid, float3 t, float3 pos, float3 dir, float3& hit, float3& norm, float4& hclr )
{
	float3 vmin;
	VDBNode* node	= getNode ( gvdb, 0, nodeid, &vmin );	// Get the VDB leaf node	
	float3  o = make_float3( node->mValue ) ;				// Atlas sub-volume to trace
	t.x = SCN_DIRECTSTEP * ceilf(t.x / SCN_DIRECTSTEP);		// Start on sampling wavefront (avoids subvoxel banding artifacts)
	float3	p = pos + t.x*dir - vmin;						// sample point in index coords			

	for (int iter=0; iter < MAX_ITER && p.x >=0 && p.y >=0 && p.z >=0 && p.x < gvdb->res[0] && p.y < gvdb->res[0] && p.z < gvdb->res[0]; iter++) 
	{	
		if (tex3D<float>(gvdb->volIn[chan], p.x+o.x, p.y+o.y, p.z+o.z ) >= SCN_THRESH ) {
			hit = p + vmin;
			norm = getGradient ( gvdb, chan, p+o );
			if ( gvdb->clr_chan != CHAN_UNDEF ) hclr = getColorF ( gvdb, gvdb->clr_chan, p+o );
			return;	
		}	
		p += SCN_DIRECTSTEP*dir;		
		t.x += SCN_DIRECTSTEP;
	}
}

// Traces a brick, rendering the surface with tricubic interpolation.
// To find an intersection, this samples using increments of `SCN_PSTEP` in `t` and stops when it finds a point greater
// than or equal to SCN_THRESH.
// Inputs:
//   `gvdb`: The volume's `VDBInfo` object
//   `chan`: The channel to render
//   `nodeid`: The index of the node at level 0
//   `t`: The current parameter of the ray
//   `pos`: The origin of the ray
//   `dir`: The direction of the ray
// Outputs:
//   `hit`: If hit.z == NOHIT, no intersection; otherwise, the coordinates of the intersection
//   `norm`: The normal at the intersection
//   `hclr`: The color of the color channel at the intersection point.
__device__ void raySurfaceTricubicBrick ( VDBInfo* gvdb, uchar chan, int nodeid, float3 t, float3 pos, float3 dir, float3& hit, float3& norm, float4& hclr )
{
	float3 vmin;
	VDBNode* node	= getNode ( gvdb, 0, nodeid, &vmin );			// Get the VDB leaf node	
	float3  o = make_float3( node->mValue ) ;				// Atlas sub-volume to trace	
	float3	p = pos + t.x*dir - vmin;		// sample point in index coords		
	float3  v;

	for (int iter=0; iter < MAX_ITER && p.x >=0 && p.y >=0 && p.z >=0 && p.x < gvdb->res[0] && p.y < gvdb->res[0] && p.z < gvdb->res[0]; iter++) 
	{
		v.z = getTricubic ( gvdb, chan, p, o );
		if ( v.z >= SCN_THRESH) {
			v.x = getTricubic ( gvdb, chan, p - SCN_FINESTEP*dir, o );
			v.y = (v.z - SCN_THRESH)/(v.z-v.x);
			p += -v.y*SCN_FINESTEP*dir;
			hit = p + vmin;
			norm = getGradientTricubic ( gvdb, chan, p, o );
			if ( gvdb->clr_chan != CHAN_UNDEF ) hclr = getColorF ( gvdb, gvdb->clr_chan, p+o );
			return;			
		}	
		p += SCN_DIRECTSTEP*dir;		
		t.x += SCN_DIRECTSTEP;
	}
}

// Looks up the stored NDC value from a depth buffer (i.e. after perspective projection), and inverts this
// to get the world-space depth of the stored fragment.
inline __device__ float getLinearDepth(float* depthBufFloat)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;					// Pixel coordinates
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float z = depthBufFloat[(SCN_HEIGHT - 1 - y) * SCN_WIDTH + x];	// Get depth value
	float n = scn.camnear;
	float f = scn.camfar;
	return (-n * f / (f - n)) / (z - (f / (f - n)));				// Return linear depth
}

// Get the value of t at which the ray starting at the camera position with direction `dir` intersects the depth buffer
// at the current thread. INFINITY if there is no depth buffer.
inline __device__ float getRayDepthBufferMax(const float3& rayDir) {
	if (SCN_DBUF != 0x0) {
		// Solve
		//   t * (length of rayDir in app space) == getLinearDepth(SCN_DBUF)
		// for t, where (length of rayDir in app space) is length((SCN_XFORM * float4(rayDir, 0)).xyz):
		float3 rayInWorldSpace;
		rayInWorldSpace.x = rayDir.x * SCN_XFORM[0] + rayDir.y * SCN_XFORM[4] + rayDir.z * SCN_XFORM[ 8];
		rayInWorldSpace.y = rayDir.x * SCN_XFORM[1] + rayDir.y * SCN_XFORM[5] + rayDir.z * SCN_XFORM[ 9];
		rayInWorldSpace.z = rayDir.x * SCN_XFORM[2] + rayDir.y * SCN_XFORM[6] + rayDir.z * SCN_XFORM[10];
		return getLinearDepth(SCN_DBUF) / length(rayInWorldSpace);
	}
	else {
		return INFINITY;
	}
}

#define EPSTEST(a,b,c)	(a>b-c && a<b+c)
#define VOXEL_EPS 0.0001

// Traces a brick, rendering the level set surface with default interpolation.
// To find an intersection, this samples using increments of `SCN_PSTEP` in `t` and stops when it finds a point less
// than SCN_THRESH.
// Inputs:
//   `gvdb`: The volume's `VDBInfo` object
//   `chan`: The channel to render
//   `nodeid`: The index of the node at level 0
//   `t`: The current parameter of the ray
//   `pos`: The origin of the ray
//   `dir`: The direction of the ray
// Outputs:
//   `hit`: If hit.z == NOHIT, no intersection; otherwise, the coordinates of the intersection
//   `norm`: The normal at the intersection
//   `hclr`: The color of the color channel at the intersection point
__device__ void rayLevelSetBrick ( VDBInfo* gvdb, uchar chan, int nodeid, float3 t, float3 pos, float3 dir, float3& hit, float3& norm, float4& hclr )
{
	float3 vmin;
	VDBNode* node	= getNode ( gvdb, 0, nodeid, &vmin );			// Get the VDB leaf node	
	float3  o = make_float3( node->mValue ) ;				// Atlas sub-volume to trace	
	float3	p = pos + t.x*dir - vmin;					// sample point in index coords			
	t.x = SCN_DIRECTSTEP * ceilf( t.x / SCN_DIRECTSTEP );

	for (int iter=0; iter < MAX_ITER && p.x >=0 && p.y >=0 && p.z >=0 && p.x <= gvdb->res[0] && p.y <= gvdb->res[0] && p.z <= gvdb->res[0]; iter++) {	

		if (tex3D<float>(gvdb->volIn[chan], p.x+o.x, p.y+o.y, p.z+o.z ) < SCN_THRESH ) {			// test atlas for zero crossing
			hit = rayLevelSet ( gvdb, chan, p, o, pos, dir, vmin );
			if ( hit.z != NOHIT ) {
				norm = getGradientLevelSet ( gvdb, chan, p+o );
				if (gvdb->clr_chan != CHAN_UNDEF) hclr = getColorF(gvdb, gvdb->clr_chan, p + o);				
				return;
			}
		}	
		p += SCN_DIRECTSTEP*dir;
		t.x += SCN_DIRECTSTEP;
	}
}

// Empty intersector that simply reports a hit at the current position.
// Inputs:
//   `gvdb`: Unused
//   `chan`: Unused
//   `nodeid`: Unused
//   `t`: The current parameter of the ray
//   `pos`: The origin of the ray
//   `dir`: The direction of the ray
// Outputs:
//   `hit`: The coordinates of the intersection
//   `norm`: Unused
//   `hclr`: Unused
__device__ void rayEmptySkipBrick ( VDBInfo* gvdb, uchar chan, int nodeid, float3 t, float3 pos, float3 dir, float3& hit, float3& norm, float4& clr )
{
	hit = pos + t.x * dir; // Return brick hit
}

// Returns deep shadow accumulation along a ray. Each sample's value is mapped to a density value using the transfer
// function. This sample density is then treated as an opaque layer with opacity
//   exp( SCN_EXTINCT * density * SCN_SHADOWSTEP / (1.0 + t.x * 0.4) )
// where t.x in the above equation increments in steps of SCN_SHADOWSTEP, while the parameter of the ray increments in steps
// of SCN_DIRECTSTEP.
// Inputs:
//   `gvdb`: The volume's `VDBInfo` object
//   `chan`: The channel to render
//   `nodeid`: The index of the node at level 0
//   `t`: The current parameter of the ray
//   `pos`: The origin of the ray
//   `dir`: The direction of the ray
// Outputs:
//   `hit`: Unused
//   `norm`: Unused
//   `clr`: Accumulated color and transparency along the ray.
__device__ void rayShadowBrick ( VDBInfo* gvdb, uchar chan, int nodeid, float3 t, float3 pos, float3 dir, float3& hit, float3& norm, float4& clr )
{
	float3 vmin;
	VDBNode* node = getNode ( gvdb, 0, nodeid, &vmin );			// Get the VDB leaf node	
	t.x += gvdb->epsilon;												// make sure we start inside
	t.y -= gvdb->epsilon;												// make sure we end insidoke	
	float3 o = make_float3( node->mValue );					// atlas sub-volume to trace	
	float3 p = pos + t.x*dir - vmin;		// sample point in index coords
	float3 pt = SCN_DIRECTSTEP * dir;								// index increment	
	float val = 0;

	// accumulate remaining voxels	
	for (; clr.w < 1 && p.x >=0 && p.y >=0 && p.z >=0 && p.x < gvdb->res[0] && p.y < gvdb->res[0] && p.z < gvdb->res[0];) {		
		val = exp ( SCN_EXTINCT * transfer( gvdb, tex3D<float> ( gvdb->volIn[chan], p.x+o.x, p.y+o.y, p.z+o.z )).w * SCN_SHADOWSTEP/(1.0 + t.x * 0.4) );		// 0.4 = shadow gain
		clr.w = 1.0 - (1.0-clr.w) * val;
		p += pt;	
		t.x += SCN_SHADOWSTEP;
	}	
}

// DeepBrick - Sample into brick for deep volume raytracing
// Accumulates colors in a volume. Handles depth buffer intersections.
// This samples in increments of `SCN_PSTEP` in `t`.
// Each sample's value is mapped to a density value using the transfer function. This sample density is then treated as
// an opaque layer with opacity
//    exp(SCN_EXTINCT * val.w * SCN_DIRECTSTEP).
// Inputs:
//   `gvdb`: The volume's `VDBInfo` object
//   `chan`: The channel to render
//   `nodeid`: The index of the node at level 0
//   `t`: The current parameter of the ray
//   `pos`: The origin of the ray
//   `dir`: The direction of the ray
// Outputs:
//   `hit.x`: Front brick intersection point, equal to t.x
//   `hit.y`: Back intersection of brick (if hit.z = 0) or distance between ray origin and depth buffer intersection
//     (if hit.z = 1)
//   `hit.z`: 0 if ray passed through entire brick, 1 if intersected with depth buffer.
//   `norm`: Unused
//   `clr`: Accumulated color and transparency along the ray.
__device__ void rayDeepBrick ( VDBInfo* gvdb, uchar chan, int nodeid, float3 t, float3 pos, float3 dir, float3& hit, float3& norm, float4& clr )
{
	float3 vmin;
	VDBNode* node = getNode ( gvdb, 0, nodeid, &vmin );		// Get the VDB leaf node		
	
	t.x = SCN_DIRECTSTEP * ceilf( t.x / SCN_DIRECTSTEP );	// Start on sampling wavefront (avoids subvoxel banding artifacts)

	float3 o = make_float3( node->mValue );	// Atlas sub-volume to trace
	float3 wp = pos + t.x*dir;				// Sample position in index space
	float3 p = wp - vmin;					// Sample point in sub-volume (in units of voxels)
	const float3 wpt = SCN_DIRECTSTEP*dir;	// Increment in units of voxels
	const float dt = length(wpt);			// Change in t-value per step
	const float tDepthIntersection = getRayDepthBufferMax(dir); // The t.x at which the ray intersects the depth buffer

	// Record front hit point at first significant voxel
	if (hit.x == 0) hit.x = t.x; // length(wp - pos);

	// Accumulate remaining voxels
	for (int iter = 0; clr.w > SCN_ALPHACUT && iter < MAX_ITER && p.x >=0 && p.y >=0 && p.z >=0 && p.x < gvdb->res[0] && p.y < gvdb->res[0] && p.z < gvdb->res[0]; iter++) {			
		// Test to see if we've intersected the depth buffer (if there is no depth buffer, then this will never happen):
		if (t.x > tDepthIntersection) {
			hit.y = length(wp - pos);
			hit.z = 1;
			clr = make_float4(fmin(clr.x, 1.f), fmin(clr.y, 1.f), fmin(clr.z, 1.f), fmax(clr.w, 0.f));
			return;
		}

		// Get the value of the volume at this point. Only consider it if it's greater than SCN_MINVAL.
		const float rawSample = tex3D<float>(gvdb->volIn[chan], p.x + o.x, p.y + o.y, p.z + o.z);
		if (rawSample >= SCN_MINVAL) {
			// Apply transfer function; integrate val.w to get transmittance according to the Beer-Lambert law:
			float4 val = transfer(gvdb, rawSample);
			val.w = exp(SCN_EXTINCT * val.w * SCN_DIRECTSTEP);
			// RGB color from color channel (alpha component is unused):
			const float4 hclr = (gvdb->clr_chan == CHAN_UNDEF) ? make_float4(1, 1, 1, 1) : getColorF(gvdb, gvdb->clr_chan, p + o);
			clr.x += val.x * clr.w * (1 - val.w) * SCN_ALBEDO * hclr.x;
			clr.y += val.y * clr.w * (1 - val.w) * SCN_ALBEDO * hclr.y;
			clr.z += val.z * clr.w * (1 - val.w) * SCN_ALBEDO * hclr.z;
			clr.w *= val.w;
		}

		// Step forwards.
		p += wpt;
		wp += wpt;		
		t.x += dt;
	}			
	hit.y = t.x;  // length(wp - pos);
	clr = make_float4(fmin(clr.x, 1.f), fmin(clr.y, 1.f), fmin(clr.z, 1.f), fmax(clr.w, 0.f));
}



//----------------------------- MASTER RAYCAST FUNCTION
// 1. Performs empty skipping of GVDB hiearchy
// 2. Checks input depth buffer [if set]
// 3. Calls the specified 'brickFunc' when a brick is hit, for custom behavior
// 4. Returns a color and/or surface hit and normal
//
__device__ void rayCast ( VDBInfo* gvdb, uchar chan, float3 pos, float3 dir, float3& hit, float3& norm, float4& clr, gvdbBrickFunc_t brickFunc )
{
	int		nodeid[MAXLEV];					// level variables
	float	tMax[MAXLEV];
	int		b;	

	// GVDB - Iterative Hierarchical 3DDA on GPU
	float3 vmin;	
	int lev = gvdb->top_lev;
	nodeid[lev]		= 0;		// rootid ndx
	float3 tStart	= rayBoxIntersect ( pos, dir, gvdb->bmin, gvdb->bmax );	// intersect ray with bounding box	
	VDBNode* node	= getNode ( gvdb, lev, nodeid[lev], &vmin );			// get root VDB node	
	if ( tStart.z == NOHIT ) return;	

	// 3DDA variables		
	tStart.x += gvdb->epsilon;
	tMax[lev] = tStart.y -gvdb->epsilon;
	int iter;

	HDDAState dda;
	dda.SetFromRay(pos, dir, tStart);
	dda.Prepare(vmin, gvdb->vdel[lev]);
	const float tDepthIntersection = getRayDepthBufferMax(dir); // The t.x at which the ray intersects the depth buffer

	for (iter=0; iter < MAX_ITER && lev > 0 && lev <= gvdb->top_lev && dda.p.x >=0 && dda.p.y >=0 && dda.p.z >=0 && dda.p.x <= gvdb->res[lev] && dda.p.y <= gvdb->res[lev] && dda.p.z <= gvdb->res[lev]; iter++ ) {
		
		dda.Next();

		// Test to see if we've intersected the depth buffer (if there is no depth buffer, then this will never happen):
		if (dda.t.x > tDepthIntersection) {
			hit.z = 0;
			return;
		}

		// node active test
		b = (((int(dda.p.z) << gvdb->dim[lev]) + int(dda.p.y)) << gvdb->dim[lev]) + int(dda.p.x);	// bitmaskpos
		if ( isBitOn ( gvdb, node, b ) ) {						// check vdb bitmask for voxel occupancy						
			if ( lev == 1 ) {										// enter brick function..
				nodeid[0] = getChild ( gvdb, node, b ); 
				dda.t.x += gvdb->epsilon;
				(*brickFunc) (gvdb, chan, nodeid[0], dda.t, pos, dir, hit, norm, clr);
				if ( clr.w <= 0) {
					clr.w = 0; 
					return; 
				}														// deep termination				
				if (hit.z != NOHIT) return;								// surface termination												

				dda.Step();
			} else {				
				lev--;													// step down tree
				nodeid[lev]	= getChild ( gvdb, node, b );				// get child 
				node		= getNode ( gvdb, lev, nodeid[lev], &vmin );// child node
				dda.t.x += gvdb->epsilon;								// make sure we start inside child
				tMax[lev] = dda.t.y -gvdb->epsilon;						// t.x = entry point, t.y = exit point							
				dda.Prepare(vmin, gvdb->vdel[lev]);						// start dda at next level down
			}
		} else {
			// empty voxel, step DDA
			dda.Step();
		}
		while ( dda.t.x > tMax[lev] && lev <= gvdb->top_lev ) {
			lev++;												// step up tree
			if ( lev <= gvdb->top_lev ) {		
				node	= getNode ( gvdb, lev, nodeid[lev], &vmin );
				dda.Prepare(vmin, gvdb->vdel[lev]);				// restore dda at next level up
			}
		}
	}	
}


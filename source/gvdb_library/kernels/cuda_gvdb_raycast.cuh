//--------------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2017, NVIDIA Corporation. 
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer 
//    in the documentation and/or  other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived 
//    from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
// BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
//----------------------------------------------------------------------------------
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

// gvdbBrickFunc ( char shade, int nodeid, float3 t, float3 pos, float3 dir, float3& pstep, float3& hit, float3& norm, float4& clr );
typedef void(*gvdbBrickFunc_t)( char, int, float3, float3, float3, float3&, float3&, float3&, float4& );

#define MAXLEV			5
#define MAX_ITER		512
#define EPS				0.02

#define LO		0
#define	MID		1.0
#define	HI		2.0

inline __device__ float getTricubic ( float3 p, float3 offs, float3 vmin, float3 vdel )
{ 
	float tv[9];

	// find bottom-left corner of local 3x3x3 group		
	float3  q = floor3(p + offs) - MID;				// move to bottom-left corner	
	
	// evaluate tri-cubic
	float3 tb = frac3(p) * 0.5 + 0.25;
	float3 ta = (1.0-tb);
	float3 ta2 = ta*ta;
	float3 tb2 = tb*tb;
	float3 tab = ta*tb*2.0;

	// lookup 3x3x3 local neighborhood
	tv[0] = tex3D ( volTexIn, q.x,		q.y,		q.z );
	tv[1] = tex3D ( volTexIn, q.x+MID,	q.y,		q.z );
	tv[2] = tex3D ( volTexIn, q.x+HI,	q.y,		q.z );
	tv[3] = tex3D ( volTexIn, q.x,		q.y+MID,	q.z );
	tv[4] = tex3D ( volTexIn, q.x+MID,	q.y+MID,	q.z );
	tv[5] = tex3D ( volTexIn, q.x+HI,	q.y+MID,	q.z );
	tv[6] = tex3D ( volTexIn, q.x,		q.y+HI,		q.z );
	tv[7] = tex3D ( volTexIn, q.x+MID,	q.y+HI,		q.z );
	tv[8] = tex3D ( volTexIn, q.x+HI,	q.y+HI,		q.z );

	float3 abc = make_float3 (	tv[0]*ta2.x + tv[1]*tab.x + tv[2]*tb2.x, 
								tv[3]*ta2.x + tv[4]*tab.x + tv[5]*tb2.x,
								tv[6]*ta2.x + tv[7]*tab.x + tv[8]*tb2.x );

	tv[0] = tex3D ( volTexIn, q.x,		q.y,		q.z+MID );
	tv[1] = tex3D ( volTexIn, q.x+MID,	q.y,		q.z+MID );
	tv[2] = tex3D ( volTexIn, q.x+HI,	q.y,		q.z+MID );
	tv[3] = tex3D ( volTexIn, q.x,		q.y+MID,	q.z+MID );
	tv[4] = tex3D ( volTexIn, q.x+MID,	q.y+MID,	q.z+MID );
	tv[5] = tex3D ( volTexIn, q.x+HI,	q.y+MID,	q.z+MID );
	tv[6] = tex3D ( volTexIn, q.x,		q.y+HI,		q.z+MID );
	tv[7] = tex3D ( volTexIn, q.x+MID,	q.y+HI,		q.z+MID );
	tv[8] = tex3D ( volTexIn, q.x+HI,	q.y+HI,		q.z+MID );

	float3 def = make_float3 (	tv[0]*ta2.x + tv[1]*tab.x + tv[2]*tb2.x, 
								tv[3]*ta2.x + tv[4]*tab.x + tv[5]*tb2.x,
								tv[6]*ta2.x + tv[7]*tab.x + tv[8]*tb2.x );

	tv[0] = tex3D ( volTexIn, q.x,		q.y,		q.z+HI );
	tv[1] = tex3D ( volTexIn, q.x+MID,	q.y,		q.z+HI );
	tv[2] = tex3D ( volTexIn, q.x+HI,	q.y,		q.z+HI );
	tv[3] = tex3D ( volTexIn, q.x,		q.y+MID,	q.z+HI );
	tv[4] = tex3D ( volTexIn, q.x+MID,	q.y+MID,	q.z+HI );
	tv[5] = tex3D ( volTexIn, q.x+HI,	q.y+MID,	q.z+HI );
	tv[6] = tex3D ( volTexIn, q.x,		q.y+HI,		q.z+HI );
	tv[7] = tex3D ( volTexIn, q.x+MID,	q.y+HI,		q.z+HI );
	tv[8] = tex3D ( volTexIn, q.x+HI,	q.y+HI,		q.z+HI );

	float3 ghi = make_float3 (	tv[0]*ta2.x + tv[1]*tab.x + tv[2]*tb2.x, 
								tv[3]*ta2.x + tv[4]*tab.x + tv[5]*tb2.x,
								tv[6]*ta2.x + tv[7]*tab.x + tv[8]*tb2.x );
	
	float3 jkl = make_float3 (	abc.x*ta2.y + abc.y*tab.y + abc.z*tb2.y, 
								def.x*ta2.y + def.y*tab.y + def.z*tb2.y,
								ghi.x*ta2.y + ghi.y*tab.y + ghi.z*tb2.y );

	return jkl.x*ta2.z + jkl.y*tab.z + jkl.z*tb2.z;
}
inline __device__ float getTrilinear ( float3 wp, float3 offs, float3 vmin, float3 vdel )
{
	float3 p = offs + (wp-vmin)/vdel;		// sample point in index coords		
	return tex3D ( volTexIn, p.x, p.y, p.z );
}

#ifdef CUDA_PATHWAY
	inline __device__ unsigned char getVolSampleC ( uchar chan, float3 wpos )
	{
		float3 offs, vmin, vdel; uint64 nid;
		VDBNode* node = getNodeAtPoint ( wpos, &offs, &vmin, &vdel, &nid );				// find vdb node at point
		if ( node == 0x0 ) return 0;
		float3 p = 	offs + (wpos-vmin)/vdel;
		return tex3D<uchar> ( volIn[chan], p.x, p.y, p.z );
	}
	inline __device__ float getVolSampleF ( uchar chan, float3 wpos )
	{
		float3 offs, vmin, vdel; uint64 nid;
		VDBNode* node = getNodeAtPoint ( wpos, &offs, &vmin, &vdel, &nid );				// find vdb node at point
		if ( node == 0x0 ) return 0;
		float3 p = 	offs + (wpos-vmin)/vdel;
		return tex3D<float> ( volIn[chan], p.x, p.y, p.z );
	}
#endif

inline __device__ float3 getGradient ( float3 p )
{
	float3 g;
	// note: must use +/- 0.5 since apron may only be 1 voxel wide (cannot go beyond brick)
	g.x = 1.0* (tex3D ( volTexIn, p.x-.5, p.y,   p.z  ) - tex3D ( volTexIn, p.x+.5, p.y, p.z ));
	g.y = 1.0* (tex3D ( volTexIn, p.x,   p.y-.5, p.z  ) - tex3D ( volTexIn, p.x, p.y+.5, p.z ));
	g.z = 1.0* (tex3D ( volTexIn, p.x,   p.y,   p.z-.5) - tex3D ( volTexIn, p.x, p.y, p.z+.5 )); 
	g = normalize ( g );
	return g;
}
inline __device__ float3 getGradientLevelSet ( float3 offs, float3 pos, float3 vmin, float3 vdel )
{
	// tri-linear filtered gradient 
	// (assumes atlas has linear hardware filtering on)
	float3 vs = gvdb.voxelsize*0.5 / vdel;
	float3 g, p = offs + (pos-vmin)/vdel;	
	g.x = 0.5* (tex3D ( volTexIn, p.x+vs.x, p.y,   p.z  ) - tex3D ( volTexIn, p.x-vs.x, p.y, p.z ));
	g.y = 0.5* (tex3D ( volTexIn, p.x,   p.y+vs.y, p.z  ) - tex3D ( volTexIn, p.x, p.y-vs.y, p.z ));
	g.z = 0.5* (tex3D ( volTexIn, p.x,   p.y,   p.z+vs.z) - tex3D ( volTexIn, p.x, p.y, p.z-vs.z )); 
	g = normalize ( g );
	return g;
}
inline __device__ float3 getGradientTricubic ( float3 p, float3 offs, float3 vmin, float3 vdel )
{
	// tri-cubic filtered gradient
	const float vs = 0.5;
	float3 g;
	g.x = (getTricubic ( p+make_float3(-vs,0,0), offs,vmin,vdel)	- getTricubic ( p+make_float3(vs,0,0), offs,vmin,vdel))/(2*vs);
	g.y = (getTricubic ( p+make_float3(0,-vs,0), offs,vmin,vdel)	- getTricubic ( p+make_float3(0,vs,0), offs,vmin,vdel))/(2*vs);
	g.z = (getTricubic ( p+make_float3(0,0,-vs), offs,vmin,vdel)	- getTricubic ( p+make_float3(0,0,vs), offs,vmin,vdel))/(2*vs); 
	g = normalize ( g );
	return g;
}


__device__ float3 rayTricubic ( float3& p, float3 o, float3 rpos, float3 rdir, float3 vmin, float3 vdel )
{
	float3 pt = SCN_FSTEP * gvdb.voxelsize.x * rdir;

	for ( int i=0; i < 512; i++ ) {					
		if (  getTricubic ( p, o, vmin, vdel ) >= gvdb.thresh.x )			// tricubic test
			return p*vdel + vmin;
		p += pt;
	}
	return make_float3(NOHIT, NOHIT, NOHIT);
}

__device__ float3 rayTrilinear ( float3& p, float3 o, float3 rpos, float3 rdir, float3 vmin, float3 vdel )
{
	float dt = SCN_FSTEP * gvdb.voxelsize.x;		
	float3 pt = dt*rdir/vdel;

	for ( int i=0; i < 512; i++ ) {
		if ( tex3D ( volTexIn, p.x+o.x, p.y+o.y, p.z+o.z ) >= gvdb.thresh.x )	// trilinear test
			return p*vdel + vmin;		
		p += pt;		
	}
	return make_float3(NOHIT, NOHIT, NOHIT);
}

__device__ float3 rayLevelSet ( float t, float3 offs, float3 rpos, float3 rdir, float3 vmin, float3 vdel )
{
	float dt = SCN_FSTEP * gvdb.voxelsize.x;	
	t -= gvdb.voxelsize.x;
	float3 wpt = dt*rdir;
	float3 wp = rpos + t*rdir;		

	for ( int i=0; i < 64; i++ ) {			
		if ( getTrilinear ( wp, offs, vmin, vdel ) < 0 ) {			
			return wp;
		}
		wp += wpt;		
	}
	return make_float3(NOHIT, NOHIT, NOHIT);
}

inline __device__ uchar4 getColor ( uchar chan, float3 p )
{
	#ifdef CUDA_PATHWAY
		return tex3D<uchar4> ( volIn[chan], (int) p.x, (int) p.y, (int) p.z );
	#else
		return make_uchar4(255,255,255,255);
	#endif
}
inline __device__ float4 getColorF ( uchar chan, float3 p )
{
	#ifdef CUDA_PATHWAY	
		return make_float4 (tex3D<uchar4> ( volIn[chan], (int) p.x, (int) p.y, (int) p.z ) );
	#else
		return make_float4(1,1,1,1);
	#endif
}

//----------- RAY CASTING

#define EPSTEST(a,b,c)	(a>b-c && a<b+c)

// SurfaceBrick - Trace into brick to find surface
__device__ void raySurfaceBrick ( char shade, int nodeid, float3 t, float3 pos, float3 dir, float3& pStep, float3& hit, float3& norm, float4& hclr )
{
	dir = normalize(dir);
	const float eps = 0.0001;
	float3 vmin;
	VDBNode* node	= getNode ( 0, nodeid, &vmin );				// Get the VDB leaf node
	
	float3	p, tDel, tSide, mask;								// 3DDA variables	
	float3  o = make_float3( node->mValue ) ;	// Atlas sub-volume to trace	

	PREPARE_DDA_LEAF
	
	for (int iter=0; iter < MAX_ITER && p.x >=0 && p.y >=0 && p.z >=0 && p.x < gvdb.res[0] && p.y < gvdb.res[0] && p.z < gvdb.res[0]; iter++) {	

		if ( tex3D ( volTexIn, p.x+o.x+.5, p.y+o.y+.5, p.z+o.z+.5 ) > gvdb.thresh.x-0.05 ) {		// test texture atlas

			// smoothing
			switch ( shade ) {				
			case SHADE_VOXEL: {				// blocks. voxel hitsc and face normals
				float3 voxmin = p * gvdb.vdel[0] +vmin;
				t = rayBoxIntersect ( pos, dir, voxmin, voxmin + gvdb.voxelsize );		
				if (t.z == NOHIT) {
					hit.x = NOHIT;
					continue;
				}
				hit = getRayPoint ( pos, dir, t.x );				
				norm.x = EPSTEST(hit.x, voxmin.x + gvdb.voxelsize.x, eps) ? 1 : (EPSTEST(hit.x, voxmin.x, eps) ? -1 : 0);
				norm.y = EPSTEST(hit.y, voxmin.y + gvdb.voxelsize.y, eps) ? 1 : (EPSTEST(hit.y, voxmin.y, eps) ? -1 : 0);
				norm.z = EPSTEST(hit.z, voxmin.z + gvdb.voxelsize.z, eps) ? 1 : (EPSTEST(hit.z, voxmin.z, eps) ? -1 : 0);
				if ( gvdb.clr_chan != CHAN_UNDEF ) hclr = getColorF ( gvdb.clr_chan, p+o );
				} return;			
			case SHADE_TRILINEAR: {				// tri-linear surface with central diff normals
				t.x = length( (p* gvdb.vdel[0] +vmin) + (gvdb.voxelsize*0.5) - pos);		// find t value at center of voxel								
				//t.x = SCN_PSTEP * ceil ( t.x / SCN_PSTEP );
				hit = rayTrilinear ( p, o, pos, dir, vmin, gvdb.vdel[0] );		// p updated here
				if ( hit.x != NOHIT ) {					
					norm = getGradient ( p+o );					
					if ( gvdb.clr_chan != CHAN_UNDEF ) hclr = getColorF ( gvdb.clr_chan, p+o );	
					return;
				}
				} break;
			case SHADE_TRICUBIC: {				// tri-cubic surface with tricubic normals
				t.x = length( (p* gvdb.vdel[0] +vmin) +(gvdb.voxelsize*0.5) - pos);		// find t value at center of voxel											
				//t.x = PSTEP * ceil ( t.x/PSTEP );
				hit = rayTricubic ( p, o, pos, dir, vmin, gvdb.vdel[0] );				
				if ( hit.x != NOHIT ) {					
					//norm = getGradient ( p+o );
					norm = getGradientTricubic ( hit, o, vmin, make_float3(gvdb.noderange[0])*gvdb.voxelsize/(gvdb.res[0]-1)  );					
					if ( gvdb.clr_chan != CHAN_UNDEF ) hclr = getColorF ( gvdb.clr_chan, p+o );					
					return;
				}
				} break;
			};			
		}	

		NEXT_DDA

		STEP_DDA
	}
}

#define VOXEL_EPS	0.0001

// SurfaceVoxelBrick - Trace brick to render voxels as cubes
__device__ void raySurfaceVoxelBrick ( char shade, int nodeid, float3 t, float3 pos, float3 dir, float3& pStep, float3& hit, float3& norm, float4& hclr )
{
	float3 vmin;
	VDBNode* node	= getNode ( 0, nodeid, &vmin );				// Get the VDB leaf node
	float3	p, tDel, tSide, mask;								// 3DDA variables	
	float3  o = make_float3( node->mValue ) ;	// Atlas sub-volume to trace	

	PREPARE_DDA_LEAF
	
	for (int iter=0; iter < MAX_ITER && p.x >=0 && p.y >=0 && p.z >=0 && p.x < gvdb.res[0] && p.y < gvdb.res[0] && p.z < gvdb.res[0]; iter++) {	

		if ( tex3D ( volTexIn, p.x+o.x+.5, p.y+o.y+.5, p.z+o.z+.5 ) > gvdb.thresh.x ) {		// test texture atlas
			vmin += p * gvdb.vdel[0];		// voxel location in world
			t = rayBoxIntersect ( pos, dir, vmin, vmin + gvdb.voxelsize );		
			if (t.z == NOHIT) {
				hit.x = NOHIT;
				continue;
			}
			hit = getRayPoint ( pos, dir, t.x );				
			norm.x = EPSTEST(hit.x, vmin.x + gvdb.voxelsize.x, VOXEL_EPS) ? 1 : (EPSTEST(hit.x, vmin.x, VOXEL_EPS) ? -1 : 0);
			norm.y = EPSTEST(hit.y, vmin.y + gvdb.voxelsize.y, VOXEL_EPS) ? 1 : (EPSTEST(hit.y, vmin.y, VOXEL_EPS) ? -1 : 0);
			norm.z = EPSTEST(hit.z, vmin.z + gvdb.voxelsize.z, VOXEL_EPS) ? 1 : (EPSTEST(hit.z, vmin.z, VOXEL_EPS) ? -1 : 0);
			if ( gvdb.clr_chan != CHAN_UNDEF ) hclr = getColorF ( gvdb.clr_chan, p+o );
			return;	
		}
		NEXT_DDA
		STEP_DDA
	}
}

// SurfaceTrilinearBrick - Trace brick to render surface with trilinear smoothing
__device__ void raySurfaceTrilinearBrick ( char shade, int nodeid, float3 t, float3 pos, float3 dir, float3& pStep, float3& hit, float3& norm, float4& hclr )
{
	float3 vmin;
	VDBNode* node	= getNode ( 0, nodeid, &vmin );			// Get the VDB leaf node	
	float3  o = make_float3( node->mValue ) ;				// Atlas sub-volume to trace	
	float3	p = (pos + t.x*dir - vmin) / gvdb.vdel[0];					// sample point in index coords			
	t.x = SCN_PSTEP * ceil ( t.x / SCN_PSTEP );

	for (int iter=0; iter < MAX_ITER && p.x >=0 && p.y >=0 && p.z >=0 && p.x < gvdb.res[0] && p.y < gvdb.res[0] && p.z < gvdb.res[0]; iter++) {
	
		if ( tex3D ( volTexIn, p.x+o.x, p.y+o.y, p.z+o.z ) >= gvdb.thresh.x ) {
			hit = p*gvdb.vdel[0] + vmin;
			norm = getGradient ( p+o );
			if ( gvdb.clr_chan != CHAN_UNDEF ) hclr = getColorF ( gvdb.clr_chan, p+o );
			return;	
		}	
		p += SCN_PSTEP*dir;		
		t.x += SCN_PSTEP;
	}
}


/*__device__ void raySurfaceTrilinearBrick ( char shade, int nodeid, float3 t, float3 pos, float3 dir, float3& pStep, float3& hit, float3& norm, float4& hclr )
{
	float3 vmin;
	VDBNode* node	= getNode ( 0, nodeid, &vmin );			// Get the VDB leaf node	
	float3  o = make_float3( node->mValue ) ;				// Atlas sub-volume to trace	
	float3	p = (pos + t.x*dir - vmin) / gvdb.vdel[0];					// sample point in index coords			
	float3  v;
	t.x = SCN_PSTEP * ceil ( t.x / SCN_PSTEP );

	for (int iter=0; iter < MAX_ITER && p.x >=0 && p.y >=0 && p.z >=0 && p.x < gvdb.res[0] && p.y < gvdb.res[0] && p.z < gvdb.res[0]; iter++) {
	
		v.z = tex3D ( volTexIn, p.x+o.x, p.y+o.y, p.z+o.z );
		if ( v.z >= gvdb.thresh.x ) {			
			p -= SCN_FSTEP*dir;
			v.x = tex3D ( volTexIn, p.x+o.x, p.y+o.y, p.z+o.z );
			v.y = (v.z-gvdb.thresh.x)/(v.z-v.x);			
			p += (1.0-v.y)*SCN_FSTEP*dir;
			hit = p*gvdb.vdel[0] + vmin;
			norm = getGradient ( p+o );
			if ( gvdb.clr_chan != CHAN_UNDEF ) hclr = getColorF ( gvdb.clr_chan, p+o );
			return;	
		}	
		p += SCN_PSTEP*dir;		
		t.x += SCN_PSTEP;
	}
}*/

// SurfaceTricubicBrick - Trace brick to render surface with tricubic smoothing
__device__ void raySurfaceTricubicBrick ( char shade, int nodeid, float3 t, float3 pos, float3 dir, float3& pStep, float3& hit, float3& norm, float4& hclr )
{
	float3 vmin;
	VDBNode* node	= getNode ( 0, nodeid, &vmin );			// Get the VDB leaf node	
	float3  o = make_float3( node->mValue ) ;				// Atlas sub-volume to trace	
	float3	p = (pos + t.x*dir - vmin) / gvdb.vdel[0];		// sample point in index coords		
	float3  v;

	for (int iter=0; iter < MAX_ITER && p.x >=0 && p.y >=0 && p.z >=0 && p.x < gvdb.res[0] && p.y < gvdb.res[0] && p.z < gvdb.res[0]; iter++) {

		v.z = getTricubic ( p, o, vmin, gvdb.vdel[0] );
		if ( v.z >= gvdb.thresh.x ) {
			v.x = getTricubic ( p - SCN_FSTEP*dir, o, vmin, gvdb.vdel[0] );			
			v.y = (v.z-gvdb.thresh.x)/(v.z-v.x);
			p += -v.y*SCN_FSTEP*dir;
			hit = p*gvdb.vdel[0] + vmin;
			//hit = rayTricubic ( p, o, pos, dir, vmin, gvdb.vdel[0] );
			norm = getGradientTricubic ( p, o, vmin, gvdb.vdel[0] );
			if ( gvdb.clr_chan != CHAN_UNDEF ) hclr = getColorF ( gvdb.clr_chan, p+o );					
			return;			
		}	
		p += SCN_PSTEP*dir;		
		t.x += SCN_PSTEP;
	}
}


inline __device__ float getLinearDepth( float* depthBufFloat )
{
	// Find x y pixel coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
  
	// Decode depth from Z-buffer with depth_24_stencil_8
	float tmpfval = depthBufFloat[(SCN_HEIGHT - 1 - y) * SCN_WIDTH + x];
	uint tmpival = __float_as_int(tmpfval);
	tmpival = (tmpival & 0xFFFFFF00) >> 8;
	float z = __saturatef(static_cast< float >(tmpival) / 16777215.0f);

	/* 
	// Dont understand why this doesn't work, it is slightly off, had to do a full reprojection below
	float n = scn.camnear;
	float f = scn.camfar;
	return (-n * f / (f - n)) / (z - (f / (f - n)));
	*/

	float xx = (float(x + 0.5f) / float(SCN_WIDTH)) * 2.0f - 1.0f;
	float yy = (float(y + 0.5f) / float(SCN_HEIGHT)) * 2.0f - 1.0f;
	float zz = (z * 2.0f - 1.0f);
	float4 p = make_float4(xx, yy, zz, 1.0f);
	
	float4 wp;
#ifdef CUDA_PATHWAY			// Temporary. camivprow does not work w Optix. See SCN_DBUF
	wp.x = scn.camivprow0.x * p.x + scn.camivprow1.x * p.y + scn.camivprow2.x * p.z + scn.camivprow3.x;
	wp.y = scn.camivprow0.y * p.x + scn.camivprow1.y * p.y + scn.camivprow2.y * p.z + scn.camivprow3.y;
	wp.z = scn.camivprow0.z * p.x + scn.camivprow1.z * p.y + scn.camivprow2.z * p.z + scn.camivprow3.z;
	wp.w = scn.camivprow0.w * p.x + scn.camivprow1.w * p.y + scn.camivprow2.w * p.z + scn.camivprow3.w;
#endif
	float3 viewPos = make_float3(wp.x / wp.w, wp.y / wp.w, wp.z / wp.w);
	return sqrt(viewPos.x * viewPos.x + viewPos.y * viewPos.y + viewPos.z * viewPos.z);
}

// SurfaceDepthBrick - Trace into brick to find surface, with early termination based on depth buffer
__device__ void raySurfaceDepthBrick ( char shade, int nodeid, float3 t, float3 pos, float3 dir, float3& pStep, float3& hit, float3& norm, float4& hclr )
{
	dir = normalize(dir);
	const float eps = 0.0001;
	float3 vmin;
	VDBNode* node	= getNode ( 0, nodeid, &vmin );				// Get the VDB leaf node
	
	float3	p, tDel, tSide, mask;								// 3DDA variables	
	float3  o = make_float3( node->mValue ) ;	// Atlas sub-volume to trace	

	PREPARE_DDA_LEAF
	
	for (int iter=0; iter < MAX_ITER && p.x >=0 && p.y >=0 && p.z >=0 && p.x <= gvdb.res[0] && p.y <= gvdb.res[0] && p.z <= gvdb.res[0]; iter++) {	

		if ( tex3D ( volTexIn, p.x+o.x+.5, p.y+o.y+.5, p.z+o.z+.5 ) > gvdb.thresh.x ) {		// test texture atlas

			// smoothing
			switch ( shade ) {				
			case SHADE_VOXEL: {				// blocks. voxel hitsc and face normals
				float3 voxmin = p * gvdb.vdel[0] +vmin;
				t = rayBoxIntersect ( pos, dir, voxmin, voxmin + gvdb.voxelsize );		
				if (t.z == NOHIT) {
					hit.x = NOHIT;
					continue;
				}
				if (t.x > getLinearDepth( SCN_DBUF)) {
					hit.x = NOHIT;
					return;
				}
				hit = getRayPoint ( pos, dir, t.x );				
				norm.x = EPSTEST(hit.x, voxmin.x + gvdb.voxelsize.x, eps) ? 1 : (EPSTEST(hit.x, voxmin.x, eps) ? -1 : 0);
				norm.y = EPSTEST(hit.y, voxmin.y + gvdb.voxelsize.y, eps) ? 1 : (EPSTEST(hit.y, voxmin.y, eps) ? -1 : 0);
				norm.z = EPSTEST(hit.z, voxmin.z + gvdb.voxelsize.z, eps) ? 1 : (EPSTEST(hit.z, voxmin.z, eps) ? -1 : 0);
				if ( gvdb.clr_chan != CHAN_UNDEF ) hclr = getColorF ( gvdb.clr_chan, p+o );
				} return;			
			case SHADE_TRILINEAR: {				// tri-linear surface with central diff normals
				t.x = length( (p* gvdb.vdel[0] +vmin) + (gvdb.voxelsize*0.5) - pos);		// find t value at center of voxel								
				//t.x = SCN_PSTEP * ceil ( t.x / SCN_PSTEP );
				hit = rayTrilinear ( p, o, pos, dir, vmin, gvdb.vdel[0] );		// p updated here
				if ( hit.x != NOHIT ) {					
					norm = getGradient ( p );
					//norm = getGradient ( o, hit, vmin,  make_float3(gvdb.noderange[0])*gvdb.voxelsize/(gvdb.res[0]-1) );
					if ( gvdb.clr_chan != CHAN_UNDEF ) hclr = getColorF ( gvdb.clr_chan, p+o );					
					return;
				}
				} break;
			case SHADE_TRICUBIC: {				// tri-cubic surface with tricubic normals
				t.x = length( (p* gvdb.vdel[0] +vmin) +(gvdb.voxelsize*0.5) - pos);		// find t value at center of voxel											
				//t.x = PSTEP * ceil ( t.x/PSTEP );
				hit = rayTricubic ( p, o, pos, dir, vmin, gvdb.vdel[0] );
				if ( hit.x != NOHIT ) {					
					norm = getGradientTricubic ( p, o, vmin, make_float3(gvdb.noderange[0])*gvdb.voxelsize/(gvdb.res[0]-1)  );
					if ( gvdb.clr_chan != CHAN_UNDEF ) hclr = getColorF ( gvdb.clr_chan, p+o );	
					return;
				}
				} break;
			};						
			
		}	

		NEXT_DDA

		STEP_DDA
	}
}

// LevelSet brick - Trace into brick to find level set surface
__device__ void rayLevelSetBrick ( char shade, int nodeid, float3 t, float3 pos, float3 dir, float3& pStep, float3& hit, float3& norm, float4& clr )
{
	float3 vmin;
	VDBNode* node	= getNode ( 0, nodeid, &vmin );				// Get the VDB leaf node	
	
	float3	p, tDel, tSide, mask;								// 3DDA variables	
	float3  o = make_float3( node->mValue ) ;	// Atlas sub-volume to trace	

	PREPARE_DDA_LEAF
	
	for (int iter=0; iter < MAX_ITER && p.x >=0 && p.y >=0 && p.z >=0 && p.x <= gvdb.res[0] && p.y <= gvdb.res[0] && p.z <= gvdb.res[0]; iter++) {	

		if ( tex3D ( volTexIn, p.x+o.x+.5, p.y+o.y+.5, p.z+o.z+.5 ) < 0 ) {			// test atlas for zero crossing
			t.x = length( (p* gvdb.vdel[0] +vmin) +(gvdb.voxelsize*0.5) - pos);		// find t value at center of voxel								
			hit = rayLevelSet ( t.x, o, pos, dir, vmin, gvdb.vdel[0] );				
			if ( hit.x != NOHIT ) {
				norm = getGradientLevelSet ( o, hit, vmin, make_float3(gvdb.noderange[0])*gvdb.voxelsize/(gvdb.res[0]-1) );
				return;
			}
		}	
		NEXT_DDA

		STEP_DDA
	}
}

// EmptySkip brick - Return brick itself (do not trace values)
__device__ void rayEmptySkipBrick ( char shade, int nodeid, float3 t, float3 pos, float3 dir, float3& pStep, float3& hit, float3& norm, float4& clr )
{
	float3 vmin;
	VDBNode* node	= getNode ( 0, nodeid, &vmin );			// Get the VDB leaf node	
	float3	p;			
	p = ( pos + t.x*dir - vmin) / gvdb.vdel[0];
	hit = p * gvdb.vdel[0] + vmin;							// Return brick hit
}

// Shadow brick - Return deep shadow accumulation
__device__ void rayShadowBrick ( char shade, int nodeid, float3 t, float3 pos, float3 dir, float3& pStep, float3& hit, float3& norm, float4& clr )
{
	float3 vmin;
	VDBNode* node = getNode ( 0, nodeid, &vmin );			// Get the VDB leaf node	
	t.x += EPS;												// make sure we start inside
	t.y -= EPS;												// make sure we end insidoke	
	float3 o = make_float3( node->mValue );					// atlas sub-volume to trace	
	float3 p = (pos + t.x*dir - vmin) / gvdb.vdel[0];		// sample point in index coords
	float3 pt = SCN_PSTEP * dir;								// index increment	
	float val = 0;

	// accumulate remaining voxels	
	for (; clr.w < 1 && p.x >=0 && p.y >=0 && p.z >=0 && p.x < gvdb.res[0] && p.y < gvdb.res[0] && p.z < gvdb.res[0];) {		
		val = exp ( SCN_EXTINCT * transfer( tex3D( volTexIn, p.x+o.x, p.y+o.y, p.z+o.z )).w * SCN_SSTEP/(1.0 + t.x * 0.4) );		// 0.4 = shadow gain
		clr.w = 1.0 - (1.0-clr.w) * val;
		p += pt;	
		t.x += SCN_SSTEP;
	}	
}

// DeepBrick - Sample into brick for deep volume raytracing
__device__ void rayDeepBrick ( char shade, int nodeid, float3 t, float3 pos, float3 dir, float3& pstep, float3& hit, float3& norm, float4& clr )
{
	float3 vmin;
	VDBNode* node = getNode ( 0, nodeid, &vmin );			// Get the VDB leaf node		
	
	//t.x = SCN_PSTEP * ceil( t.x / SCN_PSTEP );						// start on sampling wavefront	

	float3 o = make_float3( node->mValue );					// atlas sub-volume to trace
	float3 wp = pos + t.x*dir;	
	float3 p = (wp-vmin) / gvdb.vdel[0];					// sample point in index coords	
	float3 wpt = SCN_PSTEP*dir * gvdb.vdel[0];					// world increment
	float4 val = make_float4(0,0,0,0);
	float4 hclr;
	int iter = 0;

	// skip empty voxels
	for (iter=0; val.w < SCN_MINVAL && iter < MAX_ITER && p.x >= 0 && p.y >=0 && p.z >=0 && p.x < gvdb.res[0] && p.y < gvdb.res[0] && p.z < gvdb.res[0]; iter++) {		
		val.w = transfer ( tex3D ( volTexIn, p.x+o.x, p.y+o.y, p.z+o.z ) ).w;
		p += SCN_PSTEP*dir;
		wp += wpt;
		t.x += SCN_PSTEP;
	}
	// record front hit point at first significant voxel
	if ( hit.z == NOHIT ) {
		hit = make_float3( length( wp - pos ), 0, 0 );
	}

	// accumulate remaining voxels
	for (; clr.w > SCN_ALPHACUT && iter < MAX_ITER && p.x >=0 && p.y >=0 && p.z >=0 && p.x < gvdb.res[0] && p.y < gvdb.res[0] && p.z < gvdb.res[0]; iter++) {			
		val = transfer ( tex3D ( volTexIn, p.x+o.x, p.y+o.y, p.z+o.z ) );			
		val.w = exp ( SCN_EXTINCT * val.w * SCN_PSTEP );
		hclr = (gvdb.clr_chan==CHAN_UNDEF) ? make_float4(1,1,1,1) : getColorF ( gvdb.clr_chan, p+o );
		clr.x += val.x * clr.w * (1 - val.w) * SCN_ALBEDO * hclr.x;
		clr.y += val.y * clr.w * (1 - val.w) * SCN_ALBEDO * hclr.y;
		clr.z += val.z * clr.w * (1 - val.w) * SCN_ALBEDO * hclr.z;
		clr.w *= val.w;		

		p += SCN_PSTEP*dir;		
		wp += wpt;		
		t.x += SCN_PSTEP;
	}			
	hit.y = length( wp - pos );
	clr = make_float4(fmin(clr.x, 1.f), fmin(clr.y, 1.f), fmin(clr.z, 1.f), fmax(clr.w, 0.f));	
}


	/*float3 absorb = make_float3 (0.1, 0.1, 0.1);
	float3 scatter = make_float3 (1.0, 1.0, 1.0);		
	float3 extinction = -scatter-absorb;
	float3 albedo = scatter/(scatter+absorb); //* 4.0;  //single scattering*/

	/*val = transfer ( tex3D ( volTexIn, p.x+o.x, p.y+o.y, p.z+o.z ) );
	norm = getGradient ( o, wp, vmin, gvdb.vdel[0] );				
	float diffuse = max(0.0f, dot( norm, normalize( lightpos - wp ) ));	
	cv = colormap(val) * diffuse * 2.0;*/
		
	/*snorm = normalize(lightpos-wp);
	rayShadow ( gvdb.top_lev, 0, wp+norm*2, snorm, shadow ); 
	clr.x += cv.x * albedo.x * (1.0-shadow) * dt;			
	clr.y += cv.y * albedo.y * (1.0-shadow) * dt;
	clr.z += cv.z * albedo.z * (1.0-shadow) * dt;*/

	/* val.w = exp ( extinction.x * val.w * PSTEP );
	dt = (1.0-clr.w) * (1 - val.w);
	clr.x += val.x * albedo.x * dt;			
	clr.y += val.y * albedo.y * dt;
	clr.z += val.z * albedo.z * dt;
	clr.w = 1.0 - (1-clr.w) * val.w;
	if ( clr.w >= 1.0 ) {clr.w=1.0; return;}*/



//----------------------------- MASTER RAYCAST FUNCTION
// 1. Performs empty skipping of GVDB hiearchy
// 2. Checks input depth buffer [if set]
// 3. Calls the specified 'brickFunc' when a brick is hit, for custom behavior
// 4. Returns a color and/or surface hit and normal
//
__device__ void rayCast ( char shade, int lev, int rootid, float3 pos, float3 dir, float3& hit, float3& norm, float4& clr, gvdbBrickFunc_t brickFunc )
{
	int		nodeid[MAXLEV];					// level variables
	float	tMax[MAXLEV];
	int		b;

	// GVDB - Iterative Hierarchical 3DDA on GPU
	float3 vmin;	
	nodeid[lev]		= rootid;
	float3 t		= rayBoxIntersect ( pos, dir, gvdb.bmin, gvdb.bmax );	// intersect ray with bounding box	
	VDBNode* node	= getNode ( lev, rootid, &vmin );						// get root VDB node	
	if ( t.z == NOHIT ) return;	
	
	// 3DDA variables		
	t.x += EPS;
	tMax[lev] = t.y - EPS;
	float3 pStep	= make_float3(isign3(dir));
	float3 p, tDel, tSide, mask;
	int iter;
	
	PREPARE_DDA	

	for (iter=0; iter < MAX_ITER && lev > 0 && lev <= gvdb.top_lev && p.x >=0 && p.y >=0 && p.z >=0 && p.x <= gvdb.res[lev] && p.y <= gvdb.res[lev] && p.z <= gvdb.res[lev]; iter++ ) {

		NEXT_DDA

		// depth buffer test [optional]
		if ( SCN_DBUF != 0x0 ) {
			if ( t.x > getLinearDepth ( SCN_DBUF ) ) {
				hit.x = NOHIT;
				return;
			}
		}			
		// node active test
		b = (((int(p.z) << gvdb.dim[lev]) + int(p.y)) << gvdb.dim[lev]) + int(p.x);
		if ( isBitOn ( node, b ) ) {							// check vdb bitmask for voxel occupancy						
			if ( lev == 1 ) {									// enter brick function..
				nodeid[0] = getChild ( node, b ); t.x += EPS;
				// gvdbBrickFunc_t ( char shade, int nodeid, float3 t, float3 pos, float3 dir, float3& pstep, float3& hit, float3& norm, float4& clr );
				(*brickFunc) ( shade, nodeid[0], t, pos, dir, pStep, hit, norm, clr );				
				if ( clr.w <= 0) {clr.w = 0; return; }			// deep termination
				if ( hit.x != NOHIT && shade != SHADE_VOLUME ) return;		// surface termination				
				STEP_DDA										// leaf node empty, step DDA
			} else {				
				lev--;											// step down tree
				nodeid[lev]	= getChild ( node, b );				// get child 
				node		= getNode ( lev, nodeid[lev], &vmin );	// child node
				t.x += EPS;										// make sure we start inside child
				tMax[lev] = t.y - EPS;							// t.x = entry point, t.y = exit point							
				PREPARE_DDA										// start dda at next level down
			}
		} else {			
			STEP_DDA											// empty voxel, step DDA
		}
		while ( t.x > tMax[lev] && lev <= gvdb.top_lev ) {						
			lev++;												// step up tree
			if ( lev <= gvdb.top_lev ) {		
				node	= getNode ( lev, nodeid[lev], &vmin );			
				PREPARE_DDA										// restore dda at next level up
			}
		}
	}
	hit.x = NOHIT;
}


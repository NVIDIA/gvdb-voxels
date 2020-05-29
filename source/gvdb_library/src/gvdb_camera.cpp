//--------------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2016-2018, NVIDIA Corporation. 
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
// Version 1.1: Rama Hoetzlein, 3/25/2018
//----------------------------------------------------------------------------------


#include "gvdb_camera.h"
using namespace nvdb;

// Camera3D IMPLEMENTATION 
// 
// The Camera3D transformation of an arbitrary point is:
//
//		Q' = Q * T * R * P
//
// where Q  = 3D point
//		 Q' = Screen point
//		 T = Camera3D Translation (moves Camera3D to origin)
//		 R = Camera3D Rotation (rotates Camera3D to point 'up' along Z axis)
//       P = Projection (projects points onto XY plane)
// 
// T is a unit-coordinate system translated to origin from Camera3D:
//		[1	0  0  0]
//		[0	1  0  0]
//		[0	0  1  0]
//		[-cx -cy -cz 0]		where c is Camera3D location
// R is a basis matrix:	
//
// P is a projection matrix:
//

Camera3D::Camera3D ()
{	
	
	mProjType = Perspective;
	mWire = 0;

	up_dir.Set ( 0.0, 1.0, 0 );				// frustum params
	mAspect = (float) 800.0f/600.0f;
	mDolly = 5.0;
	mFov = 40.0;	
	mNear = (float) 0.1;
	mFar = (float) 5000.0;
	mTile.Set ( 0, 0, 1, 1 );

	for (int n=0; n < 8; n++ ) mOps[n] = false;	
	mOps[0] = false;

	setOrbit ( 0, 45, 0, Vector3DF(0,0,0), 120.0, 1.0 );
	updateMatricies ();
}

bool Camera3D::pointInFrustum ( float x, float y, float z )
{
	int p;
	for ( p = 0; p < 6; p++ )
		if( frustum[p][0] * x + frustum[p][1] * y + frustum[p][2] * z + frustum[p][3] <= 0 )
			return false;
	return true;
}

bool Camera3D::boxInFrustum ( Vector3DF bmin, Vector3DF bmax)
{
	Vector3DF vmin, vmax;
	int p;
	bool ret = true;	
	for ( p = 0; p < 6; p++ ) {
		vmin.x = ( frustum[p][0] > 0 ) ? bmin.x : bmax.x;		// Determine nearest and farthest point to plane
		vmax.x = ( frustum[p][0] > 0 ) ? bmax.x : bmin.x;		
		vmin.y = ( frustum[p][1] > 0 ) ? bmin.y : bmax.y;
		vmax.y = ( frustum[p][1] > 0 ) ? bmax.y : bmin.y;
		vmin.z = ( frustum[p][2] > 0 ) ? bmin.z : bmax.z;
		vmax.z = ( frustum[p][2] > 0 ) ? bmax.z : bmin.z;
		if ( frustum[p][0]*vmax.x + frustum[p][1]*vmax.y + frustum[p][2]*vmax.z + frustum[p][3] <= 0 ) return false;		// If nearest point is outside, Box is outside
		else if ( frustum[p][0]*vmin.x + frustum[p][1]*vmin.y + frustum[p][2]*vmin.z + frustum[p][3] <= 0 ) ret = true;		// If nearest inside and farthest point is outside, Box intersects
	}
	return ret;			// No points found outside. Box must be inside.
	
}

void Camera3D::setOrbit  ( Vector3DF angs, Vector3DF tp, float dist, float dolly )
{
	setOrbit ( angs.x, angs.y, angs.z, tp, dist, dolly );
}

void Camera3D::setOrbit ( float ax, float ay, float az, Vector3DF tp, float dist, float dolly )
{
	ang_euler.Set ( ax, ay, az );
	mOrbitDist = dist;
	mDolly = dolly;
	double dx, dy, dz;
	dx = cos ( ang_euler.y * DEGtoRAD ) * sin ( ang_euler.x * DEGtoRAD ) ;
	dy = sin ( ang_euler.y * DEGtoRAD );
	dz = cos ( ang_euler.y * DEGtoRAD ) * cos ( ang_euler.x * DEGtoRAD );
	from_pos.x = tp.x + (float) dx * mOrbitDist;
	from_pos.y = tp.y + (float) dy * mOrbitDist;
	from_pos.z = tp.z + (float) dz * mOrbitDist;
	to_pos = tp;
	orbit_set_ = true;
	updateMatricies ();
}

void Camera3D::moveOrbit ( float ax, float ay, float az, float dd )
{
	ang_euler += Vector3DF(ax,ay,az);
	mOrbitDist += dd;
	
	double dx, dy, dz;
	dx = cos ( ang_euler.y * DEGtoRAD ) * sin ( ang_euler.x * DEGtoRAD ) ;
	dy = sin ( ang_euler.y * DEGtoRAD );
	dz = cos ( ang_euler.y * DEGtoRAD ) * cos ( ang_euler.x * DEGtoRAD );
	from_pos.x = to_pos.x + (float) dx * mOrbitDist;
	from_pos.y = to_pos.y + (float) dy * mOrbitDist;
	from_pos.z = to_pos.z + (float) dz * mOrbitDist;
	updateMatricies ();
}

void Camera3D::moveToPos ( float tx, float ty, float tz )
{
	to_pos += Vector3DF(tx,ty,tz);

	double dx, dy, dz;
	dx = cos ( ang_euler.y * DEGtoRAD ) * sin ( ang_euler.x * DEGtoRAD ) ;
	dy = sin ( ang_euler.y * DEGtoRAD );
	dz = cos ( ang_euler.y * DEGtoRAD ) * cos ( ang_euler.x * DEGtoRAD );
	from_pos.x = to_pos.x + (float) dx * mOrbitDist;
	from_pos.y = to_pos.y + (float) dy * mOrbitDist;
	from_pos.z = to_pos.z + (float) dz * mOrbitDist;
	updateMatricies ();
}

void Camera3D::Copy ( Camera3D& op )
{
	mDolly = op.mDolly;
	mOrbitDist = op.mOrbitDist;
	from_pos = op.from_pos;
	to_pos = op.to_pos;
	ang_euler = op.ang_euler; 
	mProjType = op.mProjType;
	mAspect = op.mAspect;
	mFov = op.mFov;
	mNear = op.mNear;
	mFar = op.mFar;
	updateMatricies ();
}

void Camera3D::setAngles ( float ax, float ay, float az )
{
	ang_euler = Vector3DF(ax,ay,az);
	to_pos.x = from_pos.x - (float) (cos ( ang_euler.y * DEGtoRAD ) * sin ( ang_euler.x * DEGtoRAD ) * mOrbitDist);
	to_pos.y = from_pos.y - (float) (sin ( ang_euler.y * DEGtoRAD ) * mOrbitDist);
	to_pos.z = from_pos.z - (float) (cos ( ang_euler.y * DEGtoRAD ) * cos ( ang_euler.x * DEGtoRAD ) * mOrbitDist);
	updateMatricies ();
}


void Camera3D::moveRelative ( float dx, float dy, float dz )
{
	Vector3DF vec ( dx, dy, dz );
	vec *= invrot_matrix;
	to_pos += vec;
	from_pos += vec;
	updateMatricies ();
}

void Camera3D::setProjection (eProjection proj_type)
{
	mProjType = proj_type;
}

void Camera3D::updateMatricies ()
{
	// THIS NEEDS TO HANDLE THE CASE WHEN THE OBJECT HAS ONLY BEEN CALLED BY setMatrices
	Matrix4F basis;
	Vector3DF temp;	
	
	if (orbit_set_) {
		// compute camera direction vectors	--- MATCHES OpenGL's gluLookAt function (DO NOT MODIFY)
		dir_vec = to_pos;					// f vector in gluLookAt docs						
		dir_vec -= from_pos;				// eye = from_pos in gluLookAt docs
		dir_vec.Normalize();
		side_vec = dir_vec;
		side_vec.Cross(up_dir);
		side_vec.Normalize();
		up_vec = side_vec;
		up_vec.Cross(dir_vec);
		up_vec.Normalize();
		dir_vec *= -1;

		// construct view matrix
		rotate_matrix.Basis(side_vec, up_vec, dir_vec);
		view_matrix = rotate_matrix;
		view_matrix.PreTranslate(Vector3DF(-from_pos.x, -from_pos.y, -from_pos.z));

		// construct projection matrix  --- MATCHES OpenGL's gluPerspective function (DO NOT MODIFY)
		float sx = (float)tan(mFov * DEGtoRAD / 2.0f) * mNear;
		float sy = sx / mAspect;
		proj_matrix = 0.0f;
		proj_matrix(0, 0) = 2.0f*mNear / sx;				// matches OpenGL definition
		proj_matrix(1, 1) = 2.0f*mNear / sy;
		proj_matrix(2, 2) = -(mFar + mNear) / (mFar - mNear);			// C
		proj_matrix(2, 3) = -(2.0f*mFar * mNear) / (mFar - mNear);		// D
		proj_matrix(3, 2) = -1.0f;
	}

	tileproj_matrix = proj_matrix;

	// construct inverse rotate and inverse projection matrix
	Vector3DF tvz(0, 0, 0);
	invrot_matrix.InverseView ( rotate_matrix.GetDataF(), tvz );		// Computed using rule: "Inverse of a basis rotation matrix is its transpose." (So long as translation is taken out)
	invproj_matrix.InverseProj ( tileproj_matrix.GetDataF() );		

	Matrix4F view_matrix_notranslation = view_matrix;
	view_matrix_notranslation(12) = 0.0f;
	view_matrix_notranslation(13) = 0.0f;
	view_matrix_notranslation(14) = 0.0f;

	invviewproj_matrix = tileproj_matrix;
	invviewproj_matrix *= view_matrix_notranslation;
	invviewproj_matrix.InvertTRS();

	origRayWorld = from_pos;
	updateFrustum();
}

void Camera3D::setModelMatrix ( float* mtx )
{
	memcpy ( model_matrix.GetDataF(), mtx, sizeof(float)*16 );
}

void Camera3D::setMatrices(const float* view_mtx, const float* proj_mtx, Vector3DF model_pos )
{
	// Assign the matrices we have
	//  p = Tc V P M p		
	view_matrix = Matrix4F(view_mtx);
	proj_matrix = Matrix4F(proj_mtx);
	tileproj_matrix = proj_matrix;

	// From position
	Matrix4F tmp( view_mtx );
	tmp.InvertTRS ();
	Vector3DF from ( tmp(0,3), tmp(1,3), tmp(2,3) );
	from_pos = from;

	// Construct inverse matrices
    Vector3DF zero(0, 0, 0);
	invrot_matrix.InverseView ( view_matrix.GetDataF(), zero );		// Computed using rule: "Inverse of a basis rotation matrix is its transpose." (So long as translation is taken out)
	invproj_matrix.InverseProj ( tileproj_matrix.GetDataF() );		

	Matrix4F view_matrix_notranslation = view_matrix;
	view_matrix_notranslation(12) = 0.0f;
	view_matrix_notranslation(13) = 0.0f;
	view_matrix_notranslation(14) = 0.0f;

	invviewproj_matrix = tileproj_matrix;					// Used for GVDB raytracing
	invviewproj_matrix *= view_matrix_notranslation;			
	invviewproj_matrix.InvertTRS();			

	// Compute mFov, mAspect, mNear, mFar
	mNear = proj_matrix(2, 3) / (proj_matrix(2, 2) - 1.0f);
	mFar  = proj_matrix(2, 3) / (proj_matrix(2, 2) + 1.0f);
	float sx = 2.0f * mNear / proj_matrix(0, 0);
	float sy = 2.0f * mNear / proj_matrix(1, 1);
	mAspect = sx / sy;
	mFov = 2.0f * atan(sx / mNear) / DEGtoRAD;

	origRayWorld = from - model_pos;
	updateFrustum();								// DO NOT call updateMatrices here. We have just set them.
}

void Camera3D::setViewMatrix ( float* mtx, float* invmtx )
{
	memcpy ( view_matrix.GetDataF(), mtx, sizeof(float)*16 );
	memcpy ( invrot_matrix.GetDataF(), invmtx, sizeof(float)*16 );
	Matrix4F tmp(mtx);
	tmp.InvertTRS();
	Vector3DF from(tmp(0, 3), tmp(1, 3), tmp(2, 3));
	from_pos = from;
	origRayWorld = from_pos; // Used by GVDB render
}
void Camera3D::setProjMatrix ( float* mtx, float* invmtx )
{
	memcpy ( proj_matrix.GetDataF(), mtx, sizeof(float)*16 );
	memcpy ( tileproj_matrix.GetDataF(), mtx, sizeof(float)*16 );
	memcpy ( invproj_matrix.GetDataF(), invmtx, sizeof(float)*16 );
}

void Camera3D::updateFrustum ()
{
	Matrix4F mv;
	mv = tileproj_matrix;					// Compute the model-view-projection matrix
	mv *= view_matrix;
	float* mvm = mv.GetDataF();
	float t;

	// Right plane
   frustum[0][0] = mvm[ 3] - mvm[ 0];
   frustum[0][1] = mvm[ 7] - mvm[ 4];
   frustum[0][2] = mvm[11] - mvm[ 8];
   frustum[0][3] = mvm[15] - mvm[12];
   t = sqrt( frustum[0][0] * frustum[0][0] + frustum[0][1] * frustum[0][1] + frustum[0][2] * frustum[0][2] );
   frustum[0][0] /= t; frustum[0][1] /= t; frustum[0][2] /= t; frustum[0][3] /= t;
	// Left plane
   frustum[1][0] = mvm[ 3] + mvm[ 0];
   frustum[1][1] = mvm[ 7] + mvm[ 4];
   frustum[1][2] = mvm[11] + mvm[ 8];
   frustum[1][3] = mvm[15] + mvm[12];
   t = sqrt( frustum[1][0] * frustum[1][0] + frustum[1][1] * frustum[1][1] + frustum[1][2]    * frustum[1][2] );
   frustum[1][0] /= t; frustum[1][1] /= t; frustum[1][2] /= t; frustum[1][3] /= t;
	// Bottom plane
   frustum[2][0] = mvm[ 3] + mvm[ 1];
   frustum[2][1] = mvm[ 7] + mvm[ 5];
   frustum[2][2] = mvm[11] + mvm[ 9];
   frustum[2][3] = mvm[15] + mvm[13];
   t = sqrt( frustum[2][0] * frustum[2][0] + frustum[2][1] * frustum[2][1] + frustum[2][2]    * frustum[2][2] );
   frustum[2][0] /= t; frustum[2][1] /= t; frustum[2][2] /= t; frustum[2][3] /= t;
	// Top plane
   frustum[3][0] = mvm[ 3] - mvm[ 1];
   frustum[3][1] = mvm[ 7] - mvm[ 5];
   frustum[3][2] = mvm[11] - mvm[ 9];
   frustum[3][3] = mvm[15] - mvm[13];
   t = sqrt( frustum[3][0] * frustum[3][0] + frustum[3][1] * frustum[3][1] + frustum[3][2]    * frustum[3][2] );
   frustum[3][0] /= t; frustum[3][1] /= t; frustum[3][2] /= t; frustum[3][3] /= t;
	// Far plane
   frustum[4][0] = mvm[ 3] - mvm[ 2];
   frustum[4][1] = mvm[ 7] - mvm[ 6];
   frustum[4][2] = mvm[11] - mvm[10];
   frustum[4][3] = mvm[15] - mvm[14];
   t = sqrt( frustum[4][0] * frustum[4][0] + frustum[4][1] * frustum[4][1] + frustum[4][2]    * frustum[4][2] );
   frustum[4][0] /= t; frustum[4][1] /= t; frustum[4][2] /= t; frustum[4][3] /= t;
	// Near plane
   frustum[5][0] = mvm[ 3] + mvm[ 2];
   frustum[5][1] = mvm[ 7] + mvm[ 6];
   frustum[5][2] = mvm[11] + mvm[10];
   frustum[5][3] = mvm[15] + mvm[14];
   t = sqrt( frustum[5][0] * frustum[5][0] + frustum[5][1] * frustum[5][1] + frustum[5][2]    * frustum[5][2] );
   frustum[5][0] /= t; frustum[5][1] /= t; frustum[5][2] /= t; frustum[5][3] /= t;

   tlRayWorld = inverseRayProj(-1.0f,  1.0f, mNear );
   trRayWorld = inverseRayProj(1.0f, 1.0f, mNear );
   blRayWorld = inverseRayProj(-1.0f, -1.0f, mNear );
   brRayWorld = inverseRayProj(1.0f, -1.0f, mNear );
}

float Camera3D::calculateLOD ( Vector3DF pnt, float minlod, float maxlod, float maxdist )
{
	Vector3DF vec = pnt;
	vec -= from_pos;
	float lod = minlod + ((float) vec.Length() * (maxlod-minlod) / maxdist );	
	lod = (lod < minlod) ? minlod : lod;
	lod = (lod > maxlod) ? maxlod : lod;
	return lod;
}

float Camera3D::getDu ()
{
	return (float) tan ( mFov * DEGtoRAD/2.0f ) * mNear;
}
float Camera3D::getDv ()
{
	return (float) tan ( mFov * DEGtoRAD/2.0f ) * mNear / mAspect;
}

Vector3DF Camera3D::getU ()
{
	return side_vec;
}
Vector3DF Camera3D::getV ()
{
	return up_vec;
}
Vector3DF Camera3D::getW ()
{
	return dir_vec;
}


Vector3DF Camera3D::inverseRayProj(float x, float y, float z)
{
	Vector4DF p(x, y, z, 1.0f);

	Vector4DF wp(0.0f, 0.0f, 0.0f, 0.0f);
	wp.x = invviewproj_matrix.data[0] * p.x + invviewproj_matrix.data[4] * p.y + invviewproj_matrix.data[8] * p.z + invviewproj_matrix.data[12];
	wp.y = invviewproj_matrix.data[1] * p.x + invviewproj_matrix.data[5] * p.y + invviewproj_matrix.data[9] * p.z + invviewproj_matrix.data[13];
	wp.z = invviewproj_matrix.data[2] * p.x + invviewproj_matrix.data[6] * p.y + invviewproj_matrix.data[10] * p.z + invviewproj_matrix.data[14];
	wp.w = invviewproj_matrix.data[3] * p.x + invviewproj_matrix.data[7] * p.y + invviewproj_matrix.data[11] * p.z + invviewproj_matrix.data[15];

	return Vector3DF(wp.x / wp.w, wp.y / wp.w, wp.z / wp.w);
}

Vector3DF Camera3D::inverseRay (float x, float y, float z)
{	
	float sx = (float) tan ( mFov * DEGtoRAD/2.0f);
	float sy = sx / mAspect;
	float tu, tv;
	tu = mTile.x + x * (mTile.z-mTile.x);
	tv = mTile.y + y * (mTile.w-mTile.y);
	Vector4DF pnt ( (tu*2.0f-1.0f) * z*sx, (1.0f-tv*2.0f) * z*sy, -z, 1 );
	pnt *= invrot_matrix;
	return pnt;
}

Vector4DF Camera3D::project ( Vector3DF& p, Matrix4F& vm )
{
	Vector4DF q = p;								// World coordinates
	
	q *= vm;										// Eye coordinates
	
	q *= proj_matrix;								// Projection 

	q /= q.w;										// Normalized Device Coordinates (w-divide)
	
	q.x *= 0.5f;
	q.y *= -0.5f;
	q.z = q.z*0.5f + 0.5f;							// Stored depth buffer value
		
	return q;
}

Vector4DF Camera3D::project ( Vector3DF& p )
{
	Vector4DF q = p;								// World coordinates
	q *= view_matrix;								// Eye coordinates

	q *= proj_matrix;								// Clip coordinates
	
	q /= q.w;										// Normalized Device Coordinates (w-divide)

	q.x *= 0.5f;
	q.y *= -0.5f;
	q.z = q.z*0.5f + 0.5f;							// Stored depth buffer value
		
	return q;
}

void PivotX::setPivot ( float x, float y, float z, float rx, float ry, float rz )
{
	from_pos.Set ( x,y,z);
	ang_euler.Set ( rx,ry,rz );
}

void PivotX::updateTform ()
{
	trans.RotateZYXT ( ang_euler, from_pos );
}


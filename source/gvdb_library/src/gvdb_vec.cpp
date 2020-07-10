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

#include "gvdb_vec.h"
using namespace nvdb;

Vector4DF &Vector4DF::operator*= (const Matrix4F &op)
{
	float xa, ya, za, wa;
	xa = x * op.data[0] + y * op.data[4] + z * op.data[8] + w * op.data[12];
	ya = x * op.data[1] + y * op.data[5] + z * op.data[9] + w * op.data[13];
	za = x * op.data[2] + y * op.data[6] + z * op.data[10] + w * op.data[14];
	wa = x * op.data[3] + y * op.data[7] + z * op.data[11] + w * op.data[15];
	x = xa; y = ya; z = za; w = wa;
	return *this;
}

Vector4DF &Vector4DF::operator*= (const float* op)
{
	float xa, ya, za, wa;
	xa = x * op[0] + y * op[4] + z * op[8] + w * op[12];
	ya = x * op[1] + y * op[5] + z * op[9] + w * op[13];
	za = x * op[2] + y * op[6] + z * op[10] + w * op[14];
	wa = x * op[3] + y * op[7] + z * op[11] + w * op[15];
	x = xa; y = ya; z = za; w = wa;
	return *this;
}

// Vector4DF Code Definition

#define VNAME		4DF
#define VTYPE		float

// Constructors/Destructors
Vector4DF::Vector4DF (const VTYPE xa, const VTYPE ya, const VTYPE za, const VTYPE wa) {x=xa; y=ya; z=za; w=wa;}

Vector4DF::Vector4DF (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 0;}
Vector4DF::Vector4DF (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 0;}
Vector4DF::Vector4DF (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) op.w;}

// Member Functions
Vector4DF &Vector4DF::operator= (const int op) {x= (VTYPE) op; y= (VTYPE) op; z= (VTYPE) op; w = (VTYPE) op; return *this;}
Vector4DF &Vector4DF::operator= (const double op) {x= (VTYPE) op; y= (VTYPE) op; z= (VTYPE) op; w = (VTYPE) op; return *this;}

Vector4DF &Vector4DF::operator= (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 0;  return *this;}
Vector4DF &Vector4DF::operator= (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 0; return *this;}
Vector4DF &Vector4DF::operator= (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) op.w; return *this;}	
	
Vector4DF &Vector4DF::operator+= (const int op) {x+= (VTYPE) op; y+= (VTYPE) op; z+= (VTYPE) op; w += (VTYPE) op; return *this;}
Vector4DF &Vector4DF::operator+= (const double op) {x+= (VTYPE) op; y+= (VTYPE) op; z+= (VTYPE) op; w += (VTYPE) op; return *this;}
Vector4DF &Vector4DF::operator+= (const Vector3DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
Vector4DF &Vector4DF::operator+= (const Vector3DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
Vector4DF &Vector4DF::operator+= (const Vector4DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; w+=(VTYPE) op.w; return *this;}	

Vector4DF &Vector4DF::operator-= (const int op) {x-= (VTYPE) op; y-= (VTYPE) op; z-= (VTYPE) op; w -= (VTYPE) op; return *this;}
Vector4DF &Vector4DF::operator-= (const double op) {x-= (VTYPE) op; y-= (VTYPE) op; z-= (VTYPE) op; w -= (VTYPE) op; return *this;}
Vector4DF &Vector4DF::operator-= (const Vector3DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
Vector4DF &Vector4DF::operator-= (const Vector3DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
Vector4DF &Vector4DF::operator-= (const Vector4DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; w-=(VTYPE) op.w; return *this;}	

Vector4DF &Vector4DF::operator*= (const int op) {x*= (VTYPE) op; y*= (VTYPE) op; z*= (VTYPE) op; w *= (VTYPE) op; return *this;}
Vector4DF &Vector4DF::operator*= (const double op) {x*= (VTYPE) op; y*= (VTYPE) op; z*= (VTYPE) op; w *= (VTYPE) op; return *this;}
Vector4DF &Vector4DF::operator*= (const Vector3DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
Vector4DF &Vector4DF::operator*= (const Vector3DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
Vector4DF &Vector4DF::operator*= (const Vector4DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; w*=(VTYPE) op.w; return *this;}	

Vector4DF &Vector4DF::operator/= (const int op) {x/= (VTYPE) op; y/= (VTYPE) op; z/= (VTYPE) op; w /= (VTYPE) op; return *this;}
Vector4DF &Vector4DF::operator/= (const double op) {x/= (VTYPE) op; y/= (VTYPE) op; z/= (VTYPE) op; w /= (VTYPE) op; return *this;}
Vector4DF &Vector4DF::operator/= (const Vector3DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
Vector4DF &Vector4DF::operator/= (const Vector3DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
Vector4DF &Vector4DF::operator/= (const Vector4DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; w/=(VTYPE) op.w; return *this;}	

Vector4DF &Vector4DF::Cross (const Vector4DF &v) {double ax = x, ay = y, az = z, aw = w; x = (VTYPE) (ay * (double) v.z - az * (double) v.y); y = (VTYPE) (-ax * (double) v.z + az * (double) v.x); z = (VTYPE) (ax * (double) v.y - ay * (double) v.x); w = (VTYPE) 0; return *this;}
		
double Vector4DF::Dot(const Vector4DF &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z + (double) w*v.w; return dot;}

double Vector4DF::Dist (const Vector4DF &v)		{double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}

double Vector4DF::DistSq (const Vector4DF &v)		{double a,b,c,d; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; d = (double) w - (double) v.w; return (a*a + b*b + c*c + d*d);}

Vector4DF &Vector4DF::Normalize (void) {
	double n = (double) x*x + (double) y*y + (double) z*z + (double) w*w;
	if (n!=0.0) {
		n = sqrt(n);
		x /= (float) n; y /= (float) n; z /= (float) n; w /= (float) n;
	}
	return *this;
}
double Vector4DF::Length (void) { double n; n = (double) x*x + (double) y*y + (double) z*z + (double) w*w; if (n != 0.0) return sqrt(n); return 0.0; }

#undef VTYPE
#undef VNAME


// Matrix4F Code Definition
#undef VTYPE
#define VNAME		F
#define VTYPE		float

// Constructors/Destructors

Matrix4F::Matrix4F ( const float* src )	{ for (int n=0; n < 16; n++) data[n] = src[n]; }
Matrix4F::Matrix4F ( float f0, float f1, float f2, float f3, 
							float f4, float f5, float f6, float f7, 
							float f8, float f9, float f10, float f11,
							float f12, float f13, float f14, float f15 )
{
	data[0] = f0;	data[1] = f1;	data[2] = f2;	data[3] = f3;
	data[4] = f4;	data[5] = f5;	data[6] = f6;	data[7] = f7;
	data[8] = f8;	data[9] = f9;	data[10] = f10;	data[11] = f11;
	data[12] = f12;	data[13] = f13;	data[14] = f14;	data[15] = f15;
}

Matrix4F Matrix4F::operator* (const float &op)
{
	return Matrix4F ( data[0]*op,	data[1]*op, data[2]*op, data[3],
					  data[4]*op,	data[5]*op, data[6]*op,	data[7],
					  data[8]*op,	data[9]*op, data[10]*op, data[11],
					  data[12],		data[13],	data[14],	data[15] );
}

Vector3DF Matrix4F::operator* (const Vector3DF& op) {
	Vector3DF result;
	result.x = data[0] * op.x + data[4] * op.y + data[ 8] * op.z + data[12];
	result.y = data[1] * op.x + data[5] * op.y + data[ 9] * op.z + data[13];
	result.z = data[2] * op.x + data[6] * op.y + data[10] * op.z + data[14];
	return result;
}

Matrix4F &Matrix4F::operator= (const unsigned char op)	{for ( int n=0; n<16; n++) data[n] = (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator= (const int op)				{for ( int n=0; n<16; n++) data[n] = (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator= (const double op)			{for ( int n=0; n<16; n++) data[n] = (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator+= (const unsigned char op)	{for ( int n=0; n<16; n++) data[n] += (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator+= (const int op)				{for ( int n=0; n<16; n++) data[n] += (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator+= (const double op)			{for ( int n=0; n<16; n++) data[n] += (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator-= (const unsigned char op)	{for ( int n=0; n<16; n++) data[n] -= (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator-= (const int op)				{for ( int n=0; n<16; n++) data[n] -= (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator-= (const double op)			{for ( int n=0; n<16; n++) data[n] -= (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator*= (const unsigned char op)	{for ( int n=0; n<16; n++) data[n] *= (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator*= (const int op)				{for ( int n=0; n<16; n++) data[n] *= (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator*= (const double op)			{for ( int n=0; n<16; n++) data[n] *= (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator/= (const unsigned char op)	{for ( int n=0; n<16; n++) data[n] /= (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator/= (const int op)				{for ( int n=0; n<16; n++) data[n] /= (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator/= (const double op)			{for ( int n=0; n<16; n++) data[n] /= (VTYPE) op; return *this;}

// column-major multiply (like OpenGL)
Matrix4F &Matrix4F::operator*= (const Matrix4F &op) {
	return (*this *= op.GetDataF());
}

Matrix4F &Matrix4F::operator= (const float* op) 
{
	for (int n=0; n < 16; n++ )
		data[n] = op[n];
	return *this;
}

Matrix4F &Matrix4F::operator*= (const float* op) {
	register float orig[16];				// Temporary storage
	memcpy ( orig, data, 16*sizeof(float) );

	// Calculate First Row
	data[0] = op[0]*orig[0] + op[1]*orig[4] + op[2]*orig[8] + op[3]*orig[12];
	data[1] = op[0]*orig[1] + op[1]*orig[5] + op[2]*orig[9] + op[3]*orig[13];
	data[2] = op[0]*orig[2] + op[1]*orig[6] + op[2]*orig[10] + op[3]*orig[14];
	data[3] = op[0]*orig[3] + op[1]*orig[7] + op[2]*orig[11] + op[3]*orig[15];

	// Calculate Second Row
	data[4] = op[4]*orig[0] + op[5]*orig[4] + op[6]*orig[8] + op[7]*orig[12];
	data[5] = op[4]*orig[1] + op[5]*orig[5] + op[6]*orig[9] + op[7]*orig[13];
	data[6] = op[4]*orig[2] + op[5]*orig[6] + op[6]*orig[10] + op[7]*orig[14];
	data[7] = op[4]*orig[3] + op[5]*orig[7] + op[6]*orig[11] + op[7]*orig[15];
	
	// Calculate Third Row
	data[8] = op[8]*orig[0] + op[9]*orig[4] + op[10]*orig[8] + op[11]*orig[12];
	data[9] = op[8]*orig[1] + op[9]*orig[5] + op[10]*orig[9] + op[11]*orig[13];
	data[10] = op[8]*orig[2] + op[9]*orig[6] + op[10]*orig[10] + op[11]*orig[14];
	data[11] = op[8]*orig[3] + op[9]*orig[7] + op[10]*orig[11] + op[11]*orig[15];

	// Calculate Four Row
	data[12] = op[12]*orig[0] + op[13]*orig[4] + op[14]*orig[8] + op[15]*orig[12];
	data[13] = op[12]*orig[1] + op[13]*orig[5] + op[14]*orig[9] + op[15]*orig[13];
	data[14] = op[12]*orig[2] + op[13]*orig[6] + op[14]*orig[10] + op[15]*orig[14];
	data[15] = op[12]*orig[3] + op[13]*orig[7] + op[14]*orig[11] + op[15]*orig[15];

	return *this;
}


Matrix4F &Matrix4F::Transpose (void)
{
	register float orig[16];				// Temporary storage
	memcpy ( orig, data, 16*sizeof(VTYPE) );
	
	data[0] = orig[0];	data[1] = orig[4];	data[2] = orig[8];	data[3] = orig[12];
	data[4] = orig[1];	data[5] = orig[5];	data[6] = orig[9];	data[7] = orig[13];
	data[8] = orig[2];	data[9] = orig[6];	data[10] = orig[10];data[11] = orig[14];
	data[12] = orig[3];	data[13] = orig[7];	data[14] = orig[11];data[15] = orig[15];
	return *this;	
}

Matrix4F &Matrix4F::Identity ()
{
	memset (data, 0, 16*sizeof(VTYPE));	
	data[0] = 1.0;
	data[5] = 1.0;
	data[10] = 1.0;
	data[15] = 1.0;	
	return *this;
}

Matrix4F &Matrix4F::RotateZYX (const Vector3DF& angs)
{
	float cx = cosf(angs.x * DEGtoRAD);
	float sx = sinf(angs.x * DEGtoRAD);
	float cy = cosf(angs.y * DEGtoRAD);
	float sy = sinf(angs.y * DEGtoRAD);
	float cz = cosf(angs.z * DEGtoRAD);
	float sz = sinf(angs.z * DEGtoRAD);
	// Rotation matrices in right-handed coordinate system
	// See also Diebel 2006, "Representing Attitude", equation 67
	// (note that matrices are transposed from Diebel)
	// [cz -sz 0][ cy 0 sy][1  0   0]
	// [sz  cz 0][  0 1  0][0 cx -sx]
	// [ 0   0 1][-sy 0 cy][0 sx  cx]
	data[0] = cy * cz;
	data[1] = cy * sz;
	data[2] = -sy;
	data[3] = 0;
	data[4] = cz * sx * sy - cx * sz;
	data[5] = cx * cz + sx * sy * sz;
	data[6] = cy * sx;
	data[7] = 0;
	data[8] = cx * cz * sy + sx * sz;
	data[9] = -cz * sx + cx * sy * sz;
	data[10] = cx * cy;
	data[11] = 0;
	data[11] = 0;
	data[12] = 0;
	data[13] = 0;
	data[14] = 0;
	data[15] = 1;
	return *this;
}
Matrix4F &Matrix4F::RotateZYXT (const Vector3DF& angs, const Vector3DF& t)
{
	// Compute the matrix Rz Ry Rx T. We can do this using RotateZYX and TranslateInPlace:
	RotateZYX(angs);
	return TranslateInPlace(t);
}
Matrix4F &Matrix4F::RotateTZYX (const Vector3DF& angs, const Vector3DF& t)
{	
	// Compute the matrix T Rz Ry Rx. We can do this using RotateZYX, using
	// [1 0 0 tx][r0 r4  r8 0]   [r0 r4  r8 tx]
	// [0 1 0 ty][r1 r5  r9 0] = [r1 r5  r9 ty]
	// [0 0 1 tz][r2 r6 r10 0]   [r2 r6 r10 tz]
	// [0 0 0  1][ 0  0   0 1]   [ 0  0   0  1]
	RotateZYX(angs);
	data[12] = t.x;
	data[13] = t.y;
	data[14] = t.z;
	return *this;
}
Matrix4F &Matrix4F::RotateTZYXS (const Vector3DF& angs, const Vector3DF& t, const Vector3DF& s)
{
	// Compute the matrix T Rz Ry Rx S.
	RotateTZYX(angs, t);
	// Multiply on the right by diag(s.x, s.y, s.z, 1).
	for (int i = 0; i < 3; i++) {
		data[i + 0] *= s.x;
		data[i + 4] *= s.y;
		data[i + 8] *= s.z;
	}
	return *this;
}

Matrix4F &Matrix4F::RotateX (const double ang)
{
	// [1  0  0]
	// [0  c -s]
	// [0  s  c]
	memset(data, 0, 16 * sizeof(VTYPE));
	float c = static_cast<float>(cos(ang * DEGtoRAD));
	float s = static_cast<float>(sin(ang * DEGtoRAD));
	data[0] = 1;
	data[5] = c;	data[9] = -s;
	data[6] = s;	data[10] = c;
	data[15] = 1;
	return *this;
}

Matrix4F &Matrix4F::RotateY (const double ang)
{
	// [ c 0 s]
	// [ 0 1 0]
	// [-s 0 c]
	memset(data, 0, 16 * sizeof(VTYPE));
	float c = static_cast<float>(cos(ang * DEGtoRAD));
	float s = static_cast<float>(sin(ang * DEGtoRAD));
	data[0] = c;	data[8] = s;
	data[5] = 1;
	data[2] = -s;	data[10] = c;
	data[15] = 1;
	return *this;
}

Matrix4F &Matrix4F::RotateZ (const double ang)
{
	// [c -s 0]
	// [s  c 0]
	// [0  0 1]
	memset(data, 0, 16 * sizeof(VTYPE));
	float c = static_cast<float>(cos(ang * DEGtoRAD));
	float s = static_cast<float>(sin(ang * DEGtoRAD));
	data[0] = c;	data[4] = -s;
	data[1] = s;	data[5] = c;
	data[10] = 1; 
	data[15] = 1;
	return *this;
}

Matrix4F &Matrix4F::Ortho (double sx, double sy, double vn, double vf)
{
	// simplified version of OpenGL's glOrtho function
	data[ 0] = (VTYPE)(1.0 / sx); data[ 1] = (VTYPE)0.0;		data[ 2] = (VTYPE)0.0;						data[ 3] = (VTYPE)0.0;
	data[ 4] = (VTYPE)0.0;		  data[ 5] = (VTYPE)(1.0 / sy); data[ 6] = (VTYPE)0.0;						data[ 7] = (VTYPE)0.0;
	data[ 8] = (VTYPE)0.0;		  data[ 9] = (VTYPE)0.0;		data[10] = (VTYPE)(-2.0 / (vf - vn));		data[11] = (VTYPE)0.0;
	data[12] = (VTYPE)0.0;		  data[13] = (VTYPE)0.0;		data[14] = (VTYPE)(-(vf + vn) / (vf - vn));	data[15] = (VTYPE)1.0;
	return *this;
}

Matrix4F &Matrix4F::Translate (double tx, double ty, double tz)
{
	data[ 0] = (VTYPE) 1.0; data[ 1] = (VTYPE) 0.0;	data[ 2] = (VTYPE) 0.0; data[ 3] = (VTYPE) 0.0;
	data[ 4] = (VTYPE) 0.0; data[ 5] = (VTYPE) 1.0; data[ 6] = (VTYPE) 0.0; data[ 7] = (VTYPE) 0.0;
	data[ 8] = (VTYPE) 0.0; data[ 9] = (VTYPE) 0.0; data[10] = (VTYPE) 1.0; data[11] = (VTYPE) 0.0;
	data[12] = (VTYPE) tx;	data[13] = (VTYPE) ty;	data[14] = (VTYPE) tz;	data[15] = (VTYPE) 1.0;	
	return *this;
}

Matrix4F &Matrix4F::Scale (double sx, double sy, double sz)
{
	data[ 0] = (VTYPE) sx;  data[ 1] = (VTYPE) 0.0;	data[ 2] = (VTYPE) 0.0; data[ 3] = (VTYPE) 0.0;
	data[ 4] = (VTYPE) 0.0; data[ 5] = (VTYPE) sy;  data[ 6] = (VTYPE) 0.0; data[ 7] = (VTYPE) 0.0;
	data[ 8] = (VTYPE) 0.0; data[ 9] = (VTYPE) 0.0; data[10] = (VTYPE) sz;  data[11] = (VTYPE) 0.0;
	data[12] = (VTYPE) 0.0;	data[13] = (VTYPE) 0.0;	data[14] = (VTYPE) 0.0;	data[15] = (VTYPE) 1.0;	
	return *this;
}

Matrix4F &Matrix4F::Basis (const Vector3DF &norm)
{
	Vector3DF binorm, tang;
	binorm.Set ( 0.0, 1.0, 0 );		// up vector
	binorm.Cross ( norm );	
	binorm.Normalize ();
	tang = binorm;
	tang.Cross ( norm );	
	//tang *= -1;
	tang.Normalize ();
	
	data[ 0] = (VTYPE) binorm.x; data[ 1] = (VTYPE) binorm.y; data[ 2] = (VTYPE) binorm.z; data[ 3] = (VTYPE) 0.0;
	data[ 4] = (VTYPE) norm.x; data[ 5] = (VTYPE) norm.y; data[ 6] = (VTYPE) norm.z; data[ 7] = (VTYPE) 0.0;
	data[ 8] = (VTYPE) tang.x; data[ 9] = (VTYPE) tang.y; data[10] = (VTYPE) tang.z; data[11] = (VTYPE) 0.0;
	data[12] = (VTYPE) 0.0;	 data[13] = (VTYPE) 0.0;  data[14] = (VTYPE) 0.0;  data[15] = (VTYPE) 1.0;	
	return *this;
	
}

Matrix4F &Matrix4F::Basis (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3)
{
	data[ 0] = (VTYPE) c1.x; data[ 1] = (VTYPE) c2.x; data[ 2] = (VTYPE) c3.x; data[ 3] = (VTYPE) 0.0;
	data[ 4] = (VTYPE) c1.y; data[ 5] = (VTYPE) c2.y; data[ 6] = (VTYPE) c3.y; data[ 7] = (VTYPE) 0.0;
	data[ 8] = (VTYPE) c1.z; data[ 9] = (VTYPE) c2.z; data[10] = (VTYPE) c3.z; data[11] = (VTYPE) 0.0;
	data[12] = (VTYPE)  0.0; data[13] = (VTYPE)  0.0; data[14] = (VTYPE)  0.0; data[15] = (VTYPE) 1.0;
	return *this;
}
Matrix4F &Matrix4F::TransSRT (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const Vector3DF& s)
{
	data[ 0] = (VTYPE) c1.x*s.x; data[ 4] = (VTYPE) c2.x*s.x; data[ 8] = (VTYPE) c3.x*s.x;  data[12] = (VTYPE) 0.0;
	data[ 1] = (VTYPE) c1.y*s.y; data[ 5] = (VTYPE) c2.y*s.y; data[ 9] = (VTYPE) c3.y*s.y;  data[13] = (VTYPE) 0.0;
	data[ 2] = (VTYPE) c1.z*s.z; data[ 6] = (VTYPE) c2.z*s.z; data[10] = (VTYPE) c3.z*s.z;  data[14] = (VTYPE) 0.0;
	data[ 3] = (VTYPE) t.x;		 data[ 7] = (VTYPE) t.y;	  data[11] = (VTYPE) t.z;		data[15] = (VTYPE) 1.0;	
	return *this;
}

Matrix4F &Matrix4F::SRT (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const Vector3DF& s)
{
	data[ 0] = (VTYPE) c1.x*s.x; data[ 1] = (VTYPE) c2.x*s.x; data[ 2] = (VTYPE) c3.x*s.x;  data[ 3] = (VTYPE) 0.0;
	data[ 4] = (VTYPE) c1.y*s.y; data[ 5] = (VTYPE) c2.y*s.y; data[ 6] = (VTYPE) c3.y*s.y;  data[ 7] = (VTYPE) 0.0;
	data[ 8] = (VTYPE) c1.z*s.z; data[ 9] = (VTYPE) c2.z*s.z; data[10] = (VTYPE) c3.z*s.z;  data[11] = (VTYPE) 0.0;
	data[12] = (VTYPE) t.x;		 data[13] = (VTYPE) t.y;	  data[14] = (VTYPE) t.z;		data[15] = (VTYPE) 1.0;	
	return *this;
}
Matrix4F &Matrix4F::SRT (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const float s)
{
	data[ 0] = (VTYPE) c1.x*s; data[ 1] = (VTYPE) c1.y*s; data[ 2] = (VTYPE) c1.z*s;  data[ 3] = (VTYPE) 0.0;
	data[ 4] = (VTYPE) c2.x*s; data[ 5] = (VTYPE) c2.y*s; data[ 6] = (VTYPE) c2.z*s;  data[ 7] = (VTYPE) 0.0;
	data[ 8] = (VTYPE) c3.x*s; data[ 9] = (VTYPE) c3.y*s; data[10] = (VTYPE) c3.z*s;  data[11] = (VTYPE) 0.0;
	data[12] = (VTYPE) t.x;		 data[13] = (VTYPE) t.y;	  data[14] = (VTYPE) t.z;		data[15] = (VTYPE) 1.0;	
	return *this;
}

Matrix4F &Matrix4F::InvTRS (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const Vector3DF& s)
{
	data[ 0] = (VTYPE) c1.x/s.x; data[ 1] = (VTYPE) c1.y/s.y; data[ 2] = (VTYPE) c1.z/s.z;  data[ 3] = (VTYPE) 0.0;
	data[ 4] = (VTYPE) c2.x/s.x; data[ 5] = (VTYPE) c2.y/s.y; data[ 6] = (VTYPE) c2.z/s.z;  data[ 7] = (VTYPE) 0.0;
	data[ 8] = (VTYPE) c3.x/s.x; data[ 9] = (VTYPE) c3.y/s.y; data[10] = (VTYPE) c3.z/s.z;  data[11] = (VTYPE) 0.0;
	data[12] = (VTYPE) -t.x/s.x; data[13] = (VTYPE) -t.y/s.y; data[14] = (VTYPE) -t.z/s.z;  data[15] = (VTYPE) 1.0;	
	return *this;
}

Matrix4F &Matrix4F::InvTRS (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const float s)
{
	data[ 0] = (VTYPE) c1.x/s; data[ 1] = (VTYPE) c1.y/s; data[ 2] = (VTYPE) c1.z/s;  data[ 3] = (VTYPE) 0.0;
	data[ 4] = (VTYPE) c2.x/s; data[ 5] = (VTYPE) c2.y/s; data[ 6] = (VTYPE) c2.z/s;  data[ 7] = (VTYPE) 0.0;
	data[ 8] = (VTYPE) c3.x/s; data[ 9] = (VTYPE) c3.y/s; data[10] = (VTYPE) c3.z/s;  data[11] = (VTYPE) 0.0;
	data[12] = (VTYPE) -t.x/s; data[13] = (VTYPE) -t.y/s; data[14] = (VTYPE) -t.z/s;  data[15] = (VTYPE) 1.0;	
	return *this;
}

Matrix4F &Matrix4F::InvertTRS ()
{
	// This function signature may look as if it includes optimizations for matrices composed of only translation,
	// rotation, and scale operations. However, it simply performs a general 4x4 matrix inverse.

	// Convert to double-precision
	double mat[16];
	for (int i = 0; i < 16; i++) {
		mat[i] = static_cast<double>(data[i]);
	}
	double inv[16];
	// multiplications: 16 * 12 + 4 + 16 = 212
	// additions: 16 * 5 + 3 = 83
	inv[0] = -mat[11]*mat[14]*mat[5] + mat[10]*mat[15]*mat[5] + mat[11]*mat[13]*mat[6] - mat[10]*mat[13]*mat[7] - mat[15]*mat[6]*mat[9] + mat[14]*mat[7]*mat[9];
	inv[4] = mat[11]*mat[14]*mat[4] - mat[10]*mat[15]*mat[4] - mat[11]*mat[12]*mat[6] + mat[10]*mat[12]*mat[7] + mat[15]*mat[6]*mat[8] - mat[14]*mat[7]*mat[8];
	inv[8] = -mat[11]*mat[13]*mat[4] + mat[11]*mat[12]*mat[5] - mat[15]*mat[5]*mat[8] + mat[13]*mat[7]*mat[8] + mat[15]*mat[4]*mat[9] - mat[12]*mat[7]*mat[9];
	inv[12] = mat[10]*mat[13]*mat[4] - mat[10]*mat[12]*mat[5] + mat[14]*mat[5]*mat[8] - mat[13]*mat[6]*mat[8] - mat[14]*mat[4]*mat[9] + mat[12]*mat[6]*mat[9];
	inv[1] = mat[1]*mat[11]*mat[14] - mat[1]*mat[10]*mat[15] - mat[11]*mat[13]*mat[2] + mat[10]*mat[13]*mat[3] + mat[15]*mat[2]*mat[9] - mat[14]*mat[3]*mat[9];
	inv[5] = -mat[0]*mat[11]*mat[14] + mat[0]*mat[10]*mat[15] + mat[11]*mat[12]*mat[2] - mat[10]*mat[12]*mat[3] - mat[15]*mat[2]*mat[8] + mat[14]*mat[3]*mat[8];
	inv[9] = -mat[1]*mat[11]*mat[12] + mat[0]*mat[11]*mat[13] + mat[1]*mat[15]*mat[8] - mat[13]*mat[3]*mat[8] - mat[0]*mat[15]*mat[9] + mat[12]*mat[3]*mat[9];
	inv[13] = mat[1]*mat[10]*mat[12] - mat[0]*mat[10]*mat[13] - mat[1]*mat[14]*mat[8] + mat[13]*mat[2]*mat[8] + mat[0]*mat[14]*mat[9] - mat[12]*mat[2]*mat[9];
	inv[2] = -mat[15]*mat[2]*mat[5] + mat[14]*mat[3]*mat[5] + mat[1]*mat[15]*mat[6] - mat[13]*mat[3]*mat[6] - mat[1]*mat[14]*mat[7] + mat[13]*mat[2]*mat[7];
	inv[6] = mat[15]*mat[2]*mat[4] - mat[14]*mat[3]*mat[4] - mat[0]*mat[15]*mat[6] + mat[12]*mat[3]*mat[6] + mat[0]*mat[14]*mat[7] - mat[12]*mat[2]*mat[7];
	inv[10] = -mat[1]*mat[15]*mat[4] + mat[13]*mat[3]*mat[4] + mat[0]*mat[15]*mat[5] - mat[12]*mat[3]*mat[5] + mat[1]*mat[12]*mat[7] - mat[0]*mat[13]*mat[7];
	inv[14] = mat[1]*mat[14]*mat[4] - mat[13]*mat[2]*mat[4] - mat[0]*mat[14]*mat[5] + mat[12]*mat[2]*mat[5] - mat[1]*mat[12]*mat[6] + mat[0]*mat[13]*mat[6];
	inv[3] = mat[11]*mat[2]*mat[5] - mat[10]*mat[3]*mat[5] - mat[1]*mat[11]*mat[6] + mat[1]*mat[10]*mat[7] + mat[3]*mat[6]*mat[9] - mat[2]*mat[7]*mat[9];
	inv[7] = -mat[11]*mat[2]*mat[4] + mat[10]*mat[3]*mat[4] + mat[0]*mat[11]*mat[6] - mat[0]*mat[10]*mat[7] - mat[3]*mat[6]*mat[8] + mat[2]*mat[7]*mat[8];
	inv[11] = mat[1]*mat[11]*mat[4] - mat[0]*mat[11]*mat[5] + mat[3]*mat[5]*mat[8] - mat[1]*mat[7]*mat[8] - mat[3]*mat[4]*mat[9] + mat[0]*mat[7]*mat[9];
	inv[15] = -mat[1]*mat[10]*mat[4] + mat[0]*mat[10]*mat[5] - mat[2]*mat[5]*mat[8] + mat[1]*mat[6]*mat[8] + mat[2]*mat[4]*mat[9] - mat[0]*mat[6]*mat[9];
	
	double det = data[0]*inv[0] + data[1]*inv[4] + data[2]*inv[8] + data[3]*inv[12];
	if (det == 0)    return *this;
	det = 1.0f / det;

	for (int i = 0; i < 16; i++)  
		data[i] = (float) (inv[i] * det);
	
	return *this;
}

Matrix4F &Matrix4F::operator= ( float* mat )
{
	for (int n=0; n < 16; n++) 
		data[n] = mat[n];	
	return *this;
}

// Translate after (post-translate)
// Computes: M' = T*M
//
Matrix4F &Matrix4F::operator+= (const Vector3DF& t)
{	
	data[12] += (VTYPE) t.x;
	data[13] += (VTYPE) t.y;
	data[14] += (VTYPE) t.z;
	return *this;
}

// Translate first (pre-translate)
// Computes: M' = M*T
Matrix4F &Matrix4F::PreTranslate (const Vector3DF& t)
{	
	// Multiplies on the right as follows:
	// [d0 d4  d8 d12][1 0 0 tx]   [d0 d4  d8 (tx d0 + ty d4 + tz d8  + d12)]
	// [d1 d5  d9 d13][0 1 0 ty] = [d1 d5  d9 (tx d1 + ty d5 + tz d9  + d13)]
	// [d2 d6 d10 d14][0 0 1 tz]   [d2 d6 d10 (tx d2 + ty d6 + tz d10 + d14)]
	// [d3 d7 d11 d15][0 0 0  1]   [d3 d7 d11 (tx d3 + ty d7 + tz d11 + d15)]
	data[12] += (VTYPE) data[0]*t.x + data[4]*t.y + data[8]*t.z;
	data[13] += (VTYPE) data[1]*t.x + data[5]*t.y + data[9]*t.z;
	data[14] += (VTYPE) data[2]*t.x + data[6]*t.y + data[10]*t.z;
	return *this;
}

Matrix4F &Matrix4F::operator*= (const Vector3DF& s)		// quick scale
{	
	data[0] *= (VTYPE) s.x;	data[1] *= (VTYPE) s.y;	data[2] *= (VTYPE) s.z;	
	data[4] *= (VTYPE) s.x;	data[5] *= (VTYPE) s.y;	data[6] *= (VTYPE) s.z;	
	data[8] *= (VTYPE) s.x;	data[9] *= (VTYPE) s.y;	data[10] *= (VTYPE) s.z;
	data[12] *= (VTYPE) s.x; data[13] *= (VTYPE) s.y; data[14] *= (VTYPE) s.z;

	/*data[0] *= (VTYPE) t.x;	data[1] *= (VTYPE) t.x;	data[2] *= (VTYPE) t.x;	data[3] *= (VTYPE) t.x;
	data[4] *= (VTYPE) t.y;	data[5] *= (VTYPE) t.y;	data[6] *= (VTYPE) t.y;	data[7] *= (VTYPE) t.y;
	data[8] *= (VTYPE) t.z;	data[9] *= (VTYPE) t.z;	data[10] *= (VTYPE) t.z; data[11] *= (VTYPE) t.z;*/
	return *this;
}

Matrix4F &Matrix4F::InverseProj ( const float* mat )
{
	data[0] = 1.0f/mat[0];	data[1] = 0.0f;			data[2] = 0.0f;						data[3] = 0.0f;
	data[4] = 0.0f;			data[5] = 1.0f/mat[5];	data[6] = 0.0f;						data[7] = 0.0f;
	data[8] = 0.0f;			data[9] = 0.0f;			data[10] = 0.0f;					data[11] = 1.0f/mat[14];
	data[12] = mat[8]/mat[0];		data[13] = mat[9]/mat[5];		data[14] = -1.0f;	data[15] = mat[10]/mat[14];
	return *this;
}

Matrix4F &Matrix4F::InverseView ( const float* mat, const Vector3DF& pos)
{
	// NOTE: Assumes there is no scaling in input matrix.
	// Although there can be translation (typical of a view matrix)
	data[0] = mat[0];	data[1] = mat[4];	data[2] = mat[8];	data[3] = 0.0f;
	data[4] = mat[1];	data[5] = mat[5];	data[6] = mat[9];	data[7] = 0.0f;
	data[8] = mat[2];	data[9] = mat[6];	data[10] = mat[10];	data[11] = 0.0f;
	data[12] = pos.x;	data[13] = pos.y;	data[14] =  pos.z;	data[15] = 1.0f;
	return *this;
}

Vector4DF Matrix4F::GetT ( float* mat )
{
	return Vector4DF ( mat[12], mat[13], mat[14], 1.0 );
}

void Matrix4F::Print ()
{
	printf ( "%04.3f %04.3f %04.3f %04.3f\n", data[0], data[4], data[8], data[12] );
	printf ( "%04.3f %04.3f %04.3f %04.3f\n", data[1], data[5], data[9], data[13] );
	printf ( "%04.3f %04.3f %04.3f %04.3f\n", data[2], data[6], data[10], data[14] );
	printf ( "%04.3f %04.3f %04.3f %04.3f\n\n", data[3], data[7], data[11], data[15] );
}
#include <stdio.h>

#ifdef _WIN32
	#define	snprintf	sprintf_s
#endif

std::string Matrix4F::WriteToStr ()
{
	char buf[4096];
	std::string str;
	snprintf ( buf, 4096, "   %f %f %f %f\n", data[0], data[4], data[8], data[12] ); str = buf;
	snprintf ( buf, 4096, "   %f %f %f %f\n", data[1], data[5], data[9], data[13] ); str += buf;
	snprintf ( buf, 4096, "   %f %f %f %f\n", data[2], data[6], data[10], data[14] ); str += buf;
	snprintf ( buf, 4096, "   %f %f %f %f\n", data[3], data[7], data[11], data[15] ); str += buf;
	return str;
}

//------------------------------------------- Composing operations

Matrix4F& Matrix4F::TranslateInPlace(const Vector3DF& translation) {
	*this += translation;
	return *this;
}

Matrix4F& Matrix4F::LeftMultiplyInPlace(const Matrix4F& mtx) {
	// One can check that this compiles down to good code using something like Godbolt.
	Matrix4F rightMtx(data); // Make a copy of this
	// (AB)_(ik) = sum(A_(ij) B_(jk), j)
	for (int k = 0; k < 4; k++) {
		for (int i = 0; i < 4; i++) {
			data[i + 4 * k]
				= mtx(i, 0) * rightMtx(0, k)
				+ mtx(i, 1) * rightMtx(1, k)
				+ mtx(i, 2) * rightMtx(2, k)
				+ mtx(i, 3) * rightMtx(3, k);
		}
	}
	return *this;
}

Matrix4F& Matrix4F::ScaleInPlace(const Vector3DF& scale)
{
	// One can check that this compiles down to relatively good code using something like Godbolt.
	// For optimum compilation on GCC, this copy seems to be necessary - presumably because otherwise the
	// addresses of data and scale could overlap. But this isn't required on MSVC. In both cases, the compiler
	// unrolls this for loop.
	Vector3DF s = scale;
	for (int i = 0; i < 4; i++) {
		data[4*i] *= s.x;
		data[4*i + 1] *= s.y;
		data[4*i + 2] *= s.z;
	}
	return *this;
}

Matrix4F& Matrix4F::InvTranslateInPlace(const Vector3DF& translation) {
	return TranslateInPlace(translation * -1.0f);
}

Matrix4F& Matrix4F::InvLeftMultiplyInPlace(Matrix4F mtx) {
	// Invert the matrix
	mtx.InvertTRS();
	return LeftMultiplyInPlace(mtx);
}

Matrix4F& Matrix4F::InvScaleInPlace(const Vector3DF& scale) {
	return ScaleInPlace(Vector3DF(1.0f / scale.x, 1.0f / scale.y, 1.0f / scale.z));
}


#undef VTYPE
#undef VNAME
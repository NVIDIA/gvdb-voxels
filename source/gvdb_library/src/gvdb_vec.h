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


#ifndef GVDB_VEC
	#define GVDB_VEC

	#include "gvdb_types.h"
	#include <stdlib.h>

	namespace nvdb {		
		class Vector3DI;
		class Vector3DF;		
		class Vector4DF;			
		class Matrix4F;
	}

	// Vector3DI Declaration

	#define VNAME		3DI
	#define VTYPE		int

	namespace nvdb {

	class GVDB_API Vector3DI {
	public:
		VTYPE x, y, z;
	
		// Constructors/Destructors
		Vector3DI();
		Vector3DI (const VTYPE xa, const VTYPE ya, const VTYPE za);
		Vector3DI (const Vector3DI &op);
		Vector3DI (const Vector3DF &op);
		Vector3DI (const Vector4DF &op);

		// Set Functions
		Vector3DI &Set (const VTYPE xa, const VTYPE ya, const VTYPE za);

		// Member Functions		
		Vector3DI &operator= (const Vector3DI &op);
		Vector3DI &operator= (const Vector3DF &op);
		Vector3DI &operator= (const Vector4DF &op);
				
		Vector3DI &operator+= (const Vector3DI &op);
		Vector3DI &operator+= (const Vector3DF &op);
		Vector3DI &operator+= (const Vector4DF &op);

		Vector3DI &operator-= (const Vector3DI &op);
		Vector3DI &operator-= (const Vector3DF &op);
		Vector3DI &operator-= (const Vector4DF &op);
	
		Vector3DI &operator*= (const Vector3DI &op);
		Vector3DI &operator*= (const Vector3DF &op);
		Vector3DI &operator*= (const Vector4DF &op);

		Vector3DI &operator/= (const Vector3DI &op);
		Vector3DI &operator/= (const Vector3DF &op);
		Vector3DI &operator/= (const Vector4DF &op);

		Vector3DI operator+ (const int op)			{ return Vector3DI(x+op, y+op, z+op); }
		Vector3DI operator+ (const float op)		{ return Vector3DI(int(x+op), int(y+op), int( z+op)); }
		Vector3DI operator+ (const Vector3DI &op)	{ return Vector3DI(x+op.x, y+op.y, z+op.z); }
		Vector3DI operator- (const int op)			{ return Vector3DI(x-op, y-op, z-op); }
		Vector3DI operator- (const float op)		{ return Vector3DI(int(x-op), int( y-op), int( z-op)); }
		Vector3DI operator- (const Vector3DI &op)	{ return Vector3DI(x-op.x, y-op.y, z-op.z); }
		Vector3DI operator* (const int op)			{ return Vector3DI(x*op, y*op, z*op); }
		Vector3DI operator* (const float op)		{ return Vector3DI(int(x*op), int( y*op), int( z*op)); }
		Vector3DI operator* (const Vector3DI &op)	{ return Vector3DI(x*op.x, y*op.y, z*op.z); }		
		Vector3DI operator/ (const int op)			{ return Vector3DI(x/op, y/op, z/op); }
		Vector3DI operator/ (const float op)		{ return Vector3DI(int(x/op), int(y/op), int(z/op)); }
		Vector3DI operator/ (const Vector3DI &op)	{ return Vector3DI(x/op.x, y/op.y, z/op.z); }		

		Vector3DI &Cross (const Vector3DI &v);
		Vector3DI &Cross (const Vector3DF &v);	
		
		double Dot(const Vector3DI &v);
		double Dot(const Vector3DF &v);

		double Dist (const Vector3DI &v);
		double Dist (const Vector3DF &v);
		double Dist (const Vector4DF &v);

		double DistSq (const Vector3DI &v);
		double DistSq (const Vector3DF &v);
		double DistSq (const Vector4DF &v);

		Vector3DI &Normalize (void);
		double Length (void);

		VTYPE &X(void)				{return x;}
		VTYPE &Y(void)				{return y;}
		VTYPE &Z(void)				{return z;}
		VTYPE W(void)					{return 0;}
		const VTYPE &X(void) const	{return x;}
		const VTYPE &Y(void) const	{return y;}
		const VTYPE &Z(void) const	{return z;}
		const VTYPE W(void) const		{return 0;}
		VTYPE *Data (void)			{return &x;}
	};
	}

	#undef VNAME
	#undef VTYPE

	// Vector3DF Declarations

	#define VNAME		3DF
	#define VTYPE		float

	namespace nvdb {

	class GVDB_API Vector3DF {
	public:
		VTYPE x, y, z;
	
		// Constructors/Destructors
		Vector3DF() {x=0; y=0; z=0;}
		Vector3DF (const VTYPE xa, const VTYPE ya, const VTYPE za);
		Vector3DF (const Vector3DI &op);
		Vector3DF (const Vector3DF &op);
		Vector3DF (const Vector4DF &op);

		// Set Functions
		Vector3DF &Set (const VTYPE xa, const VTYPE ya, const VTYPE za);
		
		// Member Functions
		Vector3DF &operator= (const int op);
		Vector3DF &operator= (const double op);
		Vector3DF &operator= (const Vector3DI &op);
		Vector3DF &operator= (const Vector3DF &op);
		Vector3DF &operator= (const Vector4DF &op);

		Vector3DF &operator+= (const int op);
		Vector3DF &operator+= (const double op);
		Vector3DF &operator+= (const Vector3DI &op);
		Vector3DF &operator+= (const Vector3DF &op);
		Vector3DF &operator+= (const Vector4DF &op);

		Vector3DF &operator-= (const int op);
		Vector3DF &operator-= (const double op);
		Vector3DF &operator-= (const Vector3DI &op);
		Vector3DF &operator-= (const Vector3DF &op);
		Vector3DF &operator-= (const Vector4DF &op);
	
		Vector3DF &operator*= (const int op);
		Vector3DF &operator*= (const double op);
		Vector3DF &operator*= (const Vector3DI &op);
		Vector3DF &operator*= (const Vector3DF &op);
		Vector3DF &operator*= (const Vector4DF &op);
		Vector3DF &operator*= (const Matrix4F &op);

		Vector3DF &operator/= (const int op);
		Vector3DF &operator/= (const double op);
		Vector3DF &operator/= (const Vector3DI &op);
		Vector3DF &operator/= (const Vector3DF &op);
		Vector3DF &operator/= (const Vector4DF &op);

		// Slow operations - require temporary variables
		Vector3DF operator+ (int op)			{ return Vector3DF(x+float(op), y+float(op), z+float(op)); }
		Vector3DF operator+ (float op)		{ return Vector3DF(x+op, y+op, z+op); }
		Vector3DF operator+ (const Vector3DF &op)	{ return Vector3DF(x+op.x, y+op.y, z+op.z); }
		Vector3DF operator+ (const Vector3DI &op)	{ return Vector3DF(x+op.x, y+op.y, z+op.z); }
		Vector3DF operator- (int op)			{ return Vector3DF(x-float(op), y-float(op), z-float(op)); }
		Vector3DF operator- (float op)		{ return Vector3DF(x-op, y-op, z-op); }
		Vector3DF operator- (const Vector3DF &op)	{ return Vector3DF(x-op.x, y-op.y, z-op.z); }
		Vector3DF operator- (const Vector3DI &op)	{ return Vector3DF(x-op.x, y-op.y, z-op.z); }
		Vector3DF operator* (int op)			{ return Vector3DF(x*float(op), y*float(op), z*float(op)); }
		Vector3DF operator* (float op)		{ return Vector3DF(x*op, y*op, z*op); }
		Vector3DF operator* (const Vector3DF &op)	{ return Vector3DF(x*op.x, y*op.y, z*op.z); }		
		Vector3DF operator* (const Vector3DI &op)	{ return Vector3DF(x*op.x, y*op.y, z*op.z); }		
		Vector3DF operator/ (int op)			{ return Vector3DF(x*float(op), y*float(op), z*float(op)); }
		Vector3DF operator/ (float op)		{ return Vector3DF(x/op, y/op, z/op); }
		Vector3DF operator/ (const Vector3DF &op)	{ return Vector3DF(x/op.x, y/op.y, z/op.z); }		
		Vector3DF operator/ (const Vector3DI &op)	{ return Vector3DF(x/float(op.x), y/float(op.y), z/float(op.z)); }		
		// --


		Vector3DF &Cross (const Vector3DI &v);
		Vector3DF &Cross (const Vector3DF &v);	
		
		double Dot(const Vector3DI &v);
		double Dot(const Vector3DF &v);

		double Dist (const Vector3DI &v);
		double Dist (const Vector3DF &v);
		double Dist (const Vector4DF &v);

		double DistSq (const Vector3DI &v);
		double DistSq (const Vector3DF &v);
		double DistSq (const Vector4DF &v);

		Vector3DF &Random ()		{ x=float(rand())/RAND_MAX; y=float(rand())/RAND_MAX; z=float(rand())/RAND_MAX;  return *this;}
		Vector3DF &Random (Vector3DF a, Vector3DF b)		{ x=a.x+float(rand()*(b.x-a.x))/RAND_MAX; y=a.y+float(rand()*(b.y-a.y))/RAND_MAX; z=a.z+float(rand()*(b.z-a.z))/RAND_MAX;  return *this;}
		Vector3DF &Random (float x1,float x2, float y1, float y2, float z1, float z2)	{ x=x1+float(rand()*(x2-x1))/RAND_MAX; y=y1+float(rand()*(y2-y1))/RAND_MAX; z=z1+float(rand()*(z2-z1))/RAND_MAX;  return *this;}

		Vector3DF RGBtoHSV ();
		Vector3DF HSVtoRGB ();

		Vector3DF& Normalize (void);
		Vector3DF& Clamp (float a, float b);
		double Length (void);

		VTYPE &X()				{return x;}
		VTYPE &Y()				{return y;}
		VTYPE &Z()				{return z;}
		VTYPE W()					{return 0;}
		const VTYPE &X() const	{return x;}
		const VTYPE &Y() const	{return y;}
		const VTYPE &Z() const	{return z;}
		const VTYPE W() const		{return 0;}
		VTYPE *Data ()			{return &x;}
	};
	}
	
	#undef VNAME
	#undef VTYPE


	// Vector4DF Declarations

	#define VNAME		4DF
	#define VTYPE		float

	namespace nvdb {

	class GVDB_API Vector4DF {
	public:
		VTYPE x, y, z, w;
	
		Vector4DF &Set (const float xa, const float ya, const float za)	{ x =xa; y= ya; z=za; w=1; return *this;}
		Vector4DF &Set (const float xa, const float ya, const float za, const float wa )	{ x =xa; y= ya; z=za; w=wa; return *this;}

		// Constructors/Destructors
		Vector4DF() {x=0; y=0; z=0; w=0;}
		Vector4DF (const VTYPE xa, const VTYPE ya, const VTYPE za, const VTYPE wa);
		Vector4DF (const Vector3DI &op);
		Vector4DF (const Vector3DF &op);
		Vector4DF (const Vector4DF &op);

		// Member Functions
		Vector4DF &operator= (const int op);
		Vector4DF &operator= (const double op);
		Vector4DF &operator= (const Vector3DI &op);
		Vector4DF &operator= (const Vector3DF &op);
		Vector4DF &operator= (const Vector4DF &op);

		Vector4DF &operator+= (const int op);
		Vector4DF &operator+= (const float op);
		Vector4DF &operator+= (const double op);
		Vector4DF &operator+= (const Vector3DI &op);
		Vector4DF &operator+= (const Vector3DF &op);
		Vector4DF &operator+= (const Vector4DF &op);

		Vector4DF &operator-= (const int op);
		Vector4DF &operator-= (const double op);
		Vector4DF &operator-= (const Vector3DI &op);
		Vector4DF &operator-= (const Vector3DF &op);
		Vector4DF &operator-= (const Vector4DF &op);

		Vector4DF &operator*= (const int op);
		Vector4DF &operator*= (const double op);
		Vector4DF &operator*= (const Vector3DI &op);
		Vector4DF &operator*= (const Vector3DF &op);
		Vector4DF &operator*= (const Vector4DF &op);
		Vector4DF &operator*= (const float* op );
		Vector4DF &operator*= (const Matrix4F &op);		

		Vector4DF &operator/= (const int op);
		Vector4DF &operator/= (const double op);
		Vector4DF &operator/= (const Vector3DI &op);
		Vector4DF &operator/= (const Vector3DF &op);
		Vector4DF &operator/= (const Vector4DF &op);

		// Slow operations - require temporary variables
		Vector4DF operator+ (const int op)			{ return Vector4DF(x+float(op), y+float(op), z+float(op), w+float(op)); }
		Vector4DF operator+ (const float op)		{ return Vector4DF(x+op, y+op, z+op, w*op); }
		Vector4DF operator+ (const Vector4DF &op)	{ return Vector4DF(x+op.x, y+op.y, z+op.z, w+op.w); }
		Vector4DF operator- (const int op)			{ return Vector4DF(x-float(op), y-float(op), z-float(op), w-float(op)); }
		Vector4DF operator- (const float op)		{ return Vector4DF(x-op, y-op, z-op, w*op); }
		Vector4DF operator- (const Vector4DF &op)	{ return Vector4DF(x-op.x, y-op.y, z-op.z, w-op.w); }
		Vector4DF operator* (const int op)			{ return Vector4DF(x*float(op), y*float(op), z*float(op), w*float(op)); }
		Vector4DF operator* (const float op)		{ return Vector4DF(x*op, y*op, z*op, w*op); }
		Vector4DF operator* (const Vector4DF &op)	{ return Vector4DF(x*op.x, y*op.y, z*op.z, w*op.w); }		
		// --

		Vector4DF &Set ( CLRVAL clr )	{
			x = (float) RED(clr);		// (float( c      & 0xFF)/255.0)	
			y = (float) GRN(clr);		// (float((c>>8)  & 0xFF)/255.0)
			z = (float) BLUE(clr);		// (float((c>>16) & 0xFF)/255.0)
			w = (float) ALPH(clr);		// (float((c>>24) & 0xFF)/255.0)
			return *this;
		}
		Vector4DF& fromClr ( CLRVAL clr ) { return Set (clr); }
		CLRVAL toClr () { return (CLRVAL) COLORA( x, y, z, w ); }

		Vector4DF& Clamp ( float xc, float yc, float zc, float wc )
		{
			x = (x > xc) ? xc : x;
			y = (y > yc) ? yc : y;
			z = (z > zc) ? zc : z;
			w = (w > wc) ? wc : w;
			return *this;
		}

		Vector4DF &Cross (const Vector4DF &v);	
		
		double Dot (const Vector4DF &v);

		double Dist (const Vector4DF &v);

		double DistSq (const Vector4DF &v);

		Vector4DF &Normalize (void);
		double Length (void);

		Vector4DF &Random ()		{ x=float(rand())/RAND_MAX; y=float(rand())/RAND_MAX; z=float(rand())/RAND_MAX; w = 1;  return *this;}

		VTYPE &X(void)				{return x;}
		VTYPE &Y(void)				{return y;}
		VTYPE &Z(void)				{return z;}
		VTYPE &W(void)				{return w;}
		const VTYPE &X(void) const	{return x;}
		const VTYPE &Y(void) const	{return y;}
		const VTYPE &Z(void) const	{return z;}
		const VTYPE &W(void) const	{return w;}
		VTYPE *Data (void)			{return &x;}
	};
	}
	
	#undef VNAME
	#undef VTYPE

#endif

#ifndef MATRIX_DEF
	#define MATRIX_DEF
		
	#include <stdio.h>
	#include <iostream>
	#include <memory.h>
	#include <math.h>
	#include <string>

	// MatrixF Declaration	
	#define VNAME		F
	#define VTYPE		float

	namespace nvdb {

	class GVDB_API Matrix4F {
	public:	
		VTYPE	data[16];		

		// Constructors/Destructors
		Matrix4F ( const float* dat );
		Matrix4F () { for (int n=0; n < 16; n++) data[n] = 0.0; }
		Matrix4F ( float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7, float f8, float f9, float f10, float f11,	float f12, float f13, float f14, float f15 );

		// Member Functions
		VTYPE &operator () (const int n)					{ return data[n]; }
		VTYPE &operator () (const int c, const int r)	{ return data[ (r<<2)+c ]; }		
		Matrix4F &operator= (const unsigned char c);
		Matrix4F &operator= (const int c);
		Matrix4F &operator= (const double c);				
		Matrix4F &operator+= (const unsigned char c);
		Matrix4F &operator+= (const int c);
		Matrix4F &operator+= (const double c);				
		Matrix4F &operator-= (const unsigned char c);
		Matrix4F &operator-= (const int c);
		Matrix4F &operator-= (const double c);
		Matrix4F &operator*= (const unsigned char c);
		Matrix4F &operator*= (const int c);
		Matrix4F &operator*= (const double c);
		Matrix4F &operator/= (const unsigned char c);
		Matrix4F &operator/= (const int c);
		Matrix4F &operator/= (const double c);		

		Matrix4F &operator=  (const float* op);
		Matrix4F &operator*= (const Matrix4F& op);
		Matrix4F &operator*= (const float* op);	

		Matrix4F &PreTranslate (const Vector3DF& t);
		Matrix4F &operator+= (const Vector3DF& t);		// quick translate
		Matrix4F &operator*= (const Vector3DF& t);		// quick scale
		
		Matrix4F &Transpose (void);
		Matrix4F &RotateZYX ( const Vector3DF& angs );
		Matrix4F &RotateZYXT (const Vector3DF& angs, const Vector3DF& t);
		Matrix4F &RotateTZYX (const Vector3DF& angs, const Vector3DF& t);
		Matrix4F &RotateTZYXS(const Vector3DF& angs, const Vector3DF& t, const Vector3DF& s);
		Matrix4F &RotateX (const double ang);
		Matrix4F &RotateY (const double ang);
		Matrix4F &RotateZ (const double ang);
		Matrix4F &Ortho (double sx, double sy, double n, double f);		
		Matrix4F &Translate (double tx, double ty, double tz);
		Matrix4F &Scale (double sx, double sy, double sz);
		Matrix4F &Basis (const Vector3DF &yaxis);
		Matrix4F &Basis (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3);		
		Matrix4F &InvertTRS ();
		Matrix4F &Identity ();
		Matrix4F &Identity (const int order);
		Matrix4F &Multiply (const Matrix4F &op);

		void Print ();
		std::string WriteToStr ();

		Matrix4F operator* (const float &op);	
		Matrix4F operator* (const Vector3DF &op);	

		// Scale-Rotate-Translate (compound matrix)
		Matrix4F &TransSRT (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const Vector3DF& s);
		Matrix4F &SRT (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const Vector3DF& s);
		Matrix4F &SRT (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const float s);

		// invTranslate-invRotate-invScale (compound matrix)
		Matrix4F &InvTRS (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const Vector3DF& s);
		Matrix4F &InvTRS (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const float s);

		Matrix4F &operator= ( float* mat);
		Matrix4F &InverseProj ( const float* mat );
		Matrix4F &InverseView ( const float* mat, const Vector3DF& pos );
		Vector4DF GetT ( float* mat );

		int GetX()			{ return 4; }
		int GetY()			{ return 4; }
		int GetRows(void)	{ return 4; }
		int GetCols(void)	{ return 4; }	
		int GetLength(void)	{ return 16; }
		VTYPE *GetData(void)	{ return data; }
		Vector4DF GetRowVec(int r);

		unsigned char *GetDataC (void) const	{return NULL;}
		int *GetDataI (void)	const			{return NULL;}
		float *GetDataF (void) const		{return (float*) data;}

		float GetF (const int r, const int c);
	};
	}
	#undef VNAME
	#undef VTYPE

#endif

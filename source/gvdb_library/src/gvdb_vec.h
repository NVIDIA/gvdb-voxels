//--------------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2016, NVIDIA Corporation.
// SPDX-License-Identifier: Apache-2.0
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
// Version 1.1: Rama Hoetzlein, 3/25/2018
// Version 1.1.1: Neil Bickford, 7/1/2020
//--------------------------------------------------------------------------------

#ifndef DEF_GVDB_VEC
#define DEF_GVDB_VEC

#include "gvdb_types.h"
#include <iostream>
#include <memory.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

namespace nvdb {
	class Vector4DF;
	class Matrix4F;

	template<class VTYPE>
	VTYPE min3(VTYPE a, VTYPE b, VTYPE c) {
		return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
	}

	template<class VTYPE>
	VTYPE max3(VTYPE a, VTYPE b, VTYPE c) {
		return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c);
	}

	// Represents a point in 3D as a column vector. When multiplying with a `Matrix4DF`, this is treated as a 4-element
	// vector with 4th component set to 1.
	template<class VTYPE>
	class Vector3D {
	public:
		VTYPE x = 0, y = 0, z = 0;

		// Constructors/Destructors
		Vector3D() {};
		Vector3D(const VTYPE xa, const VTYPE ya, const VTYPE za) { x = xa; y = ya; z = za; }
		template<class OTHERVTYPE>
		Vector3D(const Vector3D<OTHERVTYPE>& op) { x = (VTYPE)op.x; y = (VTYPE)op.y; z = (VTYPE)op.z; }
		Vector3D(const Vector4DF& op);

		// Set Functions
		Vector3D<VTYPE>& Set(const VTYPE xa, const VTYPE ya, const VTYPE za) {
			x = xa; y = ya; z = za;
			return *this;
		}

		// Member Functions
		template<class VECTOR>
		Vector3D<VTYPE>& operator= (const VECTOR& op) {
			return Set(static_cast<VTYPE>(op.x), static_cast<VTYPE>(op.y), static_cast<VTYPE>(op.z));
		}

		template<class T>
		Vector3D<VTYPE>& operator+= (const T& op) {
			return Set(x + static_cast<VTYPE>(op), y + static_cast<VTYPE>(op), z + static_cast<VTYPE>(op));
		}
		template<class OTHERVTYPE>
		Vector3D<VTYPE>& operator+= (const Vector3D<OTHERVTYPE>& op) {
			return Set(x + static_cast<VTYPE>(op.x), y + static_cast<VTYPE>(op.y), z + static_cast<VTYPE>(op.z));
		}
		Vector3D<VTYPE>& operator+= (const Vector4DF& op);

		template<class T>
		Vector3D<VTYPE>& operator-= (const T& op) {
			return Set(x - static_cast<VTYPE>(op), y - static_cast<VTYPE>(op), z - static_cast<VTYPE>(op));
		}
		template<class OTHERVTYPE>
		Vector3D<VTYPE>& operator-= (const Vector3D<OTHERVTYPE>& op) {
			return Set(x - static_cast<VTYPE>(op.x), y - static_cast<VTYPE>(op.y), z - static_cast<VTYPE>(op.z));
		}
		Vector3D<VTYPE>& operator-= (const Vector4DF& op);

		template<class T>
		Vector3D<VTYPE>& operator*= (const T& op) {
			return Set(x * static_cast<VTYPE>(op), y * static_cast<VTYPE>(op), z * static_cast<VTYPE>(op));
		}
		template<class OTHERVTYPE>
		Vector3D<VTYPE>& operator*= (const Vector3D<OTHERVTYPE>& op) {
			return Set(x * static_cast<VTYPE>(op.x), y * static_cast<VTYPE>(op.y), z * static_cast<VTYPE>(op.z));
		}
		Vector3D<VTYPE>& operator*= (const Vector4DF& op);

		// For floating-point vectors, sets this vector to and returns the result of multiplying this vector by the
		// matrix on the left. This vector is interpreted as a point (x, y, z, 1) when multiplying, and takes the first
		// three components of the result.
		Vector3D<VTYPE>& operator*= (const Matrix4F& op);

		template<class T>
		Vector3D<VTYPE>& operator/= (const T& op) {
			return Set(x / static_cast<VTYPE>(op), y / static_cast<VTYPE>(op), z / static_cast<VTYPE>(op));
		}
		template<class OTHERVTYPE>
		Vector3D<VTYPE>& operator/= (const Vector3D<OTHERVTYPE>& op) {
			return Set(x / static_cast<VTYPE>(op.x), y / static_cast<VTYPE>(op.y), z / static_cast<VTYPE>(op.z));
		}
		Vector3D<VTYPE>& operator/= (const Vector4DF& op);

		template<class T>
		Vector3D<VTYPE> operator+ (const T op) const { return Vector3D(x + static_cast<VTYPE>(op), y + static_cast<VTYPE>(op), z + static_cast<VTYPE>(op)); }
		template<class OTHERVTYPE>
		Vector3D<VTYPE> operator+ (const Vector3D<OTHERVTYPE>& op) const { return Vector3D(x + static_cast<VTYPE>(op.x), y + static_cast<VTYPE>(op.y), z + static_cast<VTYPE>(op.z)); }
		template<class T>
		Vector3D<VTYPE> operator- (const T op) const { return Vector3D(x - static_cast<VTYPE>(op), y - static_cast<VTYPE>(op), z - static_cast<VTYPE>(op)); }
		template<class OTHERVTYPE>
		Vector3D<VTYPE> operator- (const Vector3D<OTHERVTYPE>& op) const { return Vector3D(x - static_cast<VTYPE>(op.x), y - static_cast<VTYPE>(op.y), z - static_cast<VTYPE>(op.z)); }
		template<class T>
		Vector3D<VTYPE> operator* (const T op) const { return Vector3D(x * static_cast<VTYPE>(op), y * static_cast<VTYPE>(op), z * static_cast<VTYPE>(op)); }
		template<class OTHERVTYPE>
		Vector3D<VTYPE> operator* (const Vector3D<OTHERVTYPE>& op) const { return Vector3D(x * static_cast<VTYPE>(op.x), y * static_cast<VTYPE>(op.y), z * static_cast<VTYPE>(op.z)); }
		template<class T>
		Vector3D<VTYPE> operator/ (const T op) const { return Vector3D(x / static_cast<VTYPE>(op), y / static_cast<VTYPE>(op), z / static_cast<VTYPE>(op)); }
		template<class OTHERVTYPE>
		Vector3D<VTYPE> operator/ (const Vector3D<OTHERVTYPE>& op) const { return Vector3D(x / static_cast<VTYPE>(op.x), y / static_cast<VTYPE>(op.y), z / static_cast<VTYPE>(op.z)); }

		template<class OTHERVTYPE>
		Vector3D<VTYPE>& Cross(const Vector3D<OTHERVTYPE>& v) {
			double ax = static_cast<double>(x);
			double ay = static_cast<double>(y);
			double az = static_cast<double>(z);
			x = static_cast<VTYPE>(ay * static_cast<double>(v.z) - az * static_cast<double>(v.y));
			y = static_cast<VTYPE>(-ax * static_cast<double>(v.z) + az * static_cast<double>(v.x));
			z = static_cast<VTYPE>(ax * static_cast<double>(v.y) - ay * static_cast<double>(v.x));
			return *this;
		}

		template<class OTHERVTYPE>
		double Dot(const Vector3D<OTHERVTYPE>& v) const {
			return static_cast<double>(x) * static_cast<double>(v.x)
				+ static_cast<double>(y) * static_cast<double>(v.y)
				+ static_cast<double>(z) * static_cast<double>(v.z);
		}

		// Returns the Euclidean distance between this vector and v.
		template<class VECTOR>
		double Dist(const VECTOR& v) const {
			double distsq = DistSq(v);
			if (distsq != 0) {
				return sqrt(distsq);
			}
			return 0.0;
		}

		// Returns the squared Euclidean distance between this vector and v.
		template<class VECTOR>
		double DistSq(const VECTOR& v) const {
			double a, b, c;
			a = static_cast<double>(x) - static_cast<double>(v.x);
			b = static_cast<double>(y) - static_cast<double>(v.y);
			c = static_cast<double>(z) - static_cast<double>(v.z);
			return a * a + b * b + c * c;
		}

		// For float vectors, sets each component to a random value in [0, 1). For int vectors, sets each component to
		// a random value in [0, RAND_MAX - 1].
		Vector3D<VTYPE>& Random() {
			return Random(0, 1, 0, 1, 0, 1);
		}
		// Sets each component to a random value in [a, b).
		Vector3D<VTYPE>& Random(const Vector3D<float>& a, const Vector3D<float>& b) {
			return Random(a.x, b.x, a.y, b.y, a.z, b.z);
		}
		// Sets each component to a random value in [x1, x2) x [y1, x2) x [z1, z2).
		Vector3D<VTYPE>& Random(float x1, float x2, float y1, float y2, float z1, float z2) {
			x = static_cast<VTYPE>(x1 + float(rand()) * (x2 - x1) / RAND_MAX);
			y = static_cast<VTYPE>(y1 + float(rand()) * (y2 - y1) / RAND_MAX);
			z = static_cast<VTYPE>(z1 + float(rand()) * (z2 - z1) / RAND_MAX);
			return *this;
		}

		// Converts from RGB to HSV color. Only really makes sense for floating-point vectors.
		Vector3D<VTYPE> RGBtoHSV() {
			VTYPE h, s, v;
			VTYPE minv, maxv;
			int i;
			VTYPE f;

			minv = min3<VTYPE>(x, y, z);
			maxv = max3<VTYPE>(x, y, z);
			if (minv == maxv) {
				v = maxv;
				h = 0;
				s = 0;
			}
			else {
				v = maxv;
				s = (maxv - minv) / maxv;
				f = (x == minv) ? y - z : ((y == minv) ? z - x : x - y);
				i = (x == minv) ? 3 : ((y == minv) ? 5 : 1);
				h = (i - f / (maxv - minv)) / 6;
			}
			return Vector3D<VTYPE>(h, s, v);
		}
		// Converts from HSV to RGB color. Only really makes sense for floating-point vectors.
		Vector3D<VTYPE> HSVtoRGB() {
			VTYPE m, n, f;
			int i = (int)floor(x * 6);
			f = x * 6 - i;
			if (i % 2 == 0) f = 1 - f;
			m = z * (1 - y);
			n = z * (1 - y * f);
			switch (i) {
			case 6:
			case 0: return Vector3D<VTYPE>(z, n, m);	break;
			case 1: return Vector3D<VTYPE>(n, z, m);	break;
			case 2: return Vector3D<VTYPE>(m, z, n);	break;
			case 3: return Vector3D<VTYPE>(m, n, z);	break;
			case 4: return Vector3D<VTYPE>(n, m, z);	break;
			case 5: return Vector3D<VTYPE>(z, m, n);	break;
			};
			return Vector3D<VTYPE>(1, 1, 1);
		}

		// For non-int vectors, normalizes the vector so that it has length 1. If the vector has length 0, does nothing.
		// For int vectors, normalizes the vector so that it has length 255, then truncates each component to integer.
		Vector3D<VTYPE>& Normalize();

		// Clamps each value to the range [a, b].
		Vector3D<VTYPE>& Clamp(VTYPE a, VTYPE b) {
			x = (x < a) ? a : ((x > b) ? b : x);
			y = (y < a) ? a : ((y > b) ? b : y);
			z = (z < a) ? a : ((z > b) ? b : z);
			return *this;
		}
		// Returns the length of the vector.
		double Length() const {
			return sqrt(LengthSq());
		}
		// Returns the squared length of the vector.
		double LengthSq() const {
			double dx = static_cast<double>(x);
			double dy = static_cast<double>(y);
			double dz = static_cast<double>(z);
			return dx * dx + dy * dy + dz * dz;
		}

		VTYPE& X() { return x; }
		VTYPE& Y() { return y; }
		VTYPE& Z() { return z; }
		VTYPE W() const { return 0; }
		const VTYPE& X() const { return x; }
		const VTYPE& Y() const { return y; }
		const VTYPE& Z() const { return z; }
		VTYPE* Data(void) { return &x; }
	};

	template<class VTYPE>
	inline Vector3D<VTYPE>& Vector3D<VTYPE>::Normalize(){
		double n = LengthSq();
		if (n != 0.0) {
			double rcpLen = 1.0 / sqrt(n);
			x = static_cast<VTYPE>(x * rcpLen);
			y = static_cast<VTYPE>(y * rcpLen);
			z = static_cast<VTYPE>(z * rcpLen);
		}
		return *this;
	}
	template<>
	inline Vector3D<int>& Vector3D<int>::Normalize() {
		double n = LengthSq();
		if (n != 0.0) {
			n = sqrt(n);
			x = int(static_cast<double>(x) * 255.0 / n);
			y = int(static_cast<double>(y) * 255.0 / n);
			z = int(static_cast<double>(z) * 255.0 / n);
		}
		return *this;
	}

	// Explicit template instantiations and short names.
	// See https://anteru.net/blog/2008/c-tricks-6-explicit-template-instantiation/
	template class GVDB_API Vector3D<int>;
	template class GVDB_API Vector3D<float>;
	using Vector3DI = Vector3D<int>;
	using Vector3DF = Vector3D<float>;

	// Represents a 4-element column vector.
#define VTYPE float
	class GVDB_API Vector4DF {
	public:
		VTYPE x = 0, y = 0, z = 0, w = 0;

		Vector4DF& Set(const float xa, const float ya, const float za) { x = xa; y = ya; z = za; w = 1; return *this; }
		Vector4DF& Set(const float xa, const float ya, const float za, const float wa) { x = xa; y = ya; z = za; w = wa; return *this; }

		// Constructors/Destructors
		Vector4DF() {};
		Vector4DF(const VTYPE xa, const VTYPE ya, const VTYPE za, const VTYPE wa);
		Vector4DF(const Vector3DI& op);
		Vector4DF(const Vector3DF& op);
		Vector4DF(const Vector4DF& op);

		// Member Functions
		Vector4DF& operator= (const int op);
		Vector4DF& operator= (const double op);
		Vector4DF& operator= (const Vector3DI& op);
		Vector4DF& operator= (const Vector3DF& op);
		Vector4DF& operator= (const Vector4DF& op);

		Vector4DF& operator+= (const int op);
		Vector4DF& operator+= (const double op);
		Vector4DF& operator+= (const Vector3DI& op);
		Vector4DF& operator+= (const Vector3DF& op);
		Vector4DF& operator+= (const Vector4DF& op);

		Vector4DF& operator-= (const int op);
		Vector4DF& operator-= (const double op);
		Vector4DF& operator-= (const Vector3DI& op);
		Vector4DF& operator-= (const Vector3DF& op);
		Vector4DF& operator-= (const Vector4DF& op);

		Vector4DF& operator*= (const int op);
		Vector4DF& operator*= (const double op);
		Vector4DF& operator*= (const Vector3DI& op);
		Vector4DF& operator*= (const Vector3DF& op);
		Vector4DF& operator*= (const Vector4DF& op);
		Vector4DF& operator*= (const float* op);
		Vector4DF& operator*= (const Matrix4F& op);

		Vector4DF& operator/= (const int op);
		Vector4DF& operator/= (const double op);
		Vector4DF& operator/= (const Vector3DI& op);
		Vector4DF& operator/= (const Vector3DF& op);
		Vector4DF& operator/= (const Vector4DF& op);

		// Slow operations - require temporary variables
		Vector4DF operator+ (const int op) { return Vector4DF(x + float(op), y + float(op), z + float(op), w + float(op)); }
		Vector4DF operator+ (const float op) { return Vector4DF(x + op, y + op, z + op, w * op); }
		Vector4DF operator+ (const Vector4DF& op) { return Vector4DF(x + op.x, y + op.y, z + op.z, w + op.w); }
		Vector4DF operator- (const int op) { return Vector4DF(x - float(op), y - float(op), z - float(op), w - float(op)); }
		Vector4DF operator- (const float op) { return Vector4DF(x - op, y - op, z - op, w * op); }
		Vector4DF operator- (const Vector4DF& op) { return Vector4DF(x - op.x, y - op.y, z - op.z, w - op.w); }
		Vector4DF operator* (const int op) { return Vector4DF(x * float(op), y * float(op), z * float(op), w * float(op)); }
		Vector4DF operator* (const float op) { return Vector4DF(x * op, y * op, z * op, w * op); }
		Vector4DF operator* (const Vector4DF& op) { return Vector4DF(x * op.x, y * op.y, z * op.z, w * op.w); }
		// --

		Vector4DF& Set(CLRVAL clr) {
			x = (float)RED(clr);		// (float( c      & 0xFF)/255.0)	
			y = (float)GRN(clr);		// (float((c>>8)  & 0xFF)/255.0)
			z = (float)BLUE(clr);		// (float((c>>16) & 0xFF)/255.0)
			w = (float)ALPH(clr);		// (float((c>>24) & 0xFF)/255.0)
			return *this;
		}
		Vector4DF& fromClr(CLRVAL clr) { return Set(clr); }
		CLRVAL toClr() { return (CLRVAL)COLORA(x, y, z, w); }

		Vector4DF& Clamp(float xc, float yc, float zc, float wc)
		{
			x = (x > xc) ? xc : x;
			y = (y > yc) ? yc : y;
			z = (z > zc) ? zc : z;
			w = (w > wc) ? wc : w;
			return *this;
		}

		Vector4DF& Cross(const Vector4DF& v);

		double Dot(const Vector4DF& v);

		double Dist(const Vector4DF& v);

		double DistSq(const Vector4DF& v);

		Vector4DF& Normalize(void);
		double Length(void);

		// Vector4DF& Random() { x = float(rand()) / RAND_MAX; y = float(rand()) / RAND_MAX; z = float(rand()) / RAND_MAX; w = 1;  return *this; }

		VTYPE& X(void) { return x; }
		VTYPE& Y(void) { return y; }
		VTYPE& Z(void) { return z; }
		VTYPE& W(void) { return w; }
		const VTYPE& X(void) const { return x; }
		const VTYPE& Y(void) const { return y; }
		const VTYPE& Z(void) const { return z; }
		const VTYPE& W(void) const { return w; }
		VTYPE* Data(void) { return &x; }
	};
#undef VTYPE

#define VTYPE float
	class GVDB_API Matrix4F {
	public:
		// Stores the elements of the matrix in column-major order. That is, this represents the matrix
		// ( data[ 0] data[ 4] data[ 8] data[12] )
		// ( data[ 1] data[ 5] data[ 9] data[13] )
		// ( data[ 2] data[ 6] data[10] data[14] )
		// ( data[ 3] data[ 7] data[11] data[15] ).
		// 1.1.1: Set to the identity by default.
		VTYPE	data[16] = { 1.0, 0.0, 0.0, 0.0,
							0.0, 1.0, 0.0, 0.0,
							0.0, 0.0, 1.0, 0.0,
							0.0, 0.0, 0.0, 1.0 };

		// Constructors/Destructors
		Matrix4F() {};
		Matrix4F(const float* dat);
		Matrix4F(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7, float f8, float f9, float f10, float f11, float f12, float f13, float f14, float f15);

		// Member Functions
		VTYPE& operator () (const int n) { return data[n]; }
		// Sets all elements to c.
		Matrix4F& operator= (const unsigned char c);
		// Sets all elements to c.
		Matrix4F& operator= (const int c);
		// Sets all elements to c.
		Matrix4F& operator= (const double c);
		// Adds c to all elements.
		Matrix4F& operator+= (const unsigned char c);
		// Adds c to all elements.
		Matrix4F& operator+= (const int c);
		// Adds c to all elements.
		Matrix4F& operator+= (const double c);
		// Subtracts c from all elements.
		Matrix4F& operator-= (const unsigned char c);
		// Subtracts c from all elements.
		Matrix4F& operator-= (const int c);
		// Subtracts c from all elements.
		Matrix4F& operator-= (const double c);
		// Multiplies all elements by c.
		Matrix4F& operator*= (const unsigned char c);
		// Multiplies all elements by c.
		Matrix4F& operator*= (const int c);
		// Multiplies all elements by c.
		Matrix4F& operator*= (const double c);
		// Divides all elements by c.
		Matrix4F& operator/= (const unsigned char c);
		// Divides all elements by c.
		Matrix4F& operator/= (const int c);
		// Divides all elements by c.
		Matrix4F& operator/= (const double c);

		Matrix4F& operator=  (const float* op);
		Matrix4F& operator*= (const Matrix4F& op);
		Matrix4F& operator*= (const float* op);

		Matrix4F& PreTranslate(const Vector3DF& t);
		// Applies post-translation, i.e. sets M to T*M, where T is the matrix that translates by t.
		Matrix4F& operator+= (const Vector3DF& t);
		// Applies post-scaling, i.e. sets M to S*M, where S is the diagonal matrix with diagonal entries
		// (s.x, s.y, s.z, 1).
		Matrix4F& operator*= (const Vector3DF& s);

		Matrix4F& Transpose(void);
		// Sets this matrix to and returns a matrix representing a rotation by angs.x around the X axis, followed by a
		// rotation by angs.y around the Y axis, followed by a rotation by angs.z around the Z axis.
		// All angles are defined according to the right-hand rule (i.e. counterclockwise when looking from the
		// positive axis to the origin, and clockwise when looking from the origin along the positive axis.)
		// In other words, sets and returns the matrix ZYX, where Z is RotateZ, Y is RotateY, and X is RotateX.
		Matrix4F& RotateZYX(const Vector3DF& angs);
		// Sets this matrix to and returns a matrix representing translation by T, followed by RotateZYX.
		Matrix4F& RotateZYXT(const Vector3DF& angs, const Vector3DF& t);
		// Sets this matrix to and returns a matrix representing RotateZYX, followed by translation by T.
		Matrix4F& RotateTZYX(const Vector3DF& angs, const Vector3DF& t);
		// Sets this matrix to and returns a matrix representing scaling, followed by RotateZYX,
		// followed by translation by T.
		Matrix4F& RotateTZYXS(const Vector3DF& angs, const Vector3DF& t, const Vector3DF& s);
		// Sets this matrix to and returns a matrix representing rotation by ang around the X axis.
		// All angles are defined according to the right-hand rule (i.e. counterclockwise when looking from the
		// positive axis to the origin, and clockwise when looking from the origin along the positive axis.)
		Matrix4F& RotateX(const double ang);
		// Sets this matrix to and returns a matrix representing rotation by ang around the Y axis.
		// All angles are defined according to the right-hand rule (i.e. counterclockwise when looking from the
		// positive axis to the origin, and clockwise when looking from the origin along the positive axis.)
		Matrix4F& RotateY(const double ang);
		// Sets this matrix to and returns a matrix representing rotation by ang around the Z axis.
		// All angles are defined according to the right-hand rule (i.e. counterclockwise when looking from the
		// positive axis to the origin, and clockwise when looking from the origin along the positive axis.)
		Matrix4F& RotateZ(const double ang);
		Matrix4F& Ortho(double sx, double sy, double n, double f);
		// Makes this matrix represent the map (x, y, z) |-> (x + tx, y + ty, z + tz).
		Matrix4F& Translate(double tx, double ty, double tz);
		// Makes this matrix represent the map (x, y, z) |-> (sx * x, sy * y, sz * z).
		Matrix4F& Scale(double sx, double sy, double sz);
		Matrix4F& Basis(const Vector3DF& yaxis);
		Matrix4F& Basis(const Vector3DF& c1, const Vector3DF& c2, const Vector3DF& c3);
		// Inverts this matrix and returns itself. If the matrix has determinant 0, does nothing.
		Matrix4F& InvertTRS();
		// Sets this matrix to and returns the identity matrix.
		Matrix4F& Identity();

		void Print();
		std::string WriteToStr();

		Matrix4F operator* (const float& op);
		Vector3DF operator* (const Vector3DF& op);

		// Scale-Rotate-Translate (compound matrix)
		Matrix4F& TransSRT(const Vector3DF& c1, const Vector3DF& c2, const Vector3DF& c3, const Vector3DF& t, const Vector3DF& s);
		Matrix4F& SRT(const Vector3DF& c1, const Vector3DF& c2, const Vector3DF& c3, const Vector3DF& t, const Vector3DF& s);
		Matrix4F& SRT(const Vector3DF& c1, const Vector3DF& c2, const Vector3DF& c3, const Vector3DF& t, const float s);

		// invTranslate-invRotate-invScale (compound matrix)
		Matrix4F& InvTRS(const Vector3DF& c1, const Vector3DF& c2, const Vector3DF& c3, const Vector3DF& t, const Vector3DF& s);
		Matrix4F& InvTRS(const Vector3DF& c1, const Vector3DF& c2, const Vector3DF& c3, const Vector3DF& t, const float s);

		Matrix4F& operator= (float* mat);
		Matrix4F& InverseProj(const float* mat);
		Matrix4F& InverseView(const float* mat, const Vector3DF& pos);
		Vector4DF GetT(float* mat);

		// Composing operations
		// Sets this matrix `M` to `T*M`, where `T` translates by `translation`. Alias for `operator +=`.
		Matrix4F& TranslateInPlace(const Vector3DF& translation);
		// Sets this matrix `M` to `mtx*M`.
		Matrix4F& LeftMultiplyInPlace(const Matrix4F& mtx);
		// Sets this matrix `M` to `S*M`, where S is the diagonal matrix with entries scale.x, scale.y, scale.z, and 1.
		Matrix4F& ScaleInPlace(const Vector3DF& scale);
		// Sets this matrix `M` to `(T^-1)*M`, where `T` translates by `translation`. Equivalent to translating
		// by `-translation`.
		Matrix4F& InvTranslateInPlace(const Vector3DF& translation);
		// Sets this matrix `M` to `(mtx^-1)*M`.
		Matrix4F& InvLeftMultiplyInPlace(Matrix4F mtx);
		// Sets this matrix `M` to `(S^-1)*M`, where S is the diagonal matrix with entries scale.x, scale.y, scale.z,
		// and 1. Equivalent to scaling by 1/scale.
		Matrix4F& InvScaleInPlace(const Vector3DF& scale);


		int GetX() { return 4; }
		int GetY() { return 4; }
		int GetRows(void) { return 4; }
		int GetCols(void) { return 4; }
		int GetLength(void) { return 16; }
		VTYPE* GetData(void) { return data; }

		float* GetDataF(void) const { return (float*)data; }

		float& operator() (int row, int col) {
			return data[4 * col + row];
		}

		const float& operator()(int row, int col) const {
			return data[4 * col + row];
		}
	};
#undef VTYPE

	template<class VTYPE>
	Vector3D<VTYPE>::Vector3D(const Vector4DF& op) { x = (VTYPE)op.x; y = (VTYPE)op.y; z = (VTYPE)op.z; }

	template<class VTYPE>
	Vector3D<VTYPE>& Vector3D<VTYPE>::operator+= (const Vector4DF& op) {
		return Set(x + static_cast<VTYPE>(op.x), y + static_cast<VTYPE>(op.y), z + static_cast<VTYPE>(op.z));
	}

	template<class VTYPE>
	Vector3D<VTYPE>& Vector3D<VTYPE>::operator-= (const Vector4DF& op) {
		return Set(x - static_cast<VTYPE>(op.x), y - static_cast<VTYPE>(op.y), z - static_cast<VTYPE>(op.z));
	}

	template<class VTYPE>
	Vector3D<VTYPE>& Vector3D<VTYPE>::operator*= (const Vector4DF& op) {
		return Set(x * static_cast<VTYPE>(op.x), y * static_cast<VTYPE>(op.y), z * static_cast<VTYPE>(op.z));
	}

	template<class VTYPE>
	Vector3D<VTYPE>& Vector3D<VTYPE>::operator*= (const Matrix4F& op) {
		float xa = (float)x * op.data[0] + (float)y * op.data[4] + (float)z * op.data[8] + op.data[12];
		float ya = (float)x * op.data[1] + (float)y * op.data[5] + (float)z * op.data[9] + op.data[13];
		float za = (float)x * op.data[2] + (float)y * op.data[6] + (float)z * op.data[10] + op.data[14];
		return Set(static_cast<VTYPE>(xa), static_cast<VTYPE>(ya), static_cast<VTYPE>(za));
	}

	template<class VTYPE>
	Vector3D<VTYPE>& Vector3D<VTYPE>::operator/= (const Vector4DF& op) {
			return Set(x / static_cast<VTYPE>(op.x), y / static_cast<VTYPE>(op.y), z / static_cast<VTYPE>(op.z));
	}
	
}

#endif // #ifndef DEF_GVDB_VEC
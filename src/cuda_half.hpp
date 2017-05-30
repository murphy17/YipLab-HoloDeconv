/*
 * cuda_half.hpp
 *
 * Establish common interface between half and float types.
 * haven't defined int stuff
 *
 *  Created on: May 30, 2017
 *      Author: michaelmurphy
 */

#ifndef cuda_half
#define cuda_half

namespace cuda_fp16
{
	#include <cuda_fp16.h>
}

class half
{
private:
	cuda_fp16::half _;

public:
	// constructors
	inline __device__
	half() { ; }

	inline __device__
	half(cuda_fp16::half a) { _ = a; }

	inline __device__
	half(float a) { _ = cuda_fp16::__float2half(a); }

	// assignment
	inline __device__
    half& operator=(float a) { _ = cuda_fp16::__float2half(a); return *this; }

	// conversion
	inline __device__
	operator cuda_fp16::half() { return _; }

	inline __device__
	operator float() { return cuda_fp16::__half2float(_); }

	// addition
	inline __device__
    half operator+(half a) { return half(cuda_fp16::__hadd(_, a._)); }

	inline __device__
    half operator+=(half a) { return (*this = *this + a); }

	// subtraction
	inline __device__
    half operator-() { return half(cuda_fp16::__hneg(_)); }

	inline __device__
    half operator-(half a) { return half(cuda_fp16::__hsub(_, a._)); }

	inline __device__
    half operator-=(half a) { return (*this = *this - a); }

	// multiplication
	inline __device__
    half operator*(half a) { return half(cuda_fp16::__hmul(_, a._)); }

	inline __device__
    half operator*=(half a) { return (*this = *this * a); }

	// division
	inline __device__
    half operator/(half a) { return half(cuda_fp16::__hmul(_, cuda_fp16::hrcp(a._))); }

	inline __device__
    half operator/=(half a) { return (*this = *this / a); }
};

class half2
{
public:
	half x, y;

	// constructors
	inline __device__
	half2() { ; }

	inline __device__
	half2(cuda_fp16::half a) { x = a; y = a; }

	inline __device__
	half2(cuda_fp16::half a, cuda_fp16::half b) { x = a; y = b; }

	inline __device__
	half2(cuda_fp16::half2 a) { *(cuda_fp16::half2 *)&x = a; }

	inline __device__
	half2(float a) { *(cuda_fp16::half2 *)&x = cuda_fp16::__float2half2_rn(a); }

	inline __device__
	half2(float a, float b) { *(cuda_fp16::half2 *)&x = cuda_fp16::__floats2half2_rn(a, b); }

	// assignment
	inline __device__
	half2& operator=(cuda_fp16::half a) { x = a; y = a; return *this; }

	inline __device__
	half2& operator=(cuda_fp16::half2 a) { *(half2 *)&x = a; return *this; }

	inline __device__
	half2& operator=(float a) { *(cuda_fp16::half2 *)&x = cuda_fp16::__float2half2_rn(a); return *this; }

	inline __device__
	half2& operator=(float2 a) { *(cuda_fp16::half2 *)&x = cuda_fp16::__floats2half2_rn(a.x, a.y); return *this; }

	// conversion
	inline __device__
	operator cuda_fp16::half2() { return *(cuda_fp16::half2 *)&x; }

	inline __device__
	operator float2() { return cuda_fp16::__half22float2(*(cuda_fp16::half2 *)&x); }

	// addition
	inline __device__
    half2 operator+(half2 a) { return half2(cuda_fp16::__hadd2(*(cuda_fp16::half2 *)&x, *(cuda_fp16::half2 *)&(a.x))); }

	inline __device__
    half2 operator+=(half2 a) { return (*this = *this + a); }

	// subtraction
	inline __device__
    half2 operator-() { return half2(cuda_fp16::__hneg2(*(cuda_fp16::half2 *)&x)); }

	inline __device__
    half2 operator-(half2 a) { return half2(cuda_fp16::__hmul2(*(cuda_fp16::half2 *)&x, *(cuda_fp16::half2 *)&(a.x))); }

	inline __device__
    half2 operator-=(half2 a) { return (*this = *this - a); }

	// element-wise multiplication
	inline __device__
    half2 operator*(half2 a) { return half2(cuda_fp16::__hmul2(*(cuda_fp16::half2 *)&x, *(cuda_fp16::half2 *)&(a.x))); }

	inline __device__
    half2 operator*=(half2 a) { return (*this = *this * a); }

	// scalar multiplication
	inline __device__
    half2 operator*(half a) { return *this * half2(cuda_fp16::__half2half2(a)); }

	inline __device__
    half2 operator*=(half a) { return (*this = *this * a); }

	// element-wise division
	inline __device__
    half2 operator/(half2 a) { return half2(cuda_fp16::__hmul2(*(half2 *)&x, cuda_fp16::h2rcp(*(half2 *)&(a.x)))); }

	inline __device__
    half2 operator/=(half2 a) { return (*this = *this / a); }

	// scalar division
	inline __device__
    half2 operator/(half a) { return *this / half2(cuda_fp16::__half2half2(a)); }

	inline __device__
    half2 operator/=(half a) { return (*this = *this / a); }

//	// lerp
//	inline __device__ float2 lerp(float2 a, float2 b, float t)
//	{
//	    return a + t*(b-a);
//	}
//
//	// clamp
//	inline __device__ float2 clamp(float2 v, float a, float b)
//	{
//	    return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
//	}
//
//	inline __device__ float2 clamp(float2 v, float2 a, float2 b)
//	{
//	    return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
//	}
//
//	// dot product
//	inline __device__ float dot(float2 a, float2 b)
//	{
//	    return a.x * b.x + a.y * b.y;
//	}
//
//	// length
//	inline __device__ float length(float2 v)
//	{
//	    return sqrtf(dot(v, v));
//	}
//
//	// normalize
//	inline __device__ float2 normalize(float2 v)
//	{
//	    float invLen = rsqrtf(dot(v, v));
//	    return v * invLen;
//	}
//
//	// floor
//	inline __device__ float2 floor(const float2 v)
//	{
//	    return make_float2(floor(v.x), floor(v.y));
//	}
//
//	// reflect
//	inline __device__ float2 reflect(float2 i, float2 n)
//	{
//		return i - 2.0f * n * dot(n,i);
//	}
//
//	// absolute value
//	inline __device__ float2 fabs(float2 v)
//	{
//		return make_float2(fabs(v.x), fabs(v.y));
//	}
};

#endif

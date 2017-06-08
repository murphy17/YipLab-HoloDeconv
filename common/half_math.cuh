/*
 * cuda_half.hpp
 *
 * Establish common interface between half and float types.
 * haven't defined int stuff
 * ... these ops aren't defined for normal float2 ...
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

// maybe get rid of assignment ops from native types? can replace with constructor wherever needed?
// scalar left-multiply/right-multiply not defined?

// I dislike this file a lot... class wrapping is suspect... but if it's not broken don't fix it

class half;
class half2;

class half
{
private:
	cuda_fp16::half _;

public:
	// constructors
	inline __device__ half() { ; }
	inline __device__ half(cuda_fp16::half a) { _ = a; }
	inline __device__ half(float a) { _ = cuda_fp16::__float2half(a); }
	// conversion
	inline __device__ operator cuda_fp16::half() const { return _; }
	inline __device__ operator float() const { return cuda_fp16::__half2float(_); }
	// addition
	inline __device__ half operator+(half a) { return cuda_fp16::__hadd(*this, a); }
	inline __device__ half operator+=(half a) { return (*this = *this + a); }
	// negation
	inline __device__ half operator-() { return half(cuda_fp16::__hneg(*this)); }
	// subtraction
	inline __device__ half operator-(half a) { return cuda_fp16::__hsub(*this, a); }
	inline __device__ half operator-=(half a) { return (*this = *this - a); }
	// multiplication
	inline __device__ half operator*(half a) { return cuda_fp16::__hmul(*this, a); }
	inline __device__ half operator*=(half a) { return (*this = *this * a); }
	inline __device__ half2 operator*(half2);
	// division
	inline __device__ half operator/(half a) { return cuda_fp16::__hmul(*this, cuda_fp16::hrcp(a)); }
	inline __device__ half operator/=(half a) { return (*this = *this / a); }
	inline __device__ half2 operator/(half2);
};

class half2
{
public:
	half x, y; // note the high 4 bytes are y, low 4 bytes are x

	// constructors
	inline __device__ half2() { ; }
	inline __device__ half2(cuda_fp16::half a) { *(cuda_fp16::half2 *)&x = cuda_fp16::__half2half2(a); }
	inline __device__ half2(half a) { *(cuda_fp16::half2 *)&x = cuda_fp16::__half2half2(a); }
	inline __device__ half2(float a) { *(cuda_fp16::half2 *)&x = cuda_fp16::__float2half2_rn(a); }
	inline __device__ half2(cuda_fp16::half a, cuda_fp16::half b) { x = a; y = b; }
	inline __device__ half2(cuda_fp16::half2 a) { *(cuda_fp16::half2 *)&x = a; }
	inline __device__ half2(float a, float b) { *(cuda_fp16::half2 *)&x = cuda_fp16::__floats2half2_rn(a, b); }
	// conversion
	inline __device__ operator cuda_fp16::half2() const { return *(cuda_fp16::half2 *)&x; }
	inline __device__ operator float2() const { return cuda_fp16::__half22float2(*(cuda_fp16::half2 *)&x); }
	// addition
	inline __device__ half2 operator+(half2 a) { return cuda_fp16::__hadd2(*(cuda_fp16::half2 *)&x,
																		   *(cuda_fp16::half2 *)&(a.x)); }
	inline __device__ half2 operator+=(half2 a) { return (*this = *this + a); }
	// negation
	inline __device__ half2 operator-() { return cuda_fp16::__hneg2(*(cuda_fp16::half2 *)&x); }
	// subtraction
	inline __device__ half2 operator-(half2 a) { return cuda_fp16::__hsub2(*(cuda_fp16::half2 *)&x,
																		   *(cuda_fp16::half2 *)&(a.x)); }
	inline __device__ half2 operator-=(half2 a) { return (*this = *this - a); }
	// element-wise multiplication
	inline __device__ half2 operator*(half2 a) { return cuda_fp16::__hmul2(*(cuda_fp16::half2 *)&x,
																		   *(cuda_fp16::half2 *)&(a.x)); }
	inline __device__ half2 operator*=(half2 a) { return (*this = *this * a); }
	// scalar multiplication
	inline __device__ half2 operator*(half a) { return *this * cuda_fp16::__half2half2(a); }
	inline __device__ half2 operator*=(half a) { return (*this = *this * a); }
	// element-wise division
	inline __device__ half2 operator/(half2 a) { return cuda_fp16::__hmul2(*(half2 *)&x,
																		   cuda_fp16::h2rcp(*(half2 *)&(a.x))); }
	inline __device__ half2 operator/=(half2 a) { return (*this = *this / a); }
	// scalar division
	inline __device__ half2 operator/(half a) { return *this / cuda_fp16::__half2half2(a); }
	inline __device__ half2 operator/=(half a) { return (*this = *this / a); }
};

inline __device__ half2 operator*(half &a, half2 &b) { return b * a; }
inline __device__ half2 operator/(half &a, half2 &b) { return half2(a) / b; }

// math methods
inline __device__ half sqrt(half a) { return cuda_fp16::hsqrt(a); }
inline __device__ half2 sqrt(half2 a) { return cuda_fp16::h2sqrt(a); };
inline __device__ half2 flip(half2 a) { return cuda_fp16::__lowhigh2highlow(a); };
inline __device__ half2 conj(half2 a) { half2 b(a); *(int *)&b.x ^= 1 << 31; return b; }
inline __device__ half2 cmul(half2 a, half2 b) { return cuda_fp16::__hfma2(b, cuda_fp16::__low2half2(a),
																		   flip(conj(b))*cuda_fp16::__high2half2(a)); }

#endif

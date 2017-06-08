/*
 * half_math.hpp
 *
 * Establish common interface between half and float types.
 * haven't defined int stuff
 * ... these ops aren't defined for normal float2 ...
 *
 *  Created on: May 30, 2017
 *      Author: michaelmurphy
 */

#ifndef half_math
#define half_math

namespace cuda_fp16 {
#include <cuda_fp16.h>
}

// half operations

class half
{
private:
	cuda_fp16::half _;

public:
	inline __device__ half() { ; }

	// transparently convert with undecorated class
	inline __device__
	half(cuda_fp16::half a) { _ = a; }

	inline __device__
	operator cuda_fp16::half() const { return _; }

	// 32-bit conversions
	inline __device__
	explicit half(float a) { _ = cuda_fp16::__float2half(a); }

	inline __device__
	operator float() const { return cuda_fp16::__half2float(_); }
};

// addition
inline __device__
half operator+(half a, half b) { return cuda_fp16::__hadd(a, b); }

inline __device__
half operator+=(half a, half b) { return a + b; }

// negation
inline __device__
half operator-(half a) { return cuda_fp16::__hneg(a); }

// subtraction
inline __device__
half operator-(half a, half b) { return cuda_fp16::__hsub(a, b); }

inline __device__
half operator-=(half a, half b) { return a - b; }

// multiplication
inline __device__
half operator*(half a, half b) { return cuda_fp16::__hmul(a, b); }

inline __device__
half operator*=(half a, half b) { return a * b; }

// division
inline __device__
half operator/(half a, half b) { return cuda_fp16::__hmul(a, cuda_fp16::hrcp(b)); }

inline __device__
half operator/=(half a, half b) { return a / b; }

// square root
inline __device__
half sqrt(half a) { return cuda_fp16::hsqrt(a); };

// half2 operations

union half2
{
private:
	cuda_fp16::half2 _;

public:
	// expose halves
	// don't know if this will use intrinsics, or if that matters
	struct {
		cuda_fp16::half x, y;
	};

	inline __device__ half2() { ; }

	inline __device__
	half2(half a, half b) {
		x = a;
		y = b;
	}

	// transparently convert with undecorated class
	inline __device__
	half2(cuda_fp16::half2 a) { _ = a; }

	inline __device__
	operator cuda_fp16::half2() const { return _; }

	// 32-bit conversions
	inline __device__
	explicit half2(float2 a) { _ = cuda_fp16::__float22half2_rn(a); }

	inline __device__
	operator float2() const { return cuda_fp16::__half22float2(_); }

	// broadcast
	inline __device__
	half2(half a) { _ = cuda_fp16::__half2half2(a); }
};

// addition
inline __device__
half2 operator+(half2 a, half2 b) { return cuda_fp16::__hadd2(a, b); }

inline __device__
half2 operator+=(half2 a, half2 b) { return a + b; }

// negation
inline __device__
half2 operator-(half2 a) { return cuda_fp16::__hneg2(a); }

// conjugation
inline __device__
half2 conj(half2 a) {
	half2 b = a;
	*(int *)&b ^= 1 << 31;
	return b;
}

// subtraction
inline __device__
half2 operator-(half2 a, half2 b) { return cuda_fp16::__hsub2(a, b); }

inline __device__
half2 operator-=(half2 a, half2 b) { return a - b; }

// element-wise multiplication
inline __device__
half2 operator*(half2 a, half2 b) { return cuda_fp16::__hmul2(a, b); }

inline __device__
half2 operator*=(half2 a, half2 b) { return a * b; }

// scalar multiplication
inline __device__
half2 operator*(half2 a, half b) { return a * cuda_fp16::__half2half2(b); }

inline __device__
half2 operator*(half a, half2 b) { return b * cuda_fp16::__half2half2(a); }

inline __device__
half2 operator*=(half2 a, half b) { return a * b; }

// complex multiplication
inline __device__
half2 cmul(half2 a, half2 b) {
	return cuda_fp16::__hfma2(b, cuda_fp16::__low2half2(a),
							  cuda_fp16::__lowhigh2highlow(conj(b)) * cuda_fp16::__high2half2(a));
};

// element-wise division
inline __device__
half2 operator/(half2 a, half2 b) { return cuda_fp16::__hmul2(a, cuda_fp16::h2rcp(b)); }

inline __device__
half2 operator/=(half2 a, half2 b) { return a / b; }

// scalar division
inline __device__
half2 operator/(half2 a, half b) { return a / cuda_fp16::__half2half2(b); }

// scalar division
inline __device__
half2 operator/(half a, half2 b) { return cuda_fp16::__half2half2(a) / b; }

inline __device__
half2 operator/=(half2 a, half b) { return a / b; }

// square root
inline __device__
half2 sqrt(half2 a) { return cuda_fp16::h2sqrt(a); };

// complex modulus
inline __device__
half mod(half2 a) {
	half2 c = a * a;
	return sqrt(cuda_fp16::__low2half(a) + cuda_fp16::__high2half(a));
}

#endif

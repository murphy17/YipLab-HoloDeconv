
/*
 *
 * Proof-of-concept for GPU holographic deconvolution.
 * Michael Murphy, May 2017
 * Yip Lab
 *
 */

// FFT batching
// async memory transfer
// half-precision
// texture memory

// microoptimisation: replace x * N with x << LOG2N

// should make some wrappers for the kernels, would make some decisions clearer

// This fails to run with 5.2 CC...

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
//#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <algorithm>
#include <cuda_fp16.h>

#include "half.hpp"
#include "common.h"

#define N 1024
#define LOG2N 10
#define DX (5.32f / 1024.f)
#define DY (6.66f / 1280.f)
#define LAMBDA0 0.000488f
#define SCALE 0.00097751711f // 1/(N-1)

typedef unsigned char byte;

// Convenience method for plotting
void imshow(cv::Mat in)
{
	cv::namedWindow("Display window", cv::WINDOW_NORMAL); // Create a window for display.
	cv::Mat out;
	cudaDeviceSynchronize();
	in.convertTo(out, CV_32FC1);
	cv::normalize(out, out, 1.0, 0.0, cv::NORM_MINMAX, -1);
	cv::imshow("Display window", out); // Show our image inside it.
	cv::waitKey(0);
}
void imshow(cv::gpu::GpuMat in)
{
	cv::namedWindow("Display window", cv::WINDOW_NORMAL); // Create a window for display.
	cv::Mat out;
	cudaDeviceSynchronize();
	in.download(out);
	if (out.channels() == 2)
	{
		cv::Mat channels[2];
		cv::split(out, channels);
		cv::magnitude(channels[0], channels[1  ], out);
	}
	out.convertTo(out, CV_32FC1);
	cv::normalize(out, out, 1.0, 0.0, cv::NORM_MINMAX, -1);
	cv::imshow("Display window", out); // Show our image inside it.
	cv::waitKey(0);
}

// Kernel to construct the point-spread function at distance z.
// exploits 4-fold symmetry (PSF is also radially symmetric, but that's harder...)
// note that answer is scaled between +/-1
//__global__
//void construct_psf(float z, cufftComplex *g, float norm)
//{
//	const int i = blockIdx.x;
//	const int j = threadIdx.x; // blockDim shall equal N
//
//	const int ii = (N - 1) - i;
//	const int jj = (N - 1) - j;
//
//	// not sure whether the expansion of N/(N-1) was necessary
//	float x = (i * SCALE + i - N/2) * DX;
//	float y = (j * SCALE + j - N/2) * DY;
//
//	// could omit negation here, symmetries of trig functions take care of it
//	float r = (-2.f / LAMBDA0) * norm3df(x, y, z);
//
//	// exp(ix) = cos(x) + isin(x)
//	float re, im;
//	sincospif(r, &im, &re);
//
//	// numerical conditioning, important for half-precision FFT
//	// also corrects the sign flip above
//	r = __fdividef(r, norm); // norm = -2.f * z / LAMBDA0
//
//	// re(iz) = -im(z), im(iz) = re(z)
//	cufftComplex g_ij;
//	g_ij.x = __fdividef(-im, r); // im, r);
//	g_ij.y = __fdividef(re, r);
//
//	// CUDA takes care of coalescing the reversed access, this is fine
//	g[i*N+j] = g_ij;
//	g[i*N+jj] = g_ij;
//	g[ii*N+j] = g_ij;
//	g[ii*N+jj] = g_ij;
//}
__global__
void construct_psf(float z, half *g_re, half *g_im, float norm)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	int ii = (N - 1) - i;
	int jj = (N - 1) - j;

	// not sure whether the expansion of N/(N-1) was necessary
	float x = (i * SCALE + i - N/2) * DX;
	float y = (j * SCALE + j - N/2) * DY;

	// could omit negation here, symmetries of trig functions take care of it
	float r = (-2.f / LAMBDA0) * norm3df(x, y, z);

	// exp(ix) = cos(x) + isin(x)
	float re, im;
	sincospif(r, &im, &re);

	// numerical conditioning, important for half-precision FFT
	// also corrects the sign flip above
	r = __fdividef(r, norm); // norm = -2.f * z / LAMBDA0

	// re(iz) = -im(z), im(iz) = re(z)
	cufftComplex g_ij;
	g_ij.x = __fdividef(-im, r); // im, r);
	g_ij.y = __fdividef(re, r);

	// cast to half-precision
	half g_ij_re = __float2half(g_ij.x);
	half g_ij_im = __float2half(g_ij.y);

	// I'm really skeptical about the memory access here - each seems around 2.5ms
	// but when I tried shared memory it was slower
	g_re[i*N+j] = g_ij_re;
	g_re[i*N+jj] = g_ij_re;
	g_re[ii*N+j] = g_ij_re;
	g_re[ii*N+jj] = g_ij_re;
	g_im[i*N+j] = g_ij_im;
	g_im[i*N+jj] = g_ij_im;
	g_im[ii*N+j] = g_ij_im;
	g_im[ii*N+jj] = g_ij_im;
}

// exploit Fourier duality to shift without copying
// credit to http://www.orangeowlsolutions.com/archives/251
__global__
//void frequency_shift(cufftComplex *data)
//{
//    const int i = blockIdx.x;
//    const int j = threadIdx.x;
//
//	const float a = 1 - 2 * ((i+j) & 1); // this looks like a checkerboard?
//
//	data[i*N+j].x *= a;
//	data[i*N+j].y *= a;
//}

__global__
void invert_phase(float2 *data)
{
    int i = blockIdx.x;
    int j = threadIdx.x;

	float a = 1 - 2 * ((i+j) & 1); // this looks like a checkerboard?

	data[i*N+j].x *= a;
	data[i*N+j].y *= a;
}

__device__ __forceinline__
void _mul(void *dataOut, size_t offset, cufftComplex a, void *callerInfo, void *sharedPtr)
{
	float bx = ((cufftComplex *)callerInfo)[offset].x;
	float by = ((cufftComplex *)callerInfo)[offset].y;

	float a_temp = a.y;
	float ay_by = __fmul_rn(a_temp, by);
	float ay_bx = __fmul_rn(a_temp, bx);
	a_temp = a.x;
	bx = __fmaf_rn(a_temp, bx, -ay_by);
	by = __fmaf_rn(a_temp, by, ay_bx);

	((cufftComplex *)dataOut)[offset].x = bx;
	((cufftComplex *)dataOut)[offset].y = by;
}
__device__
cufftCallbackStoreC d_mul = _mul;

__global__
void byte_to_complex(byte *b, cufftComplex *z)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x; // blockDim shall equal N

	z[i*N+j].x = ((float)(b[i*N+j])) / 255.f;
	z[i*N+j].y = 0.f;
}

__global__
void byte_to_half2(byte *b, half2 *z)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	z[i*N+j] = __floats2half2_rn(((float)(b[i*N+j])) / 255.f, 0.f);
}

__global__
void complex_to_half2(cufftComplex *z, half2 *h)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	h[i*N+j] = __float22half2_rn(z[i*N+j]);
}

__global__
void half2_to_complex(half2 *h, cufftComplex *z)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	z[i*N+j] = __half22float2(h[i*N+j]);
}

__global__
void normalize_by(half *h, float n)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	h[i*N+j] = __hmul(h[i*N+j], __float2half(1.f / n));
}

__global__
void normalize_by(cufftComplex *z, float n)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	z[i*N+j].x /= n;
	z[i*N+j].y /= n;
}

__global__
void normalize_by(half2 *h, float n)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	h[i*N+j] = __hmul2(h[i*N+j], __float2half2_rn(1.f / n));
}

__global__
void complex_modulus(cufftComplex *z, float *r)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x; // blockDim shall equal N

	r[i*N+j] = hypotf(z[i*N+j].x, z[i*N+j].y);
}

__global__
void batch_multiply(half2 *z, const __restrict__ half2 *w)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x; // blockDim shall equal N

	half2 w_ij = w[i*N+j];
	half2 w_ij_inv = __lowhigh2highlow(w_ij);

	half2 z_ij = z[i*N+j];

	// z = a + ib, w = c + id
	half2 ac_bd = __hmul2(z_ij, w_ij);
	half re = __hsub(__low2half(ac_bd), __high2half(ac_bd));

	half2 ad_bc = __hmul2(z_ij, w_ij_inv);
	half im = __hadd(__low2half(ad_bc), __high2half(ad_bc));

	// transpose
//		half2 ac_ad = __highs2half2(ac_bd, __hneg2(ad_bc));
//		half2 bd_bc = __lows2half2(ac_bd, ad_bc);

	z[i*N+j] = __halves2half2(re, im);
//		z[i*N+j] = __hadd2(ac_ad, bd_bc);

	z += N*N;
}

__global__
void merge_filter_halves(half2 *x_f, half2 *y_f, half2 *z_f)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x;

	// x = a+ib, y = c+id, z = x+iy
	// re(z) = re(x+iy) = re(a+ib+ic-d) = a-d
	// im(z) = im(x+iy) = im(a+ib+ic-d) = b+c

	half a = __low2half(x_f[i*N+j]);
	half b = __high2half(x_f[i*N+j]);
	half c = __low2half(y_f[i*N+j]);
	half d = __high2half(y_f[i*N+j]);

	z_f[i*N+j] = __halves2half2(__hsub(a, d), __hadd(b, c));
}

__global__
void byte_to_float(byte *b, float *z)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	z[i*N+j] = ((float)(b[i*N+j])) / 255.f;
}

__global__
void half_to_float(half *h, float *f)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	f[i*N+j] = __half2float(h[i*N+j]);
}

__global__
void byte_to_half(byte *b, half *h)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	h[i*N+j] = __float2half(((float)(b[i*N+j])) / 255.f);
}

int main(int argc, char* argv[])
{
	checkCudaErrors( cudaDeviceReset() );

	long long dims[] = {N, N};
	size_t work_sizes = 0;

	half *img;
	checkCudaErrors( cudaMalloc((void **)&img, N*N*sizeof(half)) );

	byte *img_u8;
	checkCudaErrors( cudaMalloc((void **)&img_u8, N*N*sizeof(byte)) );

	float *img_f32;
	checkCudaErrors( cudaMalloc((void **)&img_f32, N*N*sizeof(float)) );

	cv::Mat A = cv::imread("test_square.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	checkCudaErrors( cudaMemcpy(img_u8, A.data, N*N*sizeof(byte), cudaMemcpyHostToDevice) );

	byte_to_half<<<N, N>>>(img_u8, img);
	normalize_by<<<N, N>>>(img, N); // cast to half2, use half the threads

	half2 *img_f;
	checkCudaErrors( cudaMalloc((void **)&img_f, N*(N/2+1)*sizeof(half2)) );
	normalize_by<<<N/2+1, N>>>(img_f, N);

	cufftHandle plan_r2c;
	cufftCreate(&plan_r2c);
	checkCudaErrors( cufftXtMakePlanMany(plan_r2c, 2, dims, \
			NULL, 1, 0, CUDA_R_16F, \
			NULL, 1, 0, CUDA_C_16F, \
			1, &work_sizes, CUDA_C_16F) );
	checkCudaErrors( cufftXtExec(plan_r2c, img, img_f, CUFFT_FORWARD) );

	// one problem - this trick breaks, not sure how to fix...
	// ugly trick, but you can just FFT-shift the PSF as you generate it
	// the Fourier ID is so slick though...
//	invert_phase<<<N/2+1, N>>>(img_f);

	half *psf_re, *psf_im;
	checkCudaErrors( cudaMalloc((void **)&psf_re, N*N*sizeof(half)) );
	checkCudaErrors( cudaMalloc((void **)&psf_im, N*N*sizeof(half)) );

	float z = 50;
	construct_psf<<<N/2, N/2>>>(z, psf_re, psf_im, -2.f * z / LAMBDA0 / N);

	half2 *psf_re_f, *psf_im_f;
	checkCudaErrors( cudaMalloc((void **)&psf_re_f, N*(N/2+1)*sizeof(half2)) );
	checkCudaErrors( cudaMalloc((void **)&psf_im_f, N*(N/2+1)*sizeof(half2)) );

	checkCudaErrors( cufftXtExec(plan_r2c, psf_re, psf_re_f, CUFFT_FORWARD) );
	checkCudaErrors( cufftXtExec(plan_r2c, psf_im, psf_im_f, CUFFT_FORWARD) );

	half2 *psf_f;
	checkCudaErrors( cudaMalloc((void **)&psf_f, N*(N/2+1)*sizeof(half2)) );

	// kinda weird allocation, think of how to deal with
	merge_filter_halves<<<N/2+1, N>>>(psf_re_f, psf_im_f, psf_f);

	batch_multiply<<<N/2+1, N>>>(img_f, psf_f);

	cufftHandle plan_c2r;
	cufftCreate(&plan_c2r);
	checkCudaErrors( cufftXtMakePlanMany(plan_c2r, 2, dims, \
			NULL, 1, 0, CUDA_C_16F, \
			NULL, 1, 0, CUDA_R_16F, \
			1, &work_sizes, CUDA_C_16F) );
	checkCudaErrors( cufftXtExec(plan_c2r, img_f, img, CUFFT_INVERSE) );

	half_to_float<<<N, N>>>(img, img_f32);

	imshow(cv::gpu::GpuMat(N, N, CV_32FC1, img_f32));

//
////	frequency_shift<<<N, N>>>(d_img_f16);
//
//	half *psf_re, *psf_im;
//	checkCudaErrors( cudaMalloc((void **)&psf_re, N*N*sizeof(half)) );
//	checkCudaErrors( cudaMalloc((void **)&psf_im, N*N*sizeof(half)) );
//
//	float z = 50;
//	construct_psf<<<N/2, N/2>>>(z, psf_re, psf_im, -2.f * z / LAMBDA0 / N);
//
//	half2 *psf_re_f, *psf_im_f;
//	checkCudaErrors( cudaMalloc((void **)&psf_re_f, N*(N/2+1)*sizeof(half2)) );
//	checkCudaErrors( cudaMalloc((void **)&psf_im_f, N*(N/2+1)*sizeof(half2)) );
//
//	checkCudaErrors( cufftXtExec(plan_r2c, psf_re, psf_re_f, CUFFT_FORWARD) );
//	checkCudaErrors( cufftXtExec(plan_r2c, psf_im, psf_im_f, CUFFT_FORWARD) );
//
//	half2 *psf;
//	checkCudaErrors( cudaMalloc((void **)&psf, N*(N/2+1)*sizeof(half2)) );
//
//	// kinda weird allocation, think of how to deal with
//	merge_filter_halves<<<N/2+1, N>>>(psf_re_f, psf_im_f, psf);
//
//	multiply_filter<<<N/2+1, N>>>(img_f, psf);
//
//	half2 *img;
//	checkCudaErrors( cudaMalloc((void **)&img, N*N*sizeof(half2)) );
//
//	cufftHandle plan_c2r;
//	cufftCreate(&plan_c2r);
//	checkCudaErrors( cufftXtMakePlanMany(plan_c2r, 2, dims, \
//			NULL, 1, 0, CUDA_C_16F, \
//			NULL, 1, 0, CUDA_R_16F, \
//			1, &work_sizes, CUDA_C_16F) );
//
//	normalize_by<<<N/2+1, N>>>(img_f, N);
//	checkCudaErrors( cufftXtExec(plan_c2r, img_f, img, CUFFT_INVERSE) );
//
//	half2_to_complex<<<N, N>>>(img, d_img);
//
//	imshow(cv::gpu::GpuMat(N, N, CV_32FC2, d_img));

	return 0;
}

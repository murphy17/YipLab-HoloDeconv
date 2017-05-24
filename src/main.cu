
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
//#include <opencv2/gpu/gpu.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <algorithm>
#include <cuda_fp16.h>

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
void imshow(cv::cuda::GpuMat in)
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
__global__
void construct_psf(float z, cufftComplex *g, float norm)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x; // blockDim shall equal N

	const int ii = (N - 1) - i;
	const int jj = (N - 1) - j;

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

	// CUDA takes care of coalescing the reversed access, this is fine
	g[i*N+j] = g_ij;
	g[i*N+jj] = g_ij;
	g[ii*N+j] = g_ij;
	g[ii*N+jj] = g_ij;
}

// exploit Fourier duality to shift without copying
// credit to http://www.orangeowlsolutions.com/archives/251
__global__
void frequency_shift(cufftComplex *data)
{
    const int i = blockIdx.x;
    const int j = threadIdx.x;

	const float a = 1 - 2 * ((i+j) & 1); // this looks like a checkerboard?

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

int main(int argc, char* argv[])
{
	checkCudaErrors( cudaDeviceReset() );

	long long dims[] = {N, N};
	size_t work_sizes = 0;
	cufftHandle plan, plan_mul;
	cufftCreate(&plan);
	cufftCreate(&plan_mul);
	checkCudaErrors( cufftXtMakePlanMany(plan, 2, dims, \
			NULL, 1, 0, CUDA_C_16F, \
			NULL, 1, 0, CUDA_C_16F, \
			1, &work_sizes, CUDA_C_16F) );

	cufftComplex *d_img;
	checkCudaErrors( cudaMalloc((void **)&d_img, N*N*sizeof(cufftComplex)) );

	byte *d_img_u8;
	checkCudaErrors( cudaMalloc((void **)&d_img_u8, N*N*sizeof(byte)) );

	half2 *d_img_f16;
	checkCudaErrors( cudaMalloc((void **)&d_img_f16, N*N*sizeof(half2)) );

	cv::Mat A = cv::imread("test_square.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	checkCudaErrors( cudaMemcpy(d_img_u8, A.data, N*N*sizeof(byte), cudaMemcpyHostToDevice) );

	// from symmetry alone, 1/N would make sense... (sqrt of N*N)

	// need to figure out normalization here
	// normalize image to [0,1] ... max pixel value is 1
	// min nonzero pixel value is 1/255
	// |Fourier coefficients| <= N*N

	byte_to_complex<<<N, N>>>(d_img_u8, d_img);
	complex_to_half2<<<N, N>>>(d_img, d_img_f16);

	// this works! division by N is dumb heuristic though
	// Parseval's theorem?
	normalize_by<<<N, N>>>(d_img_f16, N);

	checkCudaErrors( cufftXtExec(plan, d_img_f16, d_img_f16, CUFFT_FORWARD) );

	normalize_by<<<N, N>>>(d_img_f16, N);

	checkCudaErrors( cufftXtExec(plan, d_img_f16, d_img_f16, CUFFT_INVERSE) );

	half2_to_complex<<<N, N>>>(d_img_f16, d_img);

	imshow(cv::cuda::GpuMat(N, N, CV_32FC2, d_img));

	return 0;
}

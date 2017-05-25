
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

#include "common.h"

#define N 1024
#define LOG2N 10
#define DX (5.32f / 1024.f)
#define DY (6.66f / 1280.f)
#define LAMBDA0 0.000488f
#define SCALE 0.00097751711f // 1/(N-1)
#define NUM_SLICES 128
#define MAX_BLOCK_THREADS 1024

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

	// CUDA coalescing can deal with the reversed indices, this is fine
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
    int i = blockIdx.x;
    int j = threadIdx.x;

	float a = 1 - 2 * ((i+j) & 1); // this looks like a checkerboard?

	data[i*N+j].x *= a;
	data[i*N+j].y *= a;
}

// it seems you can't have too many plans simultaneously.
// workaround: conditionals in the callback?
// ... I tried this. much slower. thought branching was killing performance
// which doesn't make sense, all threads take same path
// it wasn't, which is good, sort of... turns out the *struct* was the issue

__device__
void _mul(void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
	cufftComplex a, b, c;

	a = element;
	// offset & ((1 << 20) - 1) = offset % (1024*1024); former caused a VERY BIG slowdown
	b = ((cufftComplex *)callerInfo)[offset & ((1 << 20) - 1)];

	// don't use intrinsics here, this is fastest
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;

	((cufftComplex *)dataOut)[offset] = c;
}
__device__
cufftCallbackStoreC d_mul = _mul;

__global__
void byte_to_complex(byte *b, cufftComplex *z)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	z[i*N+j].x = ((float)(b[i*N+j])) / 255.f;
	z[i*N+j].y = 0.f;
}

__global__
void complex_modulus(cufftComplex *z, float *r)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	r[i*N+j] = hypotf(z[i*N+j].x, z[i*N+j].y);
}


//dim3 grid_dims(N*N / (MAX_BLOCK_THREADS / NUM_SLICES));
//dim3 block_dims(MAX_BLOCK_THREADS / NUM_SLICES, NUM_SLICES);
//batch_multiply<<<grid_dims, block_dims, 0, math_stream>>>(d_psf, d_img);
__global__
void batch_multiply(cufftComplex *z, const __restrict__ cufftComplex *w)
{
	// threadIdx.x = slice index
	// threadIdx.y = element index

	// each thread block processes blockDims.x different elements of w
	__shared__ cufftComplex cache[MAX_BLOCK_THREADS / NUM_SLICES];

	int inIdx = blockIdx.x * blockDim.x + threadIdx.x; // blockDim.x  (MAX_BLOCK_THREADS / NUM_SLICES)
	int outIdx = threadIdx.y * (N*N) + inIdx;
//	int outIdx = threadIdx.y + inIdx * NUM_SLICES; // same elements in successive slices adjacent in memory

	if (threadIdx.y == 0)
	{
		cache[threadIdx.x] = w[inIdx];
	}
	__syncthreads();

	cufftComplex a = z[outIdx];
	float a_temp = a.y;
	float ay_by = __fmul_rn(a_temp, cache[threadIdx.x].y);
	float ay_bx = __fmul_rn(a_temp, cache[threadIdx.x].x);
	a_temp = a.x;

	z[outIdx].x = __fmaf_rn(a_temp, cache[threadIdx.x].x, -ay_by);
	z[outIdx].y = __fmaf_rn(a_temp, cache[threadIdx.x].y, ay_bx);
}

//__global__
//void batch_multiply(cufftComplex *z, const __restrict__ cufftComplex *w)
//{
//	const int i = blockIdx.x;
//	const int j = threadIdx.x;
//
//	cufftComplex b = w[i*N+j];
//	float bx = b.x; float by = b.y; // just making sure
//
//	for (int k = 0; k < NUM_SLICES; k++)
//	{
//		cufftComplex a = z[i*N+j];
//
//		// this gives like 3% speedup
////		asm(".reg .f32 ay_by;\n" // float ay_by
////			".reg .f32 ay_bx;\n" // float ay_bx
////			" mul .f32 ay_by, %2, %0;\n" // ay_by = __fmul_rn(ay, by)
////			" mul .f32 ay_bx, %2, %1;\n" // ay_bx = __fmul_rn(ay, bx)
////			" neg .f32 ay_by, ay_by;\n" // ay_by = -ay_by
////			" fma.rn .f32 %1, %3, %1, ay_by;\n" // bx = __fmaf_rn(ax, bx, ay_by);
////			" fma.rn .f32 %0, %3, %0, ay_bx;\n" : \
////			"+f"(by), "+f"(bx) : \
////			"f"(a.y), "f"(a.x)); // by = __fmaf_rn(ax, by, ay_bx);
////		z[i*N+j].x = bx;
////		z[i*N+j].y = by;
//
//		float a_temp = a.y;
//		float ay_by = __fmul_rn(a_temp, by);
//		float ay_bx = __fmul_rn(a_temp, bx);
//		a_temp = a.x;
//		z[i*N+j].x = __fmaf_rn(a_temp, bx, -ay_by);
//		z[i*N+j].y = __fmaf_rn(a_temp, by, ay_bx);
//
//		z += N*N;
//	}
//}

int main(int argc, char *argv[])
{
	checkCudaErrors( cudaDeviceReset() );

	int num_frames = 10;
	float z_min = 30;
	float z_step = 1;

	cudaStream_t math_stream, copy_stream;
	checkCudaErrors( cudaStreamCreate(&math_stream) );
	checkCudaErrors( cudaStreamCreate(&copy_stream) );

	long long dims[] = {N, N};
	size_t work_sizes = 0;
	cufftHandle plan, plan_mul, plan_img;
	cufftCreate(&plan);
//	cufftCreate(&plan_mul);
	cufftCreate(&plan_img);
	checkCudaErrors( cufftXtMakePlanMany(plan, 2, dims, \
			NULL, 1, 0, CUDA_C_32F, \
			NULL, 1, 0, CUDA_C_32F, \
			NUM_SLICES, &work_sizes, CUDA_C_32F) );
//	checkCudaErrors( cufftXtMakePlanMany(plan_mul, 2, dims, \
//			NULL, 1, 0, CUDA_C_32F, \
//			NULL, 1, 0, CUDA_C_32F, \
//			NUM_SLICES, &work_sizes, CUDA_C_32F) );
	checkCudaErrors( cufftXtMakePlanMany(plan_img, 2, dims, \
			NULL, 1, 0, CUDA_C_32F, \
			NULL, 1, 0, CUDA_C_32F, \
			1, &work_sizes, CUDA_C_32F) );

	checkCudaErrors( cufftSetStream(plan, math_stream) );
//	checkCudaErrors( cufftSetStream(plan_mul, math_stream) );
	checkCudaErrors( cufftSetStream(plan_img, math_stream) );

	cufftComplex *d_img, *d_psf;
	checkCudaErrors( cudaMalloc((void **)&d_psf, NUM_SLICES*N*N*sizeof(cufftComplex)) );
	checkCudaErrors( cudaMalloc((void **)&d_img, N*N*sizeof(cufftComplex)) );

//	cufftCallbackStoreC h_mul;
//	checkCudaErrors( cudaMemcpyFromSymbol(&h_mul, d_mul, sizeof(cufftCallbackStoreC)) );
//	checkCudaErrors( cufftXtSetCallback(plan_mul, (void **)&h_mul, CUFFT_CB_ST_COMPLEX, (void **)&d_img) );

	byte *d_img_u8;
	checkCudaErrors( cudaMalloc((void **)&d_img_u8, N*N*sizeof(byte)) );

	// full-size is necessary for downstream reduction, need the whole cube
	float *d_slices;
	checkCudaErrors( cudaMalloc((void **)&d_slices, NUM_SLICES*N*N*sizeof(float)) );

	// wouldn't exist in streaming application
	float *h_slices;
	checkCudaErrors( cudaMallocHost((void **)&h_slices, NUM_SLICES*N*N*sizeof(float)) );

	// setup multiply kernel
	dim3 grid_dims(N*N / (MAX_BLOCK_THREADS / NUM_SLICES));
	dim3 block_dims(MAX_BLOCK_THREADS / NUM_SLICES, NUM_SLICES);

	for (int frame = 0; frame < num_frames; frame++)
	{
		// this would be a copy from a frame buffer on the Tegra
		cv::Mat A = cv::imread("test_square.bmp", CV_LOAD_IMAGE_GRAYSCALE);

		checkCudaErrors( cudaMemcpyAsync(d_img_u8, A.data, N*N*sizeof(byte), cudaMemcpyHostToDevice, math_stream) );

		byte_to_complex<<<N, N, 0, math_stream>>>(d_img_u8, d_img);

		checkCudaErrors( cufftExecC2C(plan_img, d_img, d_img, CUFFT_FORWARD) );
//		checkCudaErrors( cudaStreamSynchronize(math_stream) ); // reusing an async plan

		// this is subtle - shifting in conjugate domain means we don't need to FFT shift later
		frequency_shift<<<N, N, 0, math_stream>>>(d_img);

		// can pipeline now!

		// might as well just cache the PSF if you're gonna do this...
		for (int slice = 0; slice < NUM_SLICES; slice++)
		{
			float z = z_min + z_step * slice;
			// generate the PSF, weakly taking advantage of symmetry to speed up
			construct_psf<<<N/2, N/2, 0, math_stream>>>(z, d_psf + slice*N*N, -2.f * z / LAMBDA0);
		}

//		for (int slice = 0; slice < NUM_SLICES; slice++)
//		{
//			// FFT
//			checkCudaErrors( cufftExecC2C(plan, d_psf + slice*N*N, d_psf + slice*N*N, CUFFT_FORWARD) );
//		}
		checkCudaErrors( cufftExecC2C(plan, d_psf, d_psf, CUFFT_FORWARD) );

		batch_multiply<<<grid_dims, block_dims, 0, math_stream>>>(d_psf, d_img);

//		for (int slice = 0; slice < NUM_SLICES; slice++)
//		{
//			// inverse FFT
//			checkCudaErrors( cufftExecC2C(plan, d_psf + slice*N*N, d_psf + slice*N*N, CUFFT_INVERSE) );
//		}
		checkCudaErrors( cufftExecC2C(plan, d_psf, d_psf, CUFFT_INVERSE) );

		// for FFT shift would need to invert phase now, but it doesn't matter since we're taking modulus

		// for-looping outside this kernel is *much* faster
		checkCudaErrors( cudaStreamSynchronize(copy_stream) );
		for (int slice = 0; slice < NUM_SLICES; slice++)
		{
			complex_modulus<<<N, N, 0, math_stream>>>(d_psf + slice*N*N, d_slices + slice*N*N);
		}

		checkCudaErrors( cudaStreamSynchronize(math_stream) );
		checkCudaErrors( cudaMemcpyAsync(h_slices, d_slices, NUM_SLICES*N*N*sizeof(float), \
				cudaMemcpyDeviceToHost, copy_stream) );

		if (frame == 0)
			cudaTimerStart();
	}

	checkCudaErrors( cudaDeviceSynchronize() );

	std::cout << cudaTimerStop() / (num_frames-1) << "ms" << std::endl;

	if (argc == 2)
	{
		for (int slice = 0; slice < NUM_SLICES; slice++)
		{
			cv::Mat B(N, N, CV_32FC1, h_slices + N*N*slice);
			imshow(B);
		}
	}

	checkCudaErrors( cudaFree(d_img) );
	checkCudaErrors( cudaFree(d_img_u8) );
	checkCudaErrors( cudaFree(d_psf) );
	checkCudaErrors( cudaFree(d_slices) );
	checkCudaErrors( cudaFreeHost(h_slices) );

	checkCudaErrors( cufftDestroy(plan) );
//	checkCudaErrors( cufftDestroy(plan_mul) );
	checkCudaErrors( cufftDestroy(plan_img) );

	checkCudaErrors( cudaStreamDestroy(math_stream) );
	checkCudaErrors( cudaStreamDestroy(copy_stream) );

	return 0;
}

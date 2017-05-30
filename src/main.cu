
/*
 *
 * Proof-of-concept for GPU holographic deconvolution.
 * Michael Murphy, May 2017
 * Yip Lab
 *
 */

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <algorithm>

#include "cuda_common.h"
#include "cuda_half.hpp"
#include "half.hpp"
#include "util.hpp"

//#define FP32
#define FP16

#define N 1024
#define DX (5.32f / 1024.f)
#define DY (6.66f / 1280.f)
#define DZ 1.f
#define Z0 80 // 30
#define LAMBDA0 0.000488f
#define NUM_SLICES 10
#define NUM_FRAMES 3

#ifdef FP32
typedef cufftComplex complex;
typedef float real;
#endif
#ifdef FP16
typedef half2 complex;
typedef half real;
#endif

typedef unsigned char byte;

// Kernel to construct the point-spread function at distance z
template <typename T>
__global__
void construct_psf(float z, T *g, float norm)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x; // blockDim shall equal N

	// 'FFT-even symmetry' - periodic extension must be symmetric about (0,0)
	float x = (i - N/2) * DX;
	float y = (j - N/2) * DY;

	// could omit negation here, symmetries of trig functions take care of it
	float r = (-2.f / LAMBDA0) * norm3df(x, y, z);

	// exp(ix) = cos(x) + isin(x)
	float re, im;
	sincospif(r, &im, &re);

	// numerical conditioning, important for half-precision FFT
	// also corrects the sign flip above
	r = __fdividef(r, norm); // norm = -2.f * z / LAMBDA0

	// re(iz) = -im(z), im(iz) = re(z)
	g[i*N+j].x = __fdividef(-im, r);
	g[i*N+j].y = __fdividef(re, r);
}

// exploit Fourier duality to shift without copying
// credit to http://www.orangeowlsolutions.com/archives/251
template <class T>
__global__
void frequency_shift(T *data)
{
    const int i = blockIdx.x;
    const int j = threadIdx.x;

	const float a = 1 - 2 * ((i+j) & 1); // this looks like a checkerboard?

	data[i*N+j].x *= a;
	data[i*N+j].y *= a;
}

template <class T>
__device__ __forceinline__
T _mul(T a, T b)
{
	return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

template <class T>
__global__
void batch_multiply(T *z, const __restrict__ T *w)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x;

	T w_ij = w[i*N+j];

	for (int k = 0; k < NUM_SLICES; k++)
	{
		z[i*N+j] = _mul(z[i*N+j], w_ij);

		z += N*N;
	}
}

template <class T>
__global__
void quadrant_multiply(T *z, T *w) //, int i, int j)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x;
	const int ii = N-i;
	const int jj = N-j;

	// this saves 8 registers (does it still?)
	int cond = 0;
	if (i>0&&i<N/2) cond |= 1;
	if (j>0&&j<N/2) cond |= 2;

	T w_[4];
	w_[0] = w[i*N+j];
	if (cond & 1) w_[1] = w[ii*N+j];
	if (cond & 2) w_[2] = w[i*N+jj];
	if (cond == 3) w_[3] = w[ii*N+jj];

	T z_ij;

	// conditional unwrapping
	// this had no effect... but compiler didn't seem to be doing it?
	switch (cond)
	{
		case 3:
		for (int k = 0; k < NUM_SLICES; k++)
		{
			z_ij = z[i*N+j];
			z[i*N+j] = _mul(w_[0], z_ij);
			z[ii*N+j] = _mul(w_[1], z_ij);
			z[i*N+jj] = _mul(w_[2], z_ij);
			z[ii*N+jj] = _mul(w_[3], z_ij);
			z += N*N;
		}
		break;

		case 2:
		for (int k = 0; k < NUM_SLICES; k++)
		{
			z_ij = z[i*N+j];
			z[i*N+j] = _mul(w_[0], z_ij);
			z[i*N+jj] = _mul(w_[2], z_ij);
			z += N*N;
		}
		break;

		case 1:
		for (int k = 0; k < NUM_SLICES; k++)
		{
			z_ij = z[i*N+j];
			z[i*N+j] = _mul(w_[0], z_ij);
			z[ii*N+j] = _mul(w_[1], z_ij);
			z += N*N;
		}
		break;

		case 0:
		for (int k = 0; k < NUM_SLICES; k++)
		{
			z_ij = z[i*N+j];
			z[i*N+j] = _mul(w_[0], z_ij);
			z += N*N;
		}
		break;
	}
}

//__global__
//void quadrant_multiply(complex *z, const __restrict__ complex *w)
//{
//	const int i = blockIdx.x;
//	const int j = threadIdx.x;
//
//	// permits using nicely-sized kernel dimensions
//	_quadrant_multiply(z, w, i, j);
//	if (i == N/2-1) _quadrant_multiply(z, w, i+1, j);
//	if (j == N/2-1) _quadrant_multiply(z, w, i, j+1);s
//	if (i == N/2-1 && j == N/2-1) _quadrant_multiply(z, w, i+1, j+1);
//}

template<typename T>
__global__
void mirror_quadrants(T *z)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x;
	const int ii = N-i;
	const int jj = N-j;

	if (j>0&&j<N/2) z[i*N+jj] = z[i*N+j];
	if (i>0&&i<N/2) z[ii*N+j] = z[i*N+j];
	if (i>0&&i<N/2&&j>0&&j<N/2) z[ii*N+jj] = z[i*N+j];
}

template<typename T>
__global__
void byte_to_complex(byte *b, T *z)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x; // blockDim shall equal N

	z[i*N+j].x = ((float)(b[i*N+j])) / 255.f;
	z[i*N+j].y = 0.f;
}

// not templated -- implementations a little different
// would use auto return if templating
__device__ __forceinline__
float _mod(float2 z)
{
	return hypotf(z.x, z.y);
}
__device__ __forceinline__
half _mod(half2 z)
{
	return length(z);
}

template <typename T1, typename T2>
__global__
void complex_modulus(T1 *z, T2 *r)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x; // blockDim shall equal N

	for (int slice = 0; slice < NUM_SLICES; slice++)
	{
		r[i*N+j] = _mod(z[i*N+j]);

		z += N*N;
		r += N*N;
	}
}

// holding off templating until I've setup float2 library...
__global__
void scale(half2 *x, half a)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x; // blockDim shall equal N

	for (int slice = 0; slice < NUM_SLICES; slice++)
	{
		x[i*N+j] = x[i*N+j] * a;
	}
}

template <typename T>
cudaError_t transfer_psf(T *psf, T *buffer, cudaStream_t stream)
{
	cudaMemcpy3DParms p = { 0 };
	p.srcPtr.ptr = psf;
	p.srcPtr.pitch = (N/2+1) * sizeof(T);
	p.srcPtr.xsize = (N/2+1);
	p.srcPtr.ysize = (N/2+1);
	p.dstPtr.ptr = buffer;
	p.dstPtr.pitch = N * sizeof(T);
	p.dstPtr.xsize = N;
	p.dstPtr.ysize = N;
	p.extent.width = (N/2+1) * sizeof(T);
	p.extent.height = (N/2+1);
	p.extent.depth = NUM_SLICES;
	p.kind = cudaMemcpyHostToDevice;

	return cudaMemcpy3DAsync(&p, stream);
}

int main(int argc, char* argv[])
{
	checkCudaErrors( cudaDeviceReset() );

	complex *image;
	checkCudaErrors( cudaMalloc((void **)&image, N*N*sizeof(complex)) );
	complex *psf;
	checkCudaErrors( cudaMalloc((void **)&psf, N*N*sizeof(complex)) );

	complex *host_psf;
	checkCudaErrors( cudaMallocHost((void **)&host_psf, NUM_SLICES*(N/2+1)*(N/2+1)*sizeof(complex)) );

	byte *image_u8;
	checkCudaErrors( cudaMalloc((void **)&image_u8, N*N*sizeof(byte)) );

	cudaStream_t math_stream, copy_stream;
	checkCudaErrors( cudaStreamCreate(&math_stream) );
	checkCudaErrors( cudaStreamCreate(&copy_stream) );

	complex *in_buffers[2];
	checkCudaErrors( cudaMalloc((void **)&in_buffers[0], NUM_SLICES*N*N*sizeof(complex)) );
	checkCudaErrors( cudaMalloc((void **)&in_buffers[1], NUM_SLICES*N*N*sizeof(complex)) );

	float *out_buffer;
	checkCudaErrors( cudaMalloc((void **)&out_buffer, NUM_SLICES*N*N*sizeof(float)) );

	cudaDataType fft_type;
#ifdef FP32
	fft_type = CUDA_C_32F;
#endif
#ifdef FP16
	fft_type = CUDA_C_16F;
#endif
	cufftHandle fft_plan;
	long long dims[] = {N, N};
	size_t work_sizes = 0;
	checkCudaErrors( cufftCreate(&fft_plan) );
	checkCudaErrors( cufftXtMakePlanMany( \
			fft_plan, 2, dims, \
			NULL, 1, 0, fft_type, \
			NULL, 1, 0, fft_type, \
			1, &work_sizes, fft_type) );
	checkCudaErrors( cufftSetStream(fft_plan, math_stream) );

	// cache 1/4 of the PSF (could do 1/8th too)
	for (int slice = 0; slice < NUM_SLICES; slice++)
	{
		float z = Z0 + DZ * slice;
		float norm_factor = -2.f * z / LAMBDA0;
#ifdef FP16
		norm_factor /= N;
#endif

		// generate the PSF, weakly taking advantage of symmetry to speed up
		construct_psf<<<N/2+1, N/2+1>>>(z, psf, norm_factor);
		mirror_quadrants<<<N/2+1, N/2+1>>>(psf);

		// FFT in-place
		checkCudaErrors( cufftXtExec(fft_plan, psf, psf, CUFFT_FORWARD) );
		checkCudaErrors( cudaStreamSynchronize(math_stream) );

		// do the frequency shift here instead, complex multiplication commutes
		// this is subtle - shifting in conjugate domain means we don't need to FFT shift later
		frequency_shift<<<N/2+1, N/2+1>>>(psf);

		// TODO: the PSF quadrants themselves are symmetric matrices...

		// copy the upper-left submatrix
		checkCudaErrors( cudaMemcpy2D( \
				host_psf + (N/2+1)*(N/2+1)*slice, (N/2+1)*sizeof(complex), \
				psf, N*sizeof(complex), \
				(N/2+1)*sizeof(complex), N/2+1, \
				cudaMemcpyDeviceToHost \
				) );
	}

	checkCudaErrors( cudaFree(psf) );

	// preemptively load PSF for the first frame
	checkCudaErrors( transfer_psf(host_psf, in_buffers[0], copy_stream) );
	checkCudaErrors( cudaStreamSynchronize(copy_stream) );

	volatile bool frameReady = true; // this would be updated by the camera

	// this would be a copy from a frame buffer on the Tegra
	cv::Mat A = cv::imread("test_square.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	for (int frame = 0; frame < NUM_FRAMES; frame++)
	{
		complex *in_buffer = in_buffers[frame % 2];

		// wait for a frame...
		while (!frameReady) { ; }
		// ... and copy
		checkCudaErrors( cudaMemcpyAsync(image_u8, A.data, N*N*sizeof(byte), cudaMemcpyHostToDevice, math_stream) );

		// queue transfer for next frame, waiting for it to finish if necessary(?)
		checkCudaErrors( cudaStreamSynchronize(math_stream) );
		checkCudaErrors( cudaStreamSynchronize(copy_stream) );
		checkCudaErrors( transfer_psf(host_psf, in_buffers[(frame + 1) % 2], copy_stream) );

		// up-cast to complex
		byte_to_complex<<<N, N, 0, math_stream>>>(image_u8, image);
#ifdef FP16
		scale<<<N, N, 0, math_stream>>>(image, 1.f / (float)N);
#endif

		// FFT the image in-place
		checkCudaErrors( cufftXtExec(fft_plan, image, image, CUFFT_FORWARD) );
#ifdef FP16
		scale<<<N, N, 0, math_stream>>>(image, 1.f / (float)N);
#endif

		// random thought: an abstraction layer between kernel allocation and matrix dims would be nice
		// will likely involve template method

		// batch-multiply with FFT'ed image
		// TODO: write a wrapper that takes care of ugly dimension sizes
		quadrant_multiply<<<N/2+1, N/2+1, 0, math_stream>>>(in_buffer, image);

		// inverse FFT that product
		// I have yet to see any speedup from batching the FFTs
		for (int slice = 0; slice < NUM_SLICES; slice++)
		{
			checkCudaErrors( cufftXtExec(fft_plan, in_buffer + N*N*slice, in_buffer + N*N*slice, CUFFT_INVERSE) );
		}

		// complex modulus - faster to loop outside kernel, for some reason
		complex_modulus<<<N, N, 0, math_stream>>>(in_buffer, out_buffer);

		// start timer after first run, GPU "warmup"
		if (frame == 0)
			cudaTimerStart();
	}

	checkCudaErrors( cudaDeviceSynchronize() );

	std::cout << cudaTimerStop() / (NUM_FRAMES - 1) << "ms" << std::endl;

	checkCudaErrors( cudaFree(in_buffers[0]) );
	checkCudaErrors( cudaFree(in_buffers[1]) );
	checkCudaErrors( cufftDestroy(fft_plan) );
	checkCudaErrors( cudaFree(image) );
	checkCudaErrors( cudaFree(image_u8) );
	checkCudaErrors( cudaFreeHost(host_psf) );
	// more cleanup...

	if (argc == 2)
	{
		float *result;
		checkCudaErrors( cudaMallocHost((void **)&result, NUM_SLICES*N*N*sizeof(float)) );
		checkCudaErrors( cudaMemcpy2D(result, sizeof(float), \
									  out_buffer, sizeof(complex)/2, \
									  sizeof(complex)/2, NUM_SLICES*N*N,
									  cudaMemcpyDeviceToHost) );
#ifdef FP_16
		for (int i = 0; i < N*N*NUM_SLICES; i++)
		{
			result[i] = (float)(*(half_float::half *)&(result[i]));
		}
#endif

		for (int slice = 0; slice < NUM_SLICES; slice++)
		{
			imshow(cv::Mat(N, N, CV_32FC1, result + N*N*slice));
		}
	}

	// TODO: reimplement cleanup code once satisfied with implementation

	return 0;
}

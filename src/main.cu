
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
#include <string.h>
#include <stdlib.h>

#include "cuda_common.h"
#include "half_math.cuh"
#include "half.hpp"
#include "util.hpp"

#define FP32
//#define FP16
// should probably return fp32 modulus regardless, for downstream

// 1.9sec in FFT, 0.7sec in multiply for FP32
// 2.3sec in FFT, 0.5sec in multiply for FP16

#define N 1024
#define DX (5.32f / 1024.f)
#define DY (6.66f / 1280.f)
#define DZ 1.f
#define Z0 30
#define LAMBDA0 0.000488f
#define NUM_SLICES 100
#define NUM_FRAMES 10

#ifdef FP32
typedef float2 complex;
typedef float real;
#endif
#ifdef FP16
typedef half2 complex;
typedef half real;
#endif

typedef unsigned char byte;

// Kernel to construct the point-spread function at distance z
__global__
void construct_psf(float z, complex *g, float norm)
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
	r = __fdividef(r, norm); // norm = -2.f * z / LAMBDA0 (also / N if half)

	// re(iz) = -im(z), im(iz) = re(z)
	g[i*N+j] = {__fdividef(-im, r), __fdividef(re, r)};
}

// exploit Fourier duality to shift without copying
// credit to http://www.orangeowlsolutions.com/archives/251
__global__
void frequency_shift(complex *data)
{
    const int i = blockIdx.x;
    const int j = threadIdx.x;

	const real a = (real)(1 - 2 * ((i+j) & 1)); // this looks like a checkerboard?

	data[i*N+j].x *= a;
	data[i*N+j].y *= a;
}

__device__ __forceinline__
float2 conj(float2 a)
{
	float2 c;
	c.x = a.x;
	c.y = -a.y;
	return c;
}

__device__ __forceinline__
float2 cmul(float2 a, float2 b)
{
	float2 c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
}

// using fourfold symmetry of z
__global__
void quadrant_multiply(complex *z, complex *w, char *mask) //, int i, int j)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x;
	const int ii = N-i;
	const int jj = N-j;

	if ((i>0 && i<N/2) && (j>0 && j<N/2))
	{
		complex w1 = w[i*N+j];
		complex w2 = w[ii*N+j];
		complex w3 = w[i*N+jj];
		complex w4 = w[ii*N+jj];

		for (int k = 0; k < NUM_SLICES; k++)
		{
			if (mask[k])
			{
				complex z_ij = z[i*N+j];
				z[i*N+j] = cmul(w1, z_ij);
				z[ii*N+jj] = cmul(w4, z_ij);
				z[ii*N+j] = cmul(w2, z_ij);
				z[i*N+jj] = cmul(w3, z_ij);
			}
			z += N*N;
		}
	}
	else if (i>0 && i<N/2)
	{
		complex w1 = w[i*N+j];
		complex w2 = w[ii*N+j];

		for (int k = 0; k < NUM_SLICES; k++)
		{
			if (mask[k])
			{
				complex z_ij = z[i*N+j];
				z[i*N+j] = cmul(w1, z_ij);
				z[ii*N+j] = cmul(w2, z_ij);
			}
			z += N*N;
		}
	}
	else if (j>0 && j<N/2)
	{
		complex w1 = w[i*N+j];
		complex w2 = w[i*N+jj];

		for (int k = 0; k < NUM_SLICES; k++)
		{
			if (mask[k])
			{
				complex z_ij = z[i*N+j];
				z[i*N+j] = cmul(w1, z_ij);
				z[i*N+jj] = cmul(w2, z_ij);
			}
			z += N*N;
		}
	}
	else
	{
		complex w1 = w[i*N+j];

		for (int k = 0; k < NUM_SLICES; k++)
		{
			if (mask[k])
			{
				complex z_ij = z[i*N+j];
				z[i*N+j] = cmul(w1, z_ij);
			}
			z += N*N;
		}
	}
}

__global__
void mirror_quadrants(complex *z)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x;
	const int ii = N-i;
	const int jj = N-j;

	if (j>0&&j<N/2) z[i*N+jj] = z[i*N+j];
	if (i>0&&i<N/2) z[ii*N+j] = z[i*N+j];
	if (i>0&&i<N/2&&j>0&&j<N/2) z[ii*N+jj] = z[i*N+j];
}

__global__
void byte_to_complex(byte *b, complex *z, float norm_factor = 1.0f)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x; // blockDim shall equal N

	z[i*N+j].x = (real)(((float)(b[i*N+j])) / (255.f * norm_factor));
	z[i*N+j].y = (real)0.f;
}

// careful!!! half is an OBJECT now, so it gets passed as a *reference* ???
__global__
void scale(half2 *x, float a)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x; // blockDim shall equal N

	x[i*N+j] *= (half)a;
}

cudaError_t transfer_psf(complex *psf, complex *buffer, cudaStream_t stream)
{
	// generate parameters for 3D copy
	cudaMemcpy3DParms p = { 0 };
	p.srcPtr.ptr = psf;
	p.srcPtr.pitch = (N/2+1) * sizeof(complex);
	p.srcPtr.xsize = (N/2+1);
	p.srcPtr.ysize = (N/2+1);
	p.dstPtr.ptr = buffer;
	p.dstPtr.pitch = N * sizeof(complex);
	p.dstPtr.xsize = N;
	p.dstPtr.ysize = N;
	p.extent.width = (N/2+1) * sizeof(complex);
	p.extent.height = (N/2+1);
	p.extent.depth = NUM_SLICES;
	p.kind = cudaMemcpyHostToDevice;

	return cudaMemcpy3DAsync(&p, stream);
}

__global__
void modulus(float2 *z, float *r)
{
	const int offset = blockIdx.x * N + threadIdx.x;
	r[offset] = hypotf(z[offset].x, z[offset].y);
}
// should run N/2 threads with this one
// scaling needed to prevent overflow
__global__
void modulus(half2 *z, half *r, float scale)
{
	int offset = blockIdx.x * N + threadIdx.x * 2;
	half2 r2, z2;
	half2 scale_ = (half)scale;

	z2 = z[offset] * scale_;
	z2 *= z2;
	r2.x = z2.x + z2.y;

	offset++;

	z2 = z[offset] * scale_;
	z2 *= z2;
	r2.y = z2.x + z2.y;

	offset--;

	// write 2 at a time
	*(half2 *)&r[offset] = sqrt(r2);
}

__global__
void half_to_float(half *h, float *f)
{
	f[blockIdx.x*N+threadIdx.x] = (float)h[blockIdx.x*N+threadIdx.x];
}

int main(int argc, char* argv[])
{
	checkCudaErrors( cudaDeviceReset() );

	// checkCudaErrors( cudaSetDeviceFlags(cudaDeviceMapHost) );
	// checkCudaErrors( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) );

	complex *image;
	checkCudaErrors( cudaMalloc(&image, N*N*sizeof(complex)) );
	complex *psf;
	checkCudaErrors( cudaMalloc(&psf, N*N*sizeof(complex)) );

	complex *host_psf;
	checkCudaErrors( cudaMallocHost(&host_psf, NUM_SLICES*(N/2+1)*(N/2+1)*sizeof(complex)) );

	byte *image_u8;
	checkCudaErrors( cudaMalloc(&image_u8, N*N*sizeof(byte)) );

	cudaStream_t math_stream, copy_stream;
	checkCudaErrors( cudaStreamCreate(&math_stream) );
	checkCudaErrors( cudaStreamCreate(&copy_stream) );

	complex *in_buffers[2];
	checkCudaErrors( cudaMalloc(&in_buffers[0], NUM_SLICES*N*N*sizeof(complex)) );
	checkCudaErrors( cudaMalloc(&in_buffers[1], NUM_SLICES*N*N*sizeof(complex)) );

	real *out_buffer;
	checkCudaErrors( cudaMalloc(&out_buffer, NUM_SLICES*N*N*sizeof(real)) );

	// managed memory would be much nicer here, esp on Tegra, but was causing problems w/ streams
	char *host_mask, *mask;
	// checkCudaErrors( cudaMallocManaged(&mask, NUM_SLICES*sizeof(char), cudaMemAttachGlobal) );
	checkCudaErrors( cudaMallocHost(&host_mask, NUM_SLICES*sizeof(char)) );
	checkCudaErrors( cudaMalloc(&mask, NUM_SLICES*sizeof(char)) );
	memset(host_mask, 1, NUM_SLICES);
	checkCudaErrors( cudaMemcpy(mask, host_mask, NUM_SLICES*sizeof(char), cudaMemcpyHostToDevice) );

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
	checkCudaErrors( cufftXtMakePlanMany(fft_plan, 2, dims,
										 NULL, 1, 0, fft_type,
										 NULL, 1, 0, fft_type,
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

		// copy the upper-left submatrix
		checkCudaErrors( cudaMemcpy2D(host_psf + (N/2+1)*(N/2+1)*slice, (N/2+1)*sizeof(complex),
									  psf, N*sizeof(complex),
									  (N/2+1)*sizeof(complex), N/2+1,
									  cudaMemcpyDeviceToHost) );
	}

	checkCudaErrors( cudaFree(psf) );

	// preemptively load PSF for the first frame
	checkCudaErrors( transfer_psf(host_psf, in_buffers[0], copy_stream) );
	checkCudaErrors( cudaStreamSynchronize(copy_stream) );

	volatile bool frameReady = true; // this would be updated by the camera

	// this would be a copy from a frame buffer on the Tegra
	cv::Mat A = cv::imread("../data/test_square.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	for (int frame = 0; frame < NUM_FRAMES; frame++)
	{
		complex *in_buffer = in_buffers[frame % 2];

		// wait for a frame...
		while (!frameReady) { ; }
		// ... and copy
		checkCudaErrors( cudaMemcpyAsync(image_u8, A.data, N*N*sizeof(byte),
										 cudaMemcpyHostToDevice, math_stream) );

		// queue transfer for next frame, waiting for it to finish if necessary(?)
		checkCudaErrors( cudaStreamSynchronize(math_stream) );
		checkCudaErrors( cudaStreamSynchronize(copy_stream) );
		checkCudaErrors( transfer_psf(host_psf, in_buffers[(frame + 1) % 2], copy_stream) );

#ifdef FP32
		// up-cast to complex
		byte_to_complex<<<N, N, 0, math_stream>>>(image_u8, image);
#endif
#ifdef FP16
		// up-cast to complex
		byte_to_complex<<<N, N, 0, math_stream>>>(image_u8, image, N);
#endif

		// FFT the image in-place
		checkCudaErrors( cufftXtExec(fft_plan, image, image, CUFFT_FORWARD) );

		// batch-multiply with FFT'ed image
		quadrant_multiply<<<N/2+1, N/2+1, 0, math_stream>>>(in_buffer, image, mask);

		// inverse FFT the product - batch FFT gave no speedup
		for (int slice = 0; slice < NUM_SLICES; slice++)
		{
			if (host_mask[slice])
			{
				checkCudaErrors( cufftXtExec(fft_plan,
											 in_buffer + N*N*slice,
											 in_buffer + N*N*slice,
											 CUFFT_INVERSE) );

#ifdef FP16
				modulus<<<N, N/2, 0, math_stream>>>(in_buffer + N*N*slice, out_buffer + N*N*slice, 1.f / (float)N); // 1.f / sqrt((float)N));
#endif
#ifdef FP32
				modulus<<<N, N, 0, math_stream>>>(in_buffer + N*N*slice, out_buffer + N*N*slice);
			}
#endif
		}

		// construct volume from one frame's worth of slices once they're ready...
		cudaStreamSynchronize(math_stream);
		// ... and return the next slices to query (i.e. might want to query all every 1sec or so)
		memset(host_mask, 1, NUM_SLICES);

		checkCudaErrors( cudaMemcpy(mask, host_mask, NUM_SLICES*sizeof(char), cudaMemcpyHostToDevice) );

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
		float *h_out = new float[NUM_SLICES*N*N];
		float *d_out;

#ifdef FP16
		checkCudaErrors( cudaMalloc(&d_out, NUM_SLICES*N*N*sizeof(float)) );
		for (int i=0; i<NUM_SLICES; i++) { half_to_float<<<N, N>>>(out_buffer + i*N*N, d_out + i*N*N); }
#endif
#ifdef FP32
		d_out = out_buffer;
#endif
		checkCudaErrors( cudaMemcpy(h_out, d_out, NUM_SLICES*N*N*sizeof(float), cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaFree(d_out) );

		for (int slice = 0; slice < NUM_SLICES; slice++)
		{
			imshow(cv::Mat(N, N, CV_32FC1, h_out + N*N*slice));
		}
	}

	// TODO: reimplement cleanup code once satisfied with implementation

	return 0;
}

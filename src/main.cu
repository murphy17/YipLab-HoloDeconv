
/*
 *
 * Proof-of-concept for GPU holographic deconvolution.
 * Michael Murphy, May 2017
 * Yip Lab
 *
 */

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
#define DX (5.32f / 1024.f)
#define DY (6.66f / 1280.f)
#define LAMBDA0 0.000488f
#define NUM_SLICES 100 // 100
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
		cv::magnitude(channels[0], channels[1], out);
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
void construct_psf(float z, half2 *g, float norm)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x; // blockDim shall equal N

	const int ii = (N - 1) - i;
	const int jj = (N - 1) - j;

	// not sure whether the expansion of N/(N-1) was necessary
	float scale = (float)N / (float)(N-1);
	float x = (i * scale - N/2) * DX;
	float y = (j * scale - N/2) * DY;

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
	half2 g_ij_fp16 = __floats2half2_rn(g_ij.x, g_ij.y);

	// CUDA takes care of reversed-index coalescing, this is fine
	g[i*N+j] = g_ij_fp16;
	g[i*N+jj] = g_ij_fp16;
	g[ii*N+j] = g_ij_fp16;
	g[ii*N+jj] = g_ij_fp16;
}

// exploit Fourier duality to shift without copying
// credit to http://www.orangeowlsolutions.com/archives/251
__global__
void frequency_shift(half2 *data)
{
    int i = blockIdx.x;
    int j = threadIdx.x;

	float a = 1 - 2 * ((i+j) & 1); // this looks like a checkerboard?

	data[i*N+j] = __hmul2(data[i*N+j], __float2half2_rn(a));
}

__global__
void batch_multiply(half2 *z, const __restrict__ half2 *w)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x; // blockDim shall equal N

	half2 w_ij = w[i*N+j];
	half2 w_ij_inv = __lowhigh2highlow(w_ij);

	for (int k = 0; k < NUM_SLICES; k++)
	{
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
}

__global__
void byte_to_half2(const __restrict__ byte *b, half2 *z)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x; // blockDim shall equal N

	z[i*N+j] = __floats2half2_rn(((float)(b[i*N+j])) / 255.f, 0.f);
}

__global__
void batch_modulus(half2 *z, half *r)
{
	int i = blockIdx.x;
	int j = threadIdx.x * 2; // blockDim shall equal N/2

	for (int slice = 0; slice < NUM_SLICES; slice++)
	{
		half2 ax_ay = __hmul2(z[i*N+j], z[i*N+j]);
		half2 bx_by = __hmul2(z[i*N+j+1], z[i*N+j+1]);

		// 'transpose'
		half2 ax_bx = __highs2half2(ax_ay, bx_by);
		half2 ay_by = __lows2half2(ax_ay, bx_by);

		// full-byte stores
		*(half2 *)&r[i*N+j] = h2sqrt(__hadd2(ax_bx, ay_by));

		z += N*N;
		r += N*N;
	}
}

__global__
void normalize_by(half2 *h, float n)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	h[i*N+j] = __hmul2(h[i*N+j], __float2half2_rn(1.f / n));
}

int main(int argc, char* argv[])
{
	checkCudaErrors( cudaDeviceReset() );

	int num_frames = 10;
	float z_min = 30;
	float z_step = 1;

	long long dims[] = {N, N};
	size_t work_sizes = 0;
	size_t buffer_size = NUM_SLICES*N*N*sizeof(half2);

	half2 *image;
	checkCudaErrors( cudaMalloc((void **)&image, N*N*sizeof(half2)) );
	half2 *psf;
	checkCudaErrors( cudaMalloc((void **)&psf, N*N*sizeof(half2)) );
	// allocate this on the host - that way CPU can manage transfer, not GPU, lets it run in async
	// (this is 3x slower on Titan, but faster on Tegra - recall that host and GPU memory are the same thing in Tegra)
	half2 *host_psf;
	checkCudaErrors( cudaMallocHost((void **)&host_psf, buffer_size) );

	byte *image_u8;
	checkCudaErrors( cudaMalloc((void **)&image_u8, N*N*sizeof(byte)) );

	cudaStream_t math_stream, copy_stream;
	checkCudaErrors( cudaStreamCreate(&math_stream) );
	checkCudaErrors( cudaStreamCreate(&copy_stream) );

	half2 *buffers[2];
	checkCudaErrors( cudaMalloc((void **)&buffers[0], buffer_size) );
	checkCudaErrors( cudaMalloc((void **)&buffers[1], buffer_size) );

	half *modulus;
	checkCudaErrors( cudaMalloc((void **)&modulus, buffer_size / 2) );

	cufftHandle fft_plan;
	checkCudaErrors( cufftCreate(&fft_plan) );
	checkCudaErrors( cufftXtMakePlanMany( \
			fft_plan, 2, dims, \
			NULL, 1, 0, CUDA_C_16F, \
			NULL, 1, 0, CUDA_C_16F, \
			1, &work_sizes, CUDA_C_16F) );
	checkCudaErrors( cufftSetStream(fft_plan, math_stream) );

	// cache the PSF host-side
	// this causes problems for Titan (separate memory), but ~10% speedup for Tegra
	for (int slice = 0; slice < NUM_SLICES; slice++)
	{
		float z = z_min + z_step * slice;

		// generate the PSF, weakly taking advantage of symmetry to speed up
		// ... which is no longer necessary because it's only generated once
		construct_psf<<<N/2, N/2, 0, math_stream>>>(z, psf, -2.f * z / LAMBDA0 / N);

		// FFT in-place
		checkCudaErrors( cufftXtExec(fft_plan, psf, psf, CUFFT_FORWARD) );

		// do the frequency shift here instead, complex multiplication commutes
		// this is subtle - shifting in conjugate domain means we don't need to FFT shift (i.e. copy) later
		frequency_shift<<<N, N, 0, math_stream>>>(psf);

		checkCudaErrors( cudaMemcpyAsync(host_psf + N*N*slice, psf, N*N*sizeof(half2), cudaMemcpyDeviceToHost, math_stream) );
	}

	// this would be a copy from a frame buffer on the Tegra
	cv::Mat A = cv::imread("test_square.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	volatile bool frameReady = true; // this would be updated by the camera

	// initialize the buffer for the first frame
	checkCudaErrors( cudaMemcpyAsync(buffers[0], host_psf, buffer_size, cudaMemcpyHostToDevice, copy_stream) );

	for (int frame = 0; frame < num_frames; frame++)
	{
		half2 *buffer = buffers[frame % 2];

		// wait for a frame...
		while (!frameReady) { ; }
		// ... and copy
		checkCudaErrors( cudaMemcpyAsync(image_u8, A.data, N*N*sizeof(byte), cudaMemcpyHostToDevice, math_stream) );

		// start copying the PSF for the next frame
		// this is on the host so the copy doesn't occupy GPU
		checkCudaErrors( cudaStreamSynchronize(copy_stream) );
		checkCudaErrors( cudaMemcpyAsync(buffers[(frame + 1) % 2], host_psf, buffer_size, cudaMemcpyHostToDevice, copy_stream) );

		// up-cast to complex
		byte_to_half2<<<N, N, 0, math_stream>>>(image_u8, image);
		normalize_by<<<N, N, 0, math_stream>>>(image, N);

		// FFT the image in-place
		checkCudaErrors( cufftXtExec(fft_plan, image, image, CUFFT_FORWARD) );
		normalize_by<<<N, N, 0, math_stream>>>(image, N);

		// batch-multiply with FFT'ed image
		batch_multiply<<<N, N, 0, math_stream>>>(buffer, image);

		// inverse FFT that product
		// I have yet to see any speedup from batching the FFTs
		for (int slice = 0; slice < NUM_SLICES; slice++)
		{
			checkCudaErrors( cufftXtExec(fft_plan, buffer + N*N*slice, buffer + N*N*slice, CUFFT_INVERSE) );
		}

		batch_modulus<<<N, N/2, 0, math_stream>>>(buffer, modulus);

		// start timer after first run, GPU "warmup"
		if (frame == 0)
			cudaTimerStart();
	}

	checkCudaErrors( cudaDeviceSynchronize() );

	std::cout << cudaTimerStop() / (num_frames - 1) << "ms" << std::endl;

	checkCudaErrors( cudaFree(image) );
	checkCudaErrors( cudaFree(psf) );
	checkCudaErrors( cudaFreeHost(host_psf) );

	checkCudaErrors( cufftDestroy(fft_plan) );
	checkCudaErrors( cudaStreamDestroy(math_stream) );
	checkCudaErrors( cudaStreamDestroy(copy_stream) );

	checkCudaErrors( cudaFree(buffers[0]) );
	checkCudaErrors( cudaFree(buffers[1]) );

	half_float::half *host_buffer;
	checkCudaErrors( cudaMallocHost((void **)&host_buffer, buffer_size) );

	checkCudaErrors( cudaMemcpy(host_buffer, modulus, NUM_SLICES*N*N*sizeof(half), cudaMemcpyDeviceToHost) );

	if (argc == 2)
	{
		for (int slice = 0; slice < NUM_SLICES; slice++)
		{
			cv::Mat B(N, N, CV_32FC1);
			for (int i = 0; i < N*N; i++) { ((float *)(B.data))[i] = (float)((host_buffer + N*N*slice)[i]); }
			imshow(B);
		}
	}

	// TODO: reimplement cleanup code once satisfied with implementation

	return 0;
}

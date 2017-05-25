
/*
 *
 * Proof-of-concept for GPU holographic deconvolution.
 * Michael Murphy, May 2017
 * Yip Lab
 *
 */

// this is the fastest yet on Tegra, keeping PSF on device gives fastest yet on Titan
// ... consumes a TON of memory. immediately fills the Tegra
// could serialize batches of slices, I suppose - hiding latency of multiply is main thing
// ... I've got an experiment for the multiply in the works, would require transposing the PSF cube though

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

//__global__
//void batch_multiply(half2 *z, const __restrict__ half2 *w)
//{
//	// threadIdx.x = slice index
//	// threadIdx.y = element index
//
//	// each thread block processes blockDims.x different elements of w
//	__shared__ half2 cache[MAX_BLOCK_THREADS / NUM_SLICES];
//
//	int inIdx = blockIdx.x * blockDim.x + threadIdx.x; // blockDim.x  (MAX_BLOCK_THREADS / NUM_SLICES)
//	int outIdx = threadIdx.y * (N*N) + inIdx;
////	int outIdx = threadIdx.y + inIdx * NUM_SLICES; // same elements in successive slices adjacent in memory
//	// ^^^ this is wrong! threads are adjacent in x, not y!
//
//	if (threadIdx.y == 0)
//	{
//		cache[threadIdx.x] = w[inIdx];
//	}
//	__syncthreads();
//
//	half2 a = z[outIdx];
//	float a_temp = a.y;
//	float ay_by = __fmul_rn(a_temp, cache[threadIdx.x].y);
//	float ay_bx = __fmul_rn(a_temp, cache[threadIdx.x].x);
//	a_temp = a.x;
//
//	z[outIdx].x = __fmaf_rn(a_temp, cache[threadIdx.x].x, -ay_by);
//	z[outIdx].y = __fmaf_rn(a_temp, cache[threadIdx.x].y, ay_bx);
//}

__global__
void batch_multiply(half2 *z, const __restrict__ half2 *w)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x; // blockDim shall equal N

	half2 b = w[i*N+j];
	half2 b_inv = __lowhigh2highlow(b);

	for (int k = 0; k < NUM_SLICES; k++)
	{
		half2 a = z[i*N+j];

		// figure out if c_x, c_y can be packed

		half2 temp = __hmul2(a, b);
		half c_x = __hsub(__low2half(temp), __high2half(temp));

		temp = __hmul2(a, b_inv);
		half c_y = __hadd(__low2half(temp), __high2half(temp));

		z[i*N+j] = __halves2half2(c_x, c_y);

		z += N*N;

		// __syncthreads(); // coalesce?
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
void modulus_half2(half2 *z, half *r)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	half2 temp = __hmul2(z[i*N+j], z[i*N+j]); // this might saturate
	r[i*N+j] = __hadd(__low2half(temp), __high2half(temp));

	// I'm a bit concerned about how those intrinsics expand
	// too many instructions seems likely
	// write your own?
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

	int num_frames = 3;
	float z_min = 30;
	float z_step = 1;
	int num_streams = 2;

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

	cudaStream_t streams[num_streams];
	half2 *stream_buffers[num_streams];
	cufftHandle fft_plans[num_streams];

	// setup multiply kernel
//	dim3 grid_dims(N*N / (MAX_BLOCK_THREADS / NUM_SLICES));
//	dim3 block_dims(MAX_BLOCK_THREADS / NUM_SLICES, NUM_SLICES);

	// TODO: investigate properly batched FFTs
	// (does space requirement increase? if so don't)
	for (int i = 0; i < num_streams; i++)
	{
		checkCudaErrors( cudaStreamCreate(&streams[i]) );
		checkCudaErrors( cufftCreate(&fft_plans[i]) );
		checkCudaErrors( cufftXtMakePlanMany( \
				fft_plans[i], 2, dims, \
				NULL, 1, 0, CUDA_C_16F, \
				NULL, 1, 0, CUDA_C_16F, \
				1, &work_sizes, CUDA_C_16F) );
		checkCudaErrors( cufftSetStream(fft_plans[i], streams[i]) );
		checkCudaErrors( cudaMalloc((void **)&stream_buffers[i], buffer_size) );
	}

	// cache the PSF host-side
	// this causes problems for Titan (separate memory), but ~10% speedup for Tegra
	for (int slice = 0; slice < NUM_SLICES; slice++)
	{
		float z = z_min + z_step * slice;

		// generate the PSF, weakly taking advantage of symmetry to speed up
		construct_psf<<<N/2, N/2, 0, streams[0]>>>(z, psf, -2.f * z / LAMBDA0 / N);

		// FFT in-place
		checkCudaErrors( cufftXtExec(fft_plans[0], psf, psf, CUFFT_FORWARD) );

		// do the frequency shift here instead, complex multiplication commutes
		// this is subtle - shifting in conjugate domain means we don't need to FFT shift later
		frequency_shift<<<N, N, 0, streams[0]>>>(psf);

		checkCudaErrors( cudaMemcpyAsync(host_psf + N*N*slice, psf, N*N*sizeof(half2), cudaMemcpyDeviceToHost, streams[0]) );
	}

	// this would be a copy from a frame buffer on the Tegra
	cv::Mat A = cv::imread("test_square.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	volatile bool frameReady = true; // this would be updated by the camera

	for (int frame = 0; frame < num_frames; frame++)
	{
		int stream_num = frame % num_streams;
		cudaStream_t stream = streams[stream_num];
		half2 *buffer = stream_buffers[stream_num];

		// wait for a frame...
		while (!frameReady) { ; }
		// ... and copy
		checkCudaErrors( cudaMemcpyAsync(image_u8, A.data, N*N*sizeof(byte), cudaMemcpyHostToDevice, stream) );

		// this is on the host so the copy doesn't occupy GPU
		// because the call is non-blocking, it means PSF copy for *next* frame gets cued up immediately
		checkCudaErrors( cudaMemcpyAsync(buffer, host_psf, buffer_size, cudaMemcpyHostToDevice, stream) );

		// up-cast to complex
		byte_to_half2<<<N, N, 0, stream>>>(image_u8, image);
		normalize_by<<<N, N, 0, stream>>>(image, N);

		// FFT the image in-place
		checkCudaErrors( cufftXtExec(fft_plans[stream_num], image, image, CUFFT_FORWARD) );
		normalize_by<<<N, N, 0, stream>>>(image, N);

		// batch-multiply with FFT'ed image
		batch_multiply<<<N, N, 0, stream>>>(buffer, image);
//		batch_multiply<<<grid_dims, block_dims, 0, stream>>>(buffer, image);

		// inverse FFT that product
		// TODO: doing the modulus in here as callback would be quite nice, would like to retry
		// I have yet to see any speedup from batching the FFTs
		for (int slice = 0; slice < NUM_SLICES; slice++)
		{
			checkCudaErrors( cufftXtExec(fft_plans[stream_num], buffer + N*N*slice, buffer + N*N*slice, CUFFT_INVERSE) );
		}

		// complex modulus - faster to loop outside kernel, for some reason
		// TODO: reusing the first half of the buffer ... is this fine? not sure about that!!!
		for (int slice = 0; slice < NUM_SLICES; slice++)
		{
			modulus_half2<<<N, N, 0, stream>>>(buffer + N*N*slice, (half *)buffer + N*N*slice);
		}

		// start timer after first run, GPU "warmup"
		if (frame == 0)
			cudaTimerStart();
	}

	checkCudaErrors( cudaDeviceSynchronize() );

	std::cout << cudaTimerStop() / (num_frames - 1) << "ms" << std::endl;

	// Tegra runs out of memory when I try to visualize 100 slices...

//	checkCudaErrors( cudaFree(image) );
//	checkCudaErrors( cudaFree(psf) );
//	checkCudaErrors( cudaFreeHost(host_psf) );
//
//	for (int i = 0; i < num_streams; i++)
//	{
//		checkCudaErrors( cufftDestroy(fft_plans[i]) );
//		checkCudaErrors( cudaStreamDestroy(streams[i]) );
//		// checkCudaErrors( cudaFree(stream_buffers[i]) );
//	}
//
//	checkCudaErrors( cudaFree(stream_buffers[1]) );
//
//	checkCudaErrors( cudaDeviceSynchronize() );
//
//	half_float::half *host_buffer;
//	checkCudaErrors( cudaMallocHost((void **)&host_buffer, buffer_size) );
//
//	checkCudaErrors( cudaMemcpy(host_buffer, stream_buffers[0], NUM_SLICES*N*N*sizeof(half), cudaMemcpyDeviceToHost) );
//
//	checkCudaErrors( cudaFree(stream_buffers[0]) );
//
//	if (argc == 2)
//	{
//		for (int slice = 0; slice < NUM_SLICES; slice++)
//		{
//			cv::Mat B(N, N, CV_32FC1);
//			for (int i = 0; i < N*N; i++) { ((float *)(B.data))[i] = (float)((host_buffer + N*N*slice)[i]); }
//			imshow(B);
//		}
//	}

	// TODO: reimplement cleanup code once satisfied with implementation

	return 0;
}


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

#include "common.h"

#define N 1024
#define LOG2N 10
#define DX (5.32f / 1024.f)
#define DY (6.66f / 1280.f)
#define LAMBDA0 0.000488f
#define SCALE 0.00097751711f // 1/(N-1)
#define NUM_SLICES 64 // 100
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

//	asm(".reg .f32 ay_by;\n" // float ay_by
//		".reg .f32 ay_bx;\n" // float ay_bx
//		" mul .f32 ay_by, %2, %0;\n" // ay_by = __fmul_rn(ay, by)
//		" mul .f32 ay_bx, %2, %1;\n" // ay_bx = __fmul_rn(ay, bx)
//		" neg .f32 ay_by, ay_by;\n" // ay_by = -ay_by
//		" fma.rn .f32 %1, %3, %1, ay_by;\n" // bx = __fmaf_rn(ax, bx, ay_by);
//		" fma.rn .f32 %0, %3, %0, ay_bx;\n" : \
//		"+f"(by), "+f"(bx) : \
//		"f"(a.y), "f"(a.x)); // by = __fmaf_rn(ax, by, ay_bx);

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
void batch_multiply(cufftComplex *z, const __restrict__ cufftComplex *w)
{
	// threadIdx.x = slice index
	// threadIdx.y = element index

	// each thread block processes blockDims.x different elements of w
	__shared__ cufftComplex cache[MAX_BLOCK_THREADS / NUM_SLICES];

	int inIdx = blockIdx.x * blockDim.x + threadIdx.x; // blockDim.x  (MAX_BLOCK_THREADS / NUM_SLICES)
	int outIdx = threadIdx.y * (N*N) + inIdx;
//	int outIdx = threadIdx.y + inIdx * NUM_SLICES; // same elements in successive slices adjacent in memory
	// ^^^ this is wrong! threads are adjacent in x, not y!

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

__global__
void byte_to_complex(byte *b, cufftComplex *z)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x; // blockDim shall equal N

	z[i*N+j].x = ((float)(b[i*N+j])) / 255.f;
	z[i*N+j].y = 0.f;
}

__global__
void complex_modulus(cufftComplex *z, float *r)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x; // blockDim shall equal N

	r[i*N+j] = hypotf(z[i*N+j].x, z[i*N+j].y);
}

__global__
void copy_buffer(cufftComplex *a, cufftComplex *b)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x; // blockDim shall equal N

	b[i*N+j] = a[i*N+j];
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
	size_t buffer_size = NUM_SLICES*N*N*sizeof(cufftComplex);

	cufftComplex *image;
	checkCudaErrors( cudaMalloc((void **)&image, N*N*sizeof(cufftComplex)) );
	cufftComplex *psf;
	checkCudaErrors( cudaMalloc((void **)&psf, N*N*sizeof(cufftComplex)) );
	// allocate this on the host - that way CPU can manage transfer, not GPU, lets it run in async
	// (this is 3x slower on Titan, but faster on Tegra - recall that host and GPU memory are the same thing in Tegra)
	cufftComplex *host_psf;
	checkCudaErrors( cudaMallocHost((void **)&host_psf, buffer_size) );

	byte *image_u8;
	checkCudaErrors( cudaMalloc((void **)&image_u8, N*N*sizeof(byte)) );

	cudaStream_t streams[num_streams];
	cufftComplex *stream_buffers[num_streams];
	cufftHandle fft_plans[num_streams];

	// setup multiply kernel
	dim3 grid_dims(N*N / (MAX_BLOCK_THREADS / NUM_SLICES));
	dim3 block_dims(MAX_BLOCK_THREADS / NUM_SLICES, NUM_SLICES);

	// TODO: investigate properly batched FFTs
	// (does space requirement increase? if so don't)
	for (int i = 0; i < num_streams; i++)
	{
		checkCudaErrors( cudaStreamCreate(&streams[i]) );
		checkCudaErrors( cufftCreate(&fft_plans[i]) );
		checkCudaErrors( cufftXtMakePlanMany( \
				fft_plans[i], 2, dims, \
				NULL, 1, 0, CUDA_C_32F, \
				NULL, 1, 0, CUDA_C_32F, \
				1, &work_sizes, CUDA_C_32F) );
		checkCudaErrors( cufftSetStream(fft_plans[i], streams[i]) );
		checkCudaErrors( cudaMalloc((void **)&stream_buffers[i], buffer_size) );
	}

	// cache the PSF; with a Titan or 8GB TX2 this shouldn't be an issue
	// note this allows async copying for next frame... (I think? are self-copies DMA?)
	for (int slice = 0; slice < NUM_SLICES; slice++)
	{
		float z = z_min + z_step * slice;

		// generate the PSF, weakly taking advantage of symmetry to speed up
		construct_psf<<<N/2, N/2, 0, streams[0]>>>(z, psf, -2.f * z / LAMBDA0);

		// FFT in-place
		checkCudaErrors( cufftXtExec(fft_plans[0], psf, psf, CUFFT_FORWARD) );

		// do the frequency shift here instead, complex multiplication commutes
		// this is subtle - shifting in conjugate domain means we don't need to FFT shift later
		frequency_shift<<<N, N, 0, streams[0]>>>(psf);

		checkCudaErrors( cudaMemcpyAsync(host_psf + N*N*slice, psf, N*N*sizeof(cufftComplex), cudaMemcpyDeviceToHost, streams[0]) );
	}

	// this would be a copy from a frame buffer on the Tegra
	cv::Mat A = cv::imread("test_square.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	volatile bool frameReady = true; // this would be updated by the camera

	for (int frame = 0; frame < num_frames; frame++)
	{
		int stream_num = frame % num_streams;
		cudaStream_t stream = streams[stream_num];
		cufftComplex *buffer = stream_buffers[stream_num];

		// wait for a frame...
		while (!frameReady) { ; }
		// ... and copy
		checkCudaErrors( cudaMemcpyAsync(image_u8, A.data, N*N*sizeof(byte), cudaMemcpyHostToDevice, stream) );

		// this is on the host so the copy doesn't occupy GPU
		// because the call is non-blocking, it means PSF copy for *next* frame gets cued up immediately
		checkCudaErrors( cudaMemcpyAsync(buffer, host_psf, buffer_size, cudaMemcpyHostToDevice, stream) );

		// up-cast to complex
		byte_to_complex<<<N, N, 0, stream>>>(image_u8, image);

		// FFT the image in-place
		checkCudaErrors( cufftExecC2C(fft_plans[stream_num], image, image, CUFFT_FORWARD) );

		// batch-multiply with FFT'ed image
//		batch_multiply<<<N, N, 0, stream>>>(buffer, image);
		batch_multiply<<<grid_dims, block_dims, 0, stream>>>(buffer, image);

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
			complex_modulus<<<N, N, 0, stream>>>(buffer + N*N*slice, (float *)buffer + N*N*slice);
		}

		// checkCudaErrors( cudaStreamSynchronize(streams[(frame - 1) % num_streams]) ); // this would not be needed!
		// checkCudaErrors( cudaMemcpyAsync(host_buffer, buffer, buffer_size, cudaMemcpyDeviceToHost, stream) ); // holy shit this is slow

		// start timer after first run, GPU "warmup"
		if (frame == 0)
			cudaTimerStart();
	}

	checkCudaErrors( cudaDeviceSynchronize() );

	std::cout << cudaTimerStop() / (num_frames - 1) << "ms" << std::endl;

	// Tegra runs out of memory when I try to visualize...

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
//	float *host_buffer;
//	checkCudaErrors( cudaMallocHost((void **)&host_buffer, buffer_size) );
//
//	checkCudaErrors( cudaMemcpy(host_buffer, stream_buffers[0], buffer_size, cudaMemcpyDeviceToHost) );
//
//	if (argc == 2)
//	{
//		for (int slice = 0; slice < NUM_SLICES; slice++)
//		{
//			cv::Mat B(N, N, CV_32FC1, host_buffer + N*N*slice);
//			imshow(B);
//		}
//	}

	// TODO: reimplement cleanup code once satisfied with implementation

	return 0;
}

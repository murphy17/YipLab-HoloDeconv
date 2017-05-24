
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
#define NUM_SLICES 100

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

// it seems you can't have too many plans simultaneously.
// workaround: conditionals in the callback?
// ... I tried this. much slower. thought branching was killing performance
// which doesn't make sense, all threads take same path
// it wasn't, which is good, sort of... turns out the *struct* was the issue

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
	const int i = blockIdx.x;
	const int j = threadIdx.x;

	cufftComplex b = w[i*N+j];
	float bx = b.x; float by = b.y; // just making sure

	for (int k = 0; k < NUM_SLICES; k++)
	{
		cufftComplex a = z[i*N+j];
		float a_temp = a.y;
		float ay_by = __fmul_rn(a_temp, by);
		float ay_bx = __fmul_rn(a_temp, bx);
		a_temp = a.x;
		z[i*N+j].x = __fmaf_rn(a_temp, bx, -ay_by);
		z[i*N+j].y = __fmaf_rn(a_temp, by, ay_bx);

		z += N*N;
	}
}

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

int main(int argc, char* argv[])
{
	checkCudaErrors( cudaDeviceReset() );

	int num_frames = 5;
	float z_min = 30;
	float z_step = 1;
	int num_streams = 2;

	long long dims[] = {N, N};
	size_t work_sizes = 0;
	size_t buffer_size = NUM_SLICES*N*N*sizeof(cufftComplex);

	cufftComplex *image;
	checkCudaErrors( cudaMalloc((void **)&image, N*N*sizeof(cufftComplex)) );
	cufftComplex *psf;
	checkCudaErrors( cudaMalloc((void **)&image, buffer_size) );

	byte *image_u8;
	checkCudaErrors( cudaMalloc((void **)&image_u8, N*N*sizeof(byte)) );

	// wouldn't exist in streaming application
	// don't do the copy in performance testing, also this is a LOT of memory to pin
	float *host_buffer;
	checkCudaErrors( cudaMallocHost((void **)&host_buffer, buffer_size) );

	cudaStream_t streams[num_streams];
	cufftComplex *stream_buffers[num_streams];
	cufftHandle fft_plans[num_streams];

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
		construct_psf<<<N/2, N/2, 0, math_stream>>>(z, d_psf + N*N*slice, -2.f * z / LAMBDA0);

		// FFT in-place
		checkCudaErrors( cufftXtExec(plan, d_psf + N*N*slice, d_psf + N*N*slice, CUFFT_FORWARD) );

		// do the frequency shift here instead, complex multiplication commutes
		// this is subtle - shifting in conjugate domain means we don't need to FFT shift later
		frequency_shift<<<N, N>>>(d_psf + N*N*slice);
	}

	for (int frame = 0; frame < num_frames; frame++)
	{
		int stream_num = frame % num_streams;
		cudaStream_t stream = streams[stream_num];
		cufftComplex *buffer = stream_buffers[stream_num];

		// this would be a copy from a frame buffer on the Tegra
		// this is a blocking call, simulate waiting for a frame
		cv::Mat A = cv::imread("test_square.bmp", CV_LOAD_IMAGE_GRAYSCALE);

		// non-blocking call - copy for *next* frame gets immediately cued after this one
		checkCudaErrors( cudaMemcpyAsync(image_u8, A.data, N*N*sizeof(byte), cudaMemcpyHostToDevice, stream) );

		// up-cast to complex
		byte_to_complex<<<N, N, 0, stream>>>(image_u8, image);

		// FFT the image in-place
		checkCudaErrors( cufftExecC2C(fft_plans[stream_num], image, image, CUFFT_FORWARD) );

		// load the PSFs into working area
		checkCudaErrors( cudaMemcpyAsync(buffer, psf, buffer_size, cudaMemcpyDeviceToDevice, stream) );

		// batch-multiply with FFT'ed image
		batch_multiply<<<N, N, 0, stream>>>(buffer, image);

		// inverse FFT that product
		// TODO: doing the modulus in here as callback would be quite nice, would like to retry
		// I have yet to see any speedup from batching the FFTs
		// TODO: verify this is non-blocking
		for (int slice = 0; slice < NUM_SLICES; slice++)
		{
			checkCudaErrors( cufftXtExec(fft_plans[stream_num], buffer + N*N*slice, buffer + N*N*slice, CUFFT_INVERSE) );
		}

		// complex modulus - faster to loop outside kernel, for some reason
		// TODO: reusing the first half of the buffer ... is this fine? not sure about that!!!
		for (int slice = 0; slice < NUM_SLICES; slice++)
		{
			complex_modulus<<<N, N, 0, stream>>>(buffer + N*N*slice, (float)buffer + N*N*slice);
		}

		// checkCudaErrors( cudaStreamSynchronize(streams[(frame - 1) % num_streams]) ); // this would not be needed!
		checkCudaErrors( cudaMemcpyAsync(host_buffer, buffer, buffer_size, cudaMemcpyDeviceToHost, stream) );

		// start timer after first run, GPU "warmup"
		if (frame == 0)
			cudaTimerStart();
	}

	checkCudaErrors( cudaDeviceSynchronize() );

	std::cout << cudaTimerStop() / (num_frames - 1) << "ms" << std::endl;

	if (argc == 2)
	{
		for (int slice = 0; slice < NUM_SLICES; slice++)
		{
			cv::Mat B(N, N, CV_32FC1, host_buffer + N*N*slice);
			imshow(B);
		}
	}

	// TODO: reimplement cleanup code once satisfied with implementation

	return 0;
}


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
//#include <opencv2/gpu/gpu.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <algorithm>

#include "common.h"

#define N 1024
#define LOG2N 10
#define DX (5.32f / 1024.f)
#define DY (5.32f / 1024.f) // (6.66f / 1280.f) ... are these supposed to be the *SAME*? very close, but exact same helps a lot!
#define LAMBDA0 0.000488f
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

//	const int ii = (N - 1) - i;
//	const int jj = (N - 1) - j;

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
	cufftComplex g_ij;
	g_ij.x = __fdividef(-im, r); // im, r);
	g_ij.y = __fdividef(re, r);

	// CUDA takes care of coalescing the reversed access, this is fine
	g[i*N+j] = g_ij;
//	g[i*N+jj] = g_ij;
//	g[ii*N+j] = g_ij;
//	g[ii*N+jj] = g_ij;
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

__device__
cufftComplex _multiply_helper(cufftComplex a, cufftComplex b)
{
	cufftComplex c;
	float a_temp = a.y;
	float ay_by = __fmul_rn(a_temp, b.y);
	float ay_bx = __fmul_rn(a_temp, b.x);
	a_temp = a.x;
	c.x = __fmaf_rn(a_temp, b.x, -ay_by);
	c.y = __fmaf_rn(a_temp, b.y, ay_bx);
	return c;
}

__global__
void batch_multiply(cufftComplex *z, const __restrict__ cufftComplex *w)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x;

	cufftComplex w_ij = w[i*N+j];

	for (int k = 0; k < NUM_SLICES; k++)
	{
		z[i*N+j] = _multiply_helper(z[i*N+j], w_ij);

		z += N*N;
	}
}

__global__
void quadrant_multiply(cufftComplex *z, const __restrict__ cufftComplex *w)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x;
	const int ii = N-i;
	const int jj = N-j;

	cufftComplex w_ = w[i*N+j];
	cufftComplex *z_ = z;

	// once this is working move those conditionals outside the loop
	// unless compiler is doing that already... no, it's not

	if (i>0&&i<N/2&&j>0&&j<N/2)
	{
		for (int k = 0; k < NUM_SLICES; k++)
		{
			z_[ii*N+jj] = _multiply_helper(z_[ii*N+jj], w_);
			z_ += N*N;
		}
	}
	if (i>0&&i<N/2)
	{
		for (int k = 0; k < NUM_SLICES; k++)
		{
			z_[ii*N+j] = _multiply_helper(z_[ii*N+j], w_);
			z_ += N*N;
		}
	}
	if (j>0&&j<N/2)
	{
		for (int k = 0; k < NUM_SLICES; k++)
		{
			z_[i*N+jj] = _multiply_helper(z_[i*N+jj], w_);
			z_ += N*N;
		}
	}
	for (int k = 0; k < NUM_SLICES; k++)
	{
		z_[i*N+j] = _multiply_helper(z_[i*N+j], w_);
		z_ += N*N;
	}
}

__global__
void mirror_quadrants(cufftComplex *z)
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

		checkCudaErrors( cudaMemset(psf, 0, N*N*sizeof(cufftComplex)) );

		// generate the PSF, weakly taking advantage of symmetry to speed up
		construct_psf<<<N/2+1, N/2+1, 0, streams[0]>>>(z, psf, -2.f * z / LAMBDA0);

		// testing symmetry
		mirror_quadrants<<<N/2+1, N/2+1, 0, streams[0]>>>(psf);

		// FFT in-place
		checkCudaErrors( cufftXtExec(fft_plans[0], psf, psf, CUFFT_FORWARD) );

		// testing symmetry
		mirror_quadrants<<<N/2+1, N/2+1, 0, streams[0]>>>(psf);

		// do the frequency shift here instead, complex multiplication commutes
		// this is subtle - shifting in conjugate domain means we don't need to FFT shift later
		frequency_shift<<<N, N, 0, streams[0]>>>(psf);

		// TODO: only store (N/2+1)^2 entries
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

		// random thought: an abstraction layer between kernel allocation and matrix dims would be nice
		// will likely involve template method

		// batch-multiply with FFT'ed image
		quadrant_multiply<<<N/2+1, N/2+1, 0, stream>>>(buffer, image);
//		batch_multiply<<<grid_dims, block_dims, 0, stream>>>(buffer, image);

		checkCudaErrors( cudaGetLastError() );

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

		// start timer after first run, GPU "warmup"
		if (frame == 0)
			cudaTimerStart();
	}

	checkCudaErrors( cudaDeviceSynchronize() );

	std::cout << cudaTimerStop() / (num_frames - 1) << "ms" << std::endl;

	float *host_buffer;
	checkCudaErrors( cudaMallocHost((void **)&host_buffer, buffer_size) );

	checkCudaErrors( cudaMemcpy(host_buffer, stream_buffers[0], buffer_size, cudaMemcpyDeviceToHost) );

	checkCudaErrors( cudaFree(stream_buffers[0]) );

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


/*
 *
 * Proof-of-concept for GPU holographic deconvolution.
 * Michael Murphy, May 2017
 * Yip Lab
 *
 */

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/gpu/gpu.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <algorithm>
#include <arrayfire.h>
#include <af/cuda.h>

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

	// I'm really skeptical about the memory access here
	// but when I tried shared memory it was slower
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

__global__
void byte_to_complex(byte *b, cufftComplex *z)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	z[i*N+j].x = ((float)(b[i*N+j])) / 255.f;
	z[i*N+j].y = 0.f;
}

int main(int argc, char* argv[])
{
	checkCudaErrors( cudaDeviceReset() );

	int num_frames = 10;
	int num_slices = 100;
	float z_min = 30;
	float z_step = 1;

	byte *d_img_u8;
	checkCudaErrors( cudaMalloc((void **)&d_img_u8, N*N*sizeof(byte)) );

	// wouldn't exist in streaming application
	float *h_slices = (float *)af::pinned(N*N*num_slices, f32);

	af::array A(N, N, c32);
	af::array A_img(N, N, c32);
	af::array A_mod(N, N, num_slices, f32);
	cufftComplex *d_psf = (cufftComplex *)A.device<af::cfloat>();
	cufftComplex *d_img = (cufftComplex *)A_img.device<af::cfloat>();

	int d_id = af::getDevice();

	// initially query all slices
	for (int frame = 0; frame < num_frames; frame++)
	{
		// this would be a copy from a frame buffer on the Tegra
		cv::Mat img = cv::imread("test_square.bmp", CV_LOAD_IMAGE_GRAYSCALE);

		A_img.lock();
		checkCudaErrors( cudaMemcpyAsync(d_img_u8, img.data, N*N*sizeof(byte), cudaMemcpyHostToDevice, afcu::getStream(d_id)) );

		byte_to_complex<<<N, N, 0, afcu::getStream(d_id)>>>(d_img_u8, d_img);

		A_img.unlock();
		af::fft2InPlace(A_img);

		// this is subtle - shifting in conjugate domain means we don't need to FFT shift later
		A_img.lock();
		frequency_shift<<<N, N, 0, afcu::getStream(d_id)>>>(d_img);
		A_img.unlock();

		for (int slice = 0; slice < num_slices; slice++)
		{
			float z = z_min + z_step * slice;

			// generate the PSF, weakly taking advantage of symmetry to speed up
			// this could be done in AF too
			A.lock();
			construct_psf<<<N/2, N/2, 0, afcu::getStream(d_id)>>>(z, d_psf, -2.f * z / LAMBDA0); // speedup with shared memory?
			A.unlock();

			af::fft2InPlace(A);
			A *= A_img;
			af::ifft2InPlace(A);
//			af::eval(A);
//			af::sync();
//			imshow(cv::cuda::GpuMat(N, N, CV_32FC2, (cufftComplex *)A.device<af::cfloat>()));
//			A.unlock();
			// for FFT shift would need to invert phase now, but it doesn't matter since we're taking modulus
			A_mod(af::span, af::span, slice) = af::abs(A);
		}

		A_mod.host(h_slices);

		if (frame == 0)
			cudaTimerStart();
	}

	checkCudaErrors( cudaDeviceSynchronize() );

	std::cout << cudaTimerStop() / (num_frames-1) << "ms" << std::endl;

	if (argc == 2)
	{
		for (int slice = 0; slice < num_slices; slice++)
		{
			cv::Mat B(N, N, CV_32FC1, h_slices + N*N*slice);
			imshow(B);
		}
	}

//	checkCudaErrors( cudaFree(d_img) );
//	checkCudaErrors( cudaFree(d_img_u8) );
//	checkCudaErrors( cudaFree(d_psf) );
//	checkCudaErrors( cudaFree(d_slices) );
//	checkCudaErrors( cudaFreeHost(h_slices) );

	return 0;
}

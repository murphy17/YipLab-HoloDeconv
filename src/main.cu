
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
void imshow(cv::gpu::GpuMat in, bool log_scale = false)
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
	if (log_scale)
		cv::log(out, out);
	cv::normalize(out, out, 1.0, 0.0, cv::NORM_MINMAX, -1);
	cv::imshow("Display window", out); // Show our image inside it.
	cv::waitKey(0);
}

// Kernel to construct the point-spread function at distance z.
// exploits 4-fold symmetry (PSF is also radially symmetric, but that's harder...)
// note that answer is scaled between +/-1
__global__
void construct_psf(float z, half *g_re, half *g_im, float norm)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	int ii = (N - 1) - i;
	int jj = (N - 1) - j;

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
	half g_ij_re = __float2half(g_ij.x);
	half g_ij_im = __float2half(g_ij.y);

	// TODO: clean this mess up
	g_re[i*N+j] = g_ij_re;
	g_re[i*N+jj] = g_ij_re;
	g_re[ii*N+j] = g_ij_re;
	g_re[ii*N+jj] = g_ij_re;
	g_im[i*N+j] = g_ij_im;
	g_im[i*N+jj] = g_ij_im;
	g_im[ii*N+j] = g_ij_im;
	g_im[ii*N+jj] = g_ij_im;
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

		z += N*(N/2+1);
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

__global__
void merge_filter_halves(half2 *x_f, half2 *y_f, half2 *z_f)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x;

	// x = a+ib, y = c+id, z = x+iy
	// re(z) = re(x+iy) = re(a+ib+ic-d) = a-d
	// im(z) = im(x+iy) = im(a+ib+ic-d) = b+c

	half a = __low2half(x_f[i*N+j]);
	half b = __high2half(x_f[i*N+j]);
	half c = __low2half(y_f[i*N+j]);
	half d = __high2half(y_f[i*N+j]);

	z_f[i*N+j] = __halves2half2(__hsub(a, d), __hadd(b, c));
}

// inefficient memory accesses - vectorize
__global__
void byte_to_half(byte *b, half *h)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	h[i*N+j] = __float2half(((float)(b[i*N+j])) / 255.f);
}

__global__
void normalize_by(half *h, float n)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	h[i*N+j] = __hmul(h[i*N+j], __float2half(1.f / n));
}

__global__
void half_to_float(half *h, float *f)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	f[i*N+j] = __half2float(h[i*N+j]);
}

// TODO: put all these casting kernels in a header
__global__
void half2_to_complex(half2 *h, cufftComplex *z)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	z[i*N+j] = __half22float2(h[i*N+j]);
}

int main(int argc, char* argv[])
{
	checkCudaErrors( cudaDeviceReset() );

	int num_frames = 10;
	float z_min = 30;
	float z_step = 1;

	long long dims[] = {N, N};
	size_t work_sizes = 0;

	// allocate this on the host - that way CPU can manage transfer, not GPU, lets it run in async
	// (this is 3x slower on Titan, but faster on Tegra - recall that host and GPU memory are the same thing in Tegra)
	half2 *host_psf;
	checkCudaErrors( cudaMallocHost((void **)&host_psf, NUM_SLICES*N*(N/2+1)*sizeof(half2)) );

	half *img;
	checkCudaErrors( cudaMalloc((void **)&img, N*N*sizeof(half)) );

	half2 *img_f;
	checkCudaErrors( cudaMalloc((void **)&img_f, N*(N/2+1)*sizeof(half2)) );

	byte *img_u8;
	checkCudaErrors( cudaMalloc((void **)&img_u8, N*N*sizeof(byte)) );

	cudaStream_t math_stream, copy_stream;
	checkCudaErrors( cudaStreamCreate(&math_stream) );
	checkCudaErrors( cudaStreamCreate(&copy_stream) );

	// inefficient! figure out reuse!!!
	half2 *in_buffers[2];
	checkCudaErrors( cudaMalloc((void **)&in_buffers[0], NUM_SLICES*N*(N/2+1)*sizeof(half2)) );
	checkCudaErrors( cudaMalloc((void **)&in_buffers[1], NUM_SLICES*N*(N/2+1)*sizeof(half2)) );
	half *out_buffers[2];
	checkCudaErrors( cudaMalloc((void **)&out_buffers[0], NUM_SLICES*N*N*sizeof(half)) );
	checkCudaErrors( cudaMalloc((void **)&out_buffers[1], NUM_SLICES*N*N*sizeof(half)) );

	cufftHandle plan_r2c;
	cufftCreate(&plan_r2c);
	checkCudaErrors( cufftXtMakePlanMany(plan_r2c, 2, dims, \
			NULL, 1, 0, CUDA_R_16F, \
			NULL, 1, 0, CUDA_C_16F, \
			1, &work_sizes, CUDA_C_16F) );
	checkCudaErrors( cufftSetStream(plan_r2c, math_stream) );

	cufftHandle plan_c2r;
	cufftCreate(&plan_c2r);
	checkCudaErrors( cufftXtMakePlanMany(plan_c2r, 2, dims, \
			NULL, 1, 0, CUDA_C_16F, \
			NULL, 1, 0, CUDA_R_16F, \
			1, &work_sizes, CUDA_C_16F) );
	checkCudaErrors( cufftSetStream(plan_c2r, math_stream) );

	// free these after setting up PSF
	half *psf_re, *psf_im;
	checkCudaErrors( cudaMalloc((void **)&psf_re, N*N*sizeof(half)) );
	checkCudaErrors( cudaMalloc((void **)&psf_im, N*N*sizeof(half)) );
	half2 *psf_re_f, *psf_im_f;
	checkCudaErrors( cudaMalloc((void **)&psf_re_f, N*(N/2+1)*sizeof(half2)) );
	checkCudaErrors( cudaMalloc((void **)&psf_im_f, N*(N/2+1)*sizeof(half2)) );
	half2 *psf_f;
	checkCudaErrors( cudaMalloc((void **)&psf_f, N*(N/2+1)*sizeof(half2)) );

	// cache the PSF host-side
	// this causes problems for Titan (separate memory), but ~10% speedup for Tegra
	for (int slice = 0; slice < NUM_SLICES; slice++)
	{
		// TODO: do FFTs, merge, FFT-shift in float, only convert to half at end

		float z = z_min + z_step * slice;

		// generate the PSF, weakly taking advantage of symmetry to speed up
		// ... which is no longer necessary because it's only generated once
		construct_psf<<<N/2, N/2, 0, math_stream>>>(z, psf_re, psf_im, -2.f * z / LAMBDA0 / N);

		// FFT the real and imaginary halves separately
		// PSF not real-valued but it is symmetric, so the C2R transform still works!
		checkCudaErrors( cufftXtExec(plan_r2c, psf_re, psf_re_f, CUFFT_FORWARD) );
		checkCudaErrors( cufftXtExec(plan_r2c, psf_im, psf_im_f, CUFFT_FORWARD) );

		// join the real and imaginary halves, yielding an Nx(N/2+1) filter matrix
		merge_filter_halves<<<N/2+1, N>>>(psf_re_f, psf_im_f, psf_f);

		// do the frequency shift here instead, complex multiplication commutes
		// this is subtle - shifting in conjugate domain means we don't need to FFT shift (i.e. copy) later
//		frequency_shift<<<N, N, 0, math_stream>>>(psf); // disabled until I figure this out for R2C/C2R

		checkCudaErrors( cudaMemcpyAsync(host_psf + N*(N/2+1)*slice, psf_f, N*(N/2+1)*sizeof(half2), \
				cudaMemcpyDeviceToHost, math_stream) );
	}

	// this would be a copy from a frame buffer on the Tegra
	cv::Mat A = cv::imread("test_square.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	volatile bool frameReady = true; // this would be updated by the camera

	// initialize the buffer for the first frame
	checkCudaErrors( cudaMemcpyAsync(in_buffers[0], host_psf, NUM_SLICES*N*(N/2+1)*sizeof(half2), \
			cudaMemcpyHostToDevice, copy_stream) );

	for (int frame = 0; frame < num_frames; frame++)
	{
		half2 *in_buffer = in_buffers[frame % 2];
		half *out_buffer = out_buffers[frame % 2];

		// wait for a frame...
		while (!frameReady) { ; }
		// ... and copy
		checkCudaErrors( cudaMemcpyAsync(img_u8, A.data, N*N*sizeof(byte), cudaMemcpyHostToDevice, math_stream) );

		// start copying the PSF for the next frame
		// this is on the host so the copy doesn't occupy GPU
		checkCudaErrors( cudaStreamSynchronize(copy_stream) ); // wait for previous copy to finish if it hasn't
		checkCudaErrors( cudaMemcpyAsync(in_buffers[(frame + 1) % 2], host_psf, NUM_SLICES*N*(N/2+1)*sizeof(half2), \
				cudaMemcpyHostToDevice, copy_stream) );

		// convert to half... i think there's an intrinsic for this
		byte_to_half<<<N, N, 0, math_stream>>>(img_u8, img);
		// inefficient!
		normalize_by<<<N, N, 0, math_stream>>>(img, N);

		// FFT the image
		checkCudaErrors( cufftXtExec(plan_r2c, img, img_f, CUFFT_FORWARD) );

		normalize_by<<<N/2+1, N, 0, math_stream>>>(img_f, N);

		// batch-multiply with FFT'ed image
		batch_multiply<<<N/2+1, N, 0, math_stream>>>(in_buffer, img_f);

		// inverse FFT that product; cuFFT batching gave no speedup whatsoever and this permits plan reuse
		for (int slice = 0; slice < NUM_SLICES; slice++)
		{
//			float2 *img_c32;
//			checkCudaErrors( cudaMalloc((void **)&img_c32, N*(N/2+1)*sizeof(float2)) );
//			half2_to_complex<<<N/2+1, N>>>(in_buffer + N*(N/2+1)*slice, img_c32);
//			imshow(cv::gpu::GpuMat(N, N/2+1, CV_32FC2, img_c32), true);

			checkCudaErrors( cufftXtExec(plan_c2r, in_buffer + N*(N/2+1)*slice, out_buffer + N*N*slice, CUFFT_INVERSE) );
			// could now immediately start async populating the in_buffer, since this is out-of-place and iterative
			// (i.e. might not need to double buffer after all)
		}

		// start timer after first run, GPU "warmup"
		if (frame == 0)
			cudaTimerStart();
	}

	checkCudaErrors( cudaDeviceSynchronize() );

	std::cout << cudaTimerStop() / (num_frames - 1) << "ms" << std::endl;

//	checkCudaErrors( cudaFree(image) );
//	checkCudaErrors( cudaFree(psf) );
//	checkCudaErrors( cudaFreeHost(host_psf) );
//
//	checkCudaErrors( cufftDestroy(fft_plan) );
//	checkCudaErrors( cudaStreamDestroy(math_stream) );
//	checkCudaErrors( cudaStreamDestroy(copy_stream) );
//
//	checkCudaErrors( cudaFree(buffers[0]) );
//	checkCudaErrors( cudaFree(buffers[1]) );
//
	half_float::half *host_buffer;
	checkCudaErrors( cudaMallocHost((void **)&host_buffer, NUM_SLICES*N*N*sizeof(half_float::half)) );
//
	checkCudaErrors( cudaMemcpy(host_buffer, out_buffers[0], NUM_SLICES*N*N*sizeof(half), cudaMemcpyDeviceToHost) );

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

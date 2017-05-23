
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
#include <opencv2/core/cuda.hpp>
//#include <opencv2/gpu/gpu.hpp>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <algorithm>
#include <cuda_fp16.h>

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
void construct_psf(float z, half2 *g, float norm)
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

	// cast to half-precision
	half2 g_ij_fp16 = __floats2half2_rn(g_ij.x, g_ij.y);

	// I'm really skeptical about the memory access here - each seems around 2.5ms
	// but when I tried shared memory it was slower
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

// it seems you can't have too many plans simultaneously.
// workaround: conditionals in the callback?
// ... I tried this. much slower. thought branching was killing performance
// which doesn't make sense, all threads take same path
// it wasn't, which is good, sort of... turns out the *struct* was the issue

// this will have an out-of-bounds error... unless you allocate an extra element
__device__
void _mul(void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
	half2 a, b;

	a = *(half2 *)&element;
	b = ((half2 *)callerInfo)[offset];

	// don't use intrinsics here, this is fastest
//	c.x = a.x * b.x - a.y * b.y;
//	c.y = a.x * b.y + a.y * b.x;

	half2 temp = __hmul2(a, b);
	half c_x = __hadd(((half *)&temp)[0], __hneg(((half *)&temp)[1]));

	temp = __hmul2(a, __lowhigh2highlow(b));
	half c_y = __hadd(((half *)&temp)[0], ((half *)&temp)[1]);

	((half2 *)dataOut)[offset] = __halves2half2(c_x, c_y);
}
__device__
cufftCallbackStoreC d_mul = _mul;

__global__
void byte_to_complex(byte *b, half2 *z)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	z[i*N+j] = __floats2half2_rn(((float)(b[i*N+j])) / 255.f, 0.f);
}

//__global__
//void byte_to_half_to_float(byte *b, float *f)
//{
//	int i = blockIdx.x;
//	int j = threadIdx.x; // blockDim shall equal N
//
//	half h = __float2half(((float)(b[i*N+j])) / 255.f);
//	f[i*N+j] = __half2float(h);
//}

__global__
void complex_modulus(half2 *z, float *r)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	half2 temp = __hmul2(z[i*N+j], z[i*N+j]);
	r[i*N+j] = __half2float(__hadd(__high2half(temp), __low2half(temp)));

	// I'm a bit concerned about how those intrinsics expand
	// too many instructions seems likely
	// write your own?
}

__global__
void multiply_filter(half2 *z, half2 *w)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	half2 a, b;

	a = z[i*N+j];
	b = w[i*N+j];

	half2 temp = __hmul2(a, b);
	half c_x = __hadd(__high2half(temp), __low2half(temp));

	temp = __hmul2(a, __lowhigh2highlow(b));
	half c_y = __hadd(__high2half(temp), __low2half(temp));

	z[i*N+j] = __halves2half2(c_x, c_y);
}

int main(int argc, char* argv[])
{
	checkCudaErrors( cudaDeviceReset() );

	int num_frames = 5;
	int num_slices = 100;
	float z_min = 30;
	float z_step = 1;

	cudaStream_t math_stream, copy_stream;
	checkCudaErrors( cudaStreamCreate(&math_stream) );
	checkCudaErrors( cudaStreamCreate(&copy_stream) );

	// the multiply callback was throwing invalid type error...
	cufftHandle plan;
	long long dims[] = {N, N};
	size_t work_sizes = 0;
	checkCudaErrors( cufftCreate(&plan) );
	checkCudaErrors( cufftXtMakePlanMany(plan, 2, dims, \
			NULL, 1, 0, CUDA_C_16F, \
			NULL, 1, 0, CUDA_C_16F, \
			1, &work_sizes, CUDA_C_16F) );
	checkCudaErrors( cufftSetStream(plan, math_stream) );

	half2 *d_img, *d_psf;
	checkCudaErrors( cudaMalloc((void **)&d_psf, N*N*sizeof(half2)) );
	checkCudaErrors( cudaMalloc((void **)&d_img, N*N*sizeof(half2)) );

	byte *d_img_u8;
	checkCudaErrors( cudaMalloc((void **)&d_img_u8, N*N*sizeof(byte)) );

	// full-size is necessary for downstream reduction, need the whole cube
	float *d_slices;
	checkCudaErrors( cudaMalloc((void **)&d_slices, num_slices*N*N*sizeof(float)) );

	// wouldn't exist in streaming application
	float *h_slices;
	checkCudaErrors( cudaMallocHost((void **)&h_slices, num_slices*N*N*sizeof(float)) );

	for (int frame = 0; frame < num_frames; frame++)
	{
		// this would be a copy from a frame buffer on the Tegra
		cv::Mat A = cv::imread("test_square.bmp", CV_LOAD_IMAGE_GRAYSCALE);

		cudaTimerStart();

		checkCudaErrors( cudaMemcpy(d_img_u8, A.data, N*N*sizeof(byte), cudaMemcpyHostToDevice) );

		byte_to_complex<<<N, N>>>(d_img_u8, d_img);

		checkCudaErrors( cufftXtExec(plan, d_img, d_img, CUFFT_FORWARD) );
		checkCudaErrors( cudaStreamSynchronize(math_stream) ); // reusing a plan

		// problem with 16-bit FFT - the order of magnitude varies by like 5
		// can I truncate?

		// this is subtle - shifting in conjugate domain means we don't need to FFT shift later
		frequency_shift<<<N, N>>>(d_img);

		for (int slice = 0; slice < num_slices; slice++)
		{
			float z = z_min + z_step * slice;

			// generate the PSF, weakly taking advantage of symmetry to speed up
			construct_psf<<<N/2, N/2, 0, math_stream>>>(z, d_psf, -2.f * z / LAMBDA0); // speedup with shared memory?

			// FFT and multiply. the multiplication is the primary bottleneck in this workflow
			checkCudaErrors( cufftXtExec(plan, d_psf, d_psf, CUFFT_FORWARD) ); // big speedup with callback! ~40%

			multiply_filter<<<N, N, 0, math_stream>>>(d_psf, d_img); // hold off FP16 callback until FFT working

			// inverse FFT that product
			checkCudaErrors( cufftXtExec(plan, d_psf, d_psf, CUFFT_INVERSE) );

			// for FFT shift would need to invert phase now, but it doesn't matter since we're taking modulus

			// callback doesn't help here either
			complex_modulus<<<N, N, 0, math_stream>>>(d_psf, d_slices + N*N*slice); // no need to sync streams, full-size buffer

			// it's actually faster to async each slice, surprisingly
			// could use just a single image for buffer, stream sync was negligible last I checked
			cudaStreamSynchronize(math_stream);
			checkCudaErrors( cudaMemcpyAsync(h_slices + N*N*slice, d_slices + N*N*slice, N*N*sizeof(float), cudaMemcpyDeviceToHost, copy_stream) );
		}

		checkCudaErrors( cudaDeviceSynchronize() );

		std::cout << cudaTimerStop() << "ms" << std::endl;

		// now do some reduction to the whole cube...
		// ...

		// which returns which slices contained objects of interest
//		checkCudaErrors( cudaMemset(d_query, 1, num_slices) ); // for demo purpose, all of them
//		checkCudaErrors( cudaMemcpy(h_query, d_query, num_slices*sizeof(byte), cudaMemcpyDeviceToHost) );

		// for the next frame, query the neighborhoods about each of those
		// ...

		// ! this is an implicit assumption about the z-velocity of the objects in the sample
		// picking neighborhood via Kalman filtering would be very cool, but probably overkill
		// every few frames, just query everything?

		// looking at utilisation, might be able to halve (!!!) that with batching
	}

	if (argc == 2)
	{
		for (int slice = 0; slice < num_slices; slice++)
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

	checkCudaErrors( cudaStreamDestroy(math_stream) );
	checkCudaErrors( cudaStreamDestroy(copy_stream) );

	return 0;
}

/*
typedef struct {
// int N;
float lambda0;
float del_x;
float del_y;
float d_max;
float d_min;
float d_step;
int n_batch;
int n_frames;
} params_t;
// Populates frequency grid and vector of spacings.
//void setup_vars(params_t params, af::array *grid, af::array *spacings)
//{
// int n_step = (params.d_max - params.d_min) / params.d_step;
//    // build (symmetric?) path-length grid
// af::array df = (af::iota(af::dim4(N, 1)) * (float)(N)/(float)(N-1)-N/2);
//    *grid = af::tile(af::pow(df * params.del_x, 2), 1, N) + af::tile(af::pow(df * params.del_y, 2).T(), N, 1);
//    *spacings = af::pow(af::range(af::dim4(n_step)) * params.d_step + params.d_min, 2);
//}
// Expands input image into slices, performs a reduction, and writes the result out.
void process_image(params_t params, af::array &img, float *out_ptr, af::array &x_cube) //, af::array &grid, af::array &spacings)
{
af::array x(N, N, c32);
// int n_step = spacings.dims(0);
//    af::cfloat k0 = {0., (float)(-2. * af::Pi / params.lambda0)};
//    af::cfloat k1 = {0., (float)(1. / params.lambda0)};
af::cfloat unit = {0, 1};
// FFT the input image
af::array h_f = af::fft2(img);
// phase shift it
h_f = h_f * unit;
int n_step = (params.d_max - params.d_min) / params.d_step;
// process in batches to fit in memory
// ... but this seems to entirely occupy the Tegra...
for (int j = 0; j < n_step; j ++)
{
float z = params.d_min + j * params.d_step;
af::sync();
complex *d_x = (complex *)x.device<af::cfloat>();
construct_psf_4fold<<<N/2, N/2>>>(z, d_x);
cudaDeviceSynchronize();
x.unlock();
// x = af::sqrt(grid + params.d_min + i * params.d_step); // ~0.15sec
// x = k1 * af::exp(k0 * x) / x; // g
af::fft2InPlace(x); // g_f // 0.3sec
// note here: g is an even function
// so F(g) is real valued and even
// (i.e. can I just take FFT of half of g?)
x = x * h_f; // gh_f
af::ifft2InPlace(x); // h // 0.4sec
// x = af::shift(x, x.dims(0)/2, x.dims(1)/2); // FFT shift
x_cube(af::span, af::span, j) = af::abs(x); // / af::max<float>(x) * 255.; // compression probably unnecessary?
}
// simulate doing some reduction operation that returns a single image per cube
// i.e. find optimal focus -> construct 3D volume
af::array x_sum = af::sum(x_cube, 2);
// push to host
x_sum.host(out_ptr);
}
int main(void)
{
// setup experimental parameters
    params_t params;
//    N = 1024; // resolution in pixels
    params.lambda0 = 0.000488; // wavelength
    params.del_x = 5.32 / 1024; // horizontal frequency spacing
    params.del_y = 6.66 / 1280; // vertical frequency spacing
    params.d_max = 130; // max distance from CCD to object, mm
    params.d_min = 30; // min distance from CCD to object, mm
    params.d_step = 1; // step size in mm
    int N_batch = 1; // number of frames per batch; best performance with 1!?!?
    int N_frames = 3; // currently, just repeats the analysis
    // load in test image
    cv::Mat mat = cv::imread("test_square.bmp", CV_LOAD_IMAGE_GRAYSCALE);
    mat.convertTo(mat, CV_32FC1);
    // simulate a DMA buffer on the GPU, i.e. feeding in from video camera
    // will copy in the images as part of the main loop, simulate 'streaming'
    float *img_ptr;
    cudaMalloc((void **)&img_ptr, N * N * sizeof(float));
    // allocate the matrix using our predefined staging area
    af::array img(N, N, img_ptr, afDevice);
    af::eval(img);
    // pin buffer on the host
    float *h_ptr = af::pinned<float>(N * N * N_frames);
//    af::array grid;
//    af::array spacings;
//    setup_vars(params, &grid, &spacings);
    // allocate this just once and reuse, it's huge
//    int n_step = spacings.dims(0);
    int n_step = (params.d_max - params.d_min) / params.d_step;
    af::array x_cube(N, N, n_step, f32);
    for (int k = 0; k < N_frames; k++)
    {
// 'copy the image' - these would be successive frames in reality, and would probably live on GPU
// i.e. this copy would not happen
// mat = mat + 0.; // no possibility of caching
cudaMemcpy(img_ptr, mat.data, N * N * sizeof(float), cudaMemcpyHostToDevice);
// expand the image into slices, do a reduction, save result to h_ptr
af::timer::start();
process_image(params, img, h_ptr + N * N * k, x_cube); //, grid, spacings);
std::cout << af::timer::stop() << std::endl;
    }
    cv::namedWindow("Display window", cv::WINDOW_NORMAL); // Create a window for display.
    for (int i = 0; i < n_step; i++)
    {
        cv::Mat mat(cv::Size(1024, 1024), CV_32FC1, h_ptr + i * N * N);
        cv::normalize(mat, mat, 1.0, 0.0, cv::NORM_MINMAX, -1);
        cv::imshow("Display window", mat); // Show our image inside it.
        cv::waitKey(0);
    }
    return 0;
}
*/

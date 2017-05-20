/*
 *
 * Proof-of-concept for GPU holographic deconvolution.
 * Michael Murphy, May 2017
 * Yip Lab
 *
 */

// it's obvious RT on the Jetson is not going to be a possibility.
// try this on the server, first of all, it's like 10x faster than the Jetson... lol
// then try to install AF there

// alternative with ArrayFire: construct 1/4 of the kernel, concatenate

// Jetson natively supports FP16... hmmm
// play with the thread / block sizes.

// ~20ms runtimes earlier were wrong... ArrayFire has lazy evaluation, and eval() is NON-blocking!
// record for 100 slices now is ~1.25sec, AF

// priority: FFT callbacks; batching; half-precision (latter is apparently good enough for graphics)
// (suspect batching will make async execution redundant)
// if memory copying is determined a bottleneck, do the transfers async
// compare in-place and out-of-place; memory is not the bottleneck here

// run half as many kernels, use vectorized instructions...
// https://devblogs.nvidia.com/parallelfo rall/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
// http://stackoverflow.com/questions/26676806/efficiency-of-cuda-vector-types-float2-float3-float4

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <cufft.h>
//#include <cuComplex.h>
#include <cuda_runtime.h> // need to enable separate program linking, cufft static lib
#include <cufftXt.h>
//#include <cublas_v2.h>
#include "common.h"

//#include <arrayfire.h>
//#include <algorithm>

#define N 1024
#define DX (5.32 / 1024)
#define DY (6.66 / 1280)
#define LAMBDA0 0.000488

typedef cufftComplex complex;

// Convenience method for plotting
void imshow(cv::Mat img)
{
	cv::namedWindow("Display window", cv::WINDOW_NORMAL); // Create a window for display.
	cv::Mat img_norm;
	cv::normalize(img, img_norm, 1.0, 0.0, cv::NORM_MINMAX, -1);
	cv::imshow("Display window", img_norm); // Show our image inside it.
	cv::waitKey(0);
}

// Kernel to construct the point-spread function at distance z.
// exploits 4-fold symmetry (PSF is also radially symmetric, but that's harder...)
// note that answer is scaled between +/-1
__global__
void construct_psf_4fold(float z, complex *g)
{
//	__shared__ complex g_r[N/2]; // bank conflicts?

	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N
	int ii = (N - 1) - i;
	int jj = (N - 1) - j;

	float x = (i * (float)N/(float)(N-1) - N/2) * DX;
	float y = (j * (float)N/(float)(N-1) - N/2) * DY;

	float r = -2.f * norm3df(x, y, z) / LAMBDA0;

	// exp(ix) = cos(x) + isin(x)
	float re, im;
	sincospif(r, &im, &re);

	// numerical conditioning, probably unnecessary
	// also corrects the sign flip above
	r /= -2.f * z / LAMBDA0;

	// re(iz) = -im(z), im(iz) = re(z)
	complex g_ij = make_cuFloatComplex(-im / r, re / r);

	g[i*N+j] = g_ij;
	g[i*N+jj] = g_ij;
	g[ii*N+j] = g_ij;
	g[ii*N+jj] = g_ij;

	// this is slower!?!?!?!?!
	// write the half-row
//	g[i*N+j] = g_ij;
//	g[((N - 1) - i)*N+j] = g_ij;
//	// flip the half-row
//	g_r[(N/2 - 1) - j] = g_ij; // bank conflicts? can't avoid I think
//	__syncthreads(); // needed?
//	// write the flipped half-row
//	g[i*N+j+N/2] = g_r[j];
//	g[((N - 1) - i)*N+j+N/2] = g_r[j];
}

// construct the kernel directly inside the FFT call
__device__
cufftComplex _psf(void *dataIn, size_t offset, void *callerInfo, void *sharedPtr)
{
	// would you need to mod offset by N*N with batch? (probably?)
	// might be able to do some weird stuff with cufftPlanMany, like stride of 0
	// (don't want to allocate a batch's worth of empty memory)
	int i = offset % N;
	int j = offset / N; // ?

	float x = (i * (float)N/(float)(N-1) - N/2) * DX;
	float y = (j * (float)N/(float)(N-1) - N/2) * DY;
	float z = *(float *)&callerInfo;

	float r = -2.f * norm3df(x, y, z) / LAMBDA0;

	// exp(ix) = cos(x) + isin(x)
	float re, im;
	sincospif(r, &im, &re);

	// numerical conditioning, probably unnecessary
	// also corrects the sign flip above
	r /= -2.f * z / LAMBDA0;

	// re(iz) = -im(z), im(iz) = re(z)
	cufftComplex g;
	g.x = -im / r;
	g.y = re / r;

	return g;
}
__device__
cufftCallbackLoadC d_psf = _psf;

__global__
void complex_cast(float *x, complex *z)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	z[i*N+j].x = x[i*N+j];
}

//__device__
//void _mul(void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
//{
//	cufftComplex a, b, c;
//
//	a = element;
//	b = ((cufftComplex*)callerInfo)[offset];
//
//	c.x = a.x * b.x - a.y * b.y;
//	c.y = a.x * b.y + a.y * b.x;
//
//	((cufftComplex*)dataOut)[offset] = c;
//}
//_device__
//cufftCallbackStoreC _mul_d = _mul;

//__global__
//void multiply_inplace(complex *z, complex *w)
//{
//	int i = blockIdx.x;
//	int j = threadIdx.x; // blockDim shall equal N
//
//	z[i*N+j] = cuCmulf(z[i*N+j], w[i*N+j]);
//}

__global__
void complex_mod(complex *z, float *r)
{
	int i = blockIdx.x;
	int j = threadIdx.x; // blockDim shall equal N

	r[i*N+j] = hypotf(z[i*N+j].x, z[i*N+j].y);
}

// exploit Fourier duality to shift without copying
// credit to http://www.orangeowlsolutions.com/archives/251
__global__
void frequency_shift(complex *data)
{
    int i = blockIdx.x;
    int j = threadIdx.x;

	float a = 1-2*((i+j)&1);

	data[i*N+j].x *= a;
	data[i*N+j].y *= a;
}

int main(void)
{
    cudaDeviceReset();

	int num_frames = 1;
	int num_slices = 100;
	float z_min = 30;
	float dz = 1;

	float z = 50;

	cufftHandle plan_psf;
	checkCudaErrors( cufftPlan2d(&plan_psf, N, N, CUFFT_C2C) );

	cufftComplex *h;
	checkCudaErrors( cudaMalloc((void **)&h, N*N*sizeof(cufftComplex)) );

	cufftCallbackLoadC h_psf;
	checkCudaErrors( cudaMemcpyFromSymbol(&h_psf, d_psf, sizeof(cufftCallbackLoadC)) );
	checkCudaErrors( cufftXtSetCallback(plan_psf, (void **)&h_psf, CUFFT_CB_ST_COMPLEX, (void **)&z) );
	checkCudaErrors( cufftExecC2C(plan_psf, NULL, h, CUFFT_FORWARD) );

	float *f;
	checkCudaErrors( cudaMalloc((void **)&f, N*N*sizeof(float)) );

	complex_mod<<<N, N>>>(h, f);
	cv::Mat B(N, N, CV_32FC1);
	checkCudaErrors( cudaMemcpy(B.data, f, N*N*sizeof(float), cudaMemcpyDeviceToHost) );
	imshow(B);

	cufftDestroy(plan_psf);

	return 0;
}

/*

    // load in test image
    cv::Mat A = cv::imread("test_square.bmp", CV_LOAD_IMAGE_GRAYSCALE);
    A.convertTo(A, CV_32FC1);
//    imshow(A);

    // relevant! http://arrayfire.com/zero-copy-on-tegra-k1/
    // cudaSetDeviceFlags(cudaDeviceMapHost);

	complex *g, *h;
//	float *f;
	cudaMalloc((void **)&g, N*N*sizeof(complex));
	cudaMalloc((void **)&h, N*N*sizeof(complex));
//	cudaMalloc((void **)&f, N*N*sizeof(float));

	// these two take a long time...
	float *R_d, *R_h;
	cudaMallocHost((void **)&R_h, num_slices*N*N*sizeof(float));
	cudaMalloc((void **)&R_d, num_slices*N*N*sizeof(float));

	float z_h[num_slices];
	float *z_d;
	cudaMalloc((void **)&zs, num_slices*sizeof(float));
	for (int k = 0; k < num_slices; k++) z_h[k] = z_min + dz*k;
	cudaMemcpy(z_d, z_h, num_slices*sizeof(float), cudaMemcpyHostToDevice);
	// now pass this to callback

	cufftHandle plan, plan_mul, plan_psf;
	cufftPlan2d(&plan, N, N, CUFFT_C2C); // cufftplanmany for batch...
	cufftPlan2d(&plan_psf, N, N, CUFFT_C2C);
	cufftPlan2d(&plan_mul, N, N, CUFFT_C2C);
	cufftXtSetCallback(plan_psf, (void **)&_psf_ptr, CUFFT_CB_ST_COMPLEX, (void **)&h);
	cufftXtSetCallback(plan_mul, (void **)&_mul_ptr, CUFFT_CB_ST_COMPLEX, (void **)&h);

//	construct_psf_4fold<<<N/2, N/2>>>(50, g); // 0.40s for 100
//	mod<<<N, N>>>(g, f); // 0.20s for 100
//	cv::Mat B(N, N, CV_32FC1);
//	cudaMemcpy(B.data, f, N*N*sizeof(float), cudaMemcpyDeviceToHost);
//	imshow(B);

	// some behaviour I can't explain here.
	// results look great the first time
	// second time... output is corrupted (!?!)
	for (int n_frame = 0; n_frame < num_frames; n_frame++)
	{
		// transfer image to device, using padding to add imaginary channel
		 cudaMemcpy2D(h, sizeof(complex), A.data, sizeof(float), sizeof(float), N*N, cudaMemcpyHostToDevice);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		cufftExecC2C(plan, (cufftComplex *)h, (cufftComplex *)h, CUFFT_FORWARD);
		// this is subtle - shifting in conjugate domain means we don't need to FFT shift later
		frequency_shift<<<N, N>>>(h);

		float ms = 0;

		for (int k = 0; k < num_slices; k++)
		{
			float z = z_min + dz*k;

			// performance seems better with more blocks, fewer threads
			// this actually has 8-fold symmetry. rotate a symmetric matrix 4 times about the origin
			construct_psf_4fold<<<N/2, N/2>>>(z, g); // 0.40s for 100

			cufftExecC2C(plan, (cufftComplex *)g, (cufftComplex *)g, CUFFT_FORWARD); // 0.28s for 100

//			multiply_inplace<<<N, N>>>(g, h); // 1.38s for 100!?! why so slow???

//			cudaEvent_t start, stop;
//			cudaEventCreate(&start);
//			cudaEventCreate(&stop);
//			cudaEventRecord(start);

			cufftExecC2C(plan_mul, (cufftComplex *)g, (cufftComplex *)g, CUFFT_INVERSE); // 0.28s for 100
			// ordinarily would need to invert phase now, to compensate for earlier (and complete the FFT shift)...

//			cudaDeviceSynchronize();
//			cudaEventRecord(stop);
//			cudaEventSynchronize(stop);
//			float ms_ = 0;
//			cudaEventElapsedTime(&ms_, start, stop);
//			ms += ms_;

			// ... but since we're just taking the modulus, we don't care about phase
			complex_mod<<<N, N>>>(g, R_d + N*N*k); // 0.20s for 100
		}

		// cudaDestroyTextureObject(tex);

		// do some reduction on the images
		// ...

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms, start, stop);

		std::cout << ms << "ms" << std::endl;

		cudaMemcpy(R_h, R_d, num_slices*N*N*sizeof(float), cudaMemcpyDeviceToHost);
	}

	// free pointers
	cudaFree(g);
	cudaFree(h);
	cudaFree(R_d);

	for (int k = 0; k < num_slices; k++)
	{
		cv::Mat B(N, N, CV_32FC1, R_h + k*N*N);
		imshow(B);
	}

	cudaFree(R_h);

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

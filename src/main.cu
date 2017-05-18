/*
 *
 * Proof-of-concept for GPU holographic deconvolution.
 * Michael Murphy, May 2017
 * Yip Lab
 *
 */

// ~20ms runtimes earlier were wrong... ArrayFire has lazy evaluation, and eval() is NON-blocking!
// record for 100 slices now is ~1.25sec

// this is really slow compared to arrayfire!?!
// FFTs faster, but other stuff sucks!

// stuff to try:
// - batching: element-wise stuff and FFTs (cufftplanmany)
// - out-of-place FFTs and multiply
// - async

// run half as many kernels, use vectorized instructions...
// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
// http://stackoverflow.com/questions/26676806/efficiency-of-cuda-vector-types-float2-float3-float4

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <cufft.h>
//#include <cuda_runtime.h>
//#include <cublas_v2.h>

#define N 1024
#define DX (5.32 / 1024)
#define DY (6.66 / 1280)
#define LAMBDA0 0.000488

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
// note that answer will be off by a real-valued factor
__global__
void construct_psf(float z, cuFloatComplex *g)
{
	int i = threadIdx.x;
	int j = blockIdx.x; // blockDim shall equal N

	float x = (i * (float)N/(float)(N-1) - N/2) * DX;
	float y = (j * (float)N/(float)(N-1) - N/2) * DX;

	float r = -2 * norm3df(x, y, z) / LAMBDA0;

	// exp(ix) = cos(x) + isin(x)
	float re, im;
	sincospif(r, &im, &re);

	// re(iz) = -im(z), im(iz) = re(z)
	// fix the -1 thing here...
	g[i*N+j] = make_cuFloatComplex(im / r, -re / r);
}

// In-place element-wise complex multiply: z[i] <- z[i]*w[i]
__global__
void multiply_inplace(cuFloatComplex *z, const __restrict__ cuFloatComplex *w)
{
	int ij = blockDim.x * blockIdx.x + threadIdx.x;

	// re(zw) = re(z)re(w) - im(z)im(w), im(zw) = re(z)im(w) + im(z)re(z)
	z[ij] = cuCmulf(z[ij], w[ij]);
}

// complex modulus, FFT shift
__global__
void mod_shift(const __restrict__ cuFloatComplex *z, float *r)
{
	int i = threadIdx.x;
	int j = blockIdx.x; // blockDim shall equal N

	int ii = (i + N/2) % N;
	int jj = (j + N/2) % N;

	const cuFloatComplex z_ij = z[i*N+j];
	r[ii*N+jj] = hypotf(z_ij.x, z_ij.y);
}

int main(void)
{
	int num_frames = 1;
	int num_zs = 100;
	float z_min = 30;
	float dz = 1;

    // load in test image
    cv::Mat A = cv::imread("test_square.bmp", CV_LOAD_IMAGE_GRAYSCALE);
    A.convertTo(A, CV_32FC1);

    // relevant! http://arrayfire.com/zero-copy-on-tegra-k1/
    // cudaSetDeviceFlags(cudaDeviceMapHost);

    cudaDeviceReset();

	cuFloatComplex *g, *h;
	cudaMalloc((void **)&g, N*N*sizeof(cuFloatComplex));
	cudaMalloc((void **)&h, N*N*sizeof(cuFloatComplex));

	float *R_d, *R_h;
	cudaMallocHost((void **)&R_h, num_zs*N*N*sizeof(float));
	cudaMalloc((void **)&R_d, num_zs*N*N*sizeof(float));

	cufftHandle plan;
	cufftPlan2d(&plan, N, N, CUFFT_C2C); // cufftplanmany for batch...

//	cublasHandle_t handle;
//	cublasCreate(&handle);

	// some behaviour I can't explain here.
	// results look great the first time
	// second time... output is corrupted (!?!)

	for (int n_frame = 0; n_frame < num_frames; n_frame++)
	{

		// transfer image to device, adding in imaginary channel
		A = A + 0.;
		cudaMemcpy2D(h, sizeof(cuFloatComplex), A.data, sizeof(float), sizeof(float), N*N, cudaMemcpyHostToDevice);

//		cudaEventRecord(start);

		cufftExecC2C(plan, (cufftComplex *)h, (cufftComplex *)h, CUFFT_FORWARD);

		float ms = 0;
//		cuFloatComplex a = make_cuFloatComplex(1.f, 1.f);

		for (int k = 0; k < num_zs; k++)
		{
			float z = z_min + dz*k;

			construct_psf<<<N, N>>>(z, g); // 1.09s for 100

			cufftExecC2C(plan, (cufftComplex *)g, (cufftComplex *)g, CUFFT_FORWARD); // 0.28s for 100

			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start);

			multiply_inplace<<<N, N>>>(g, h); // 1.38s for 100 (!?!?!?!?!?! FFT is n2logn, this is n2!!!)

			cudaDeviceSynchronize();
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			float ms_ = 0;
			cudaEventElapsedTime(&ms_, start, stop);
			ms += ms_;

			cufftExecC2C(plan, (cufftComplex *)g, (cufftComplex *)g, CUFFT_INVERSE); // 0.28s for 100

			mod_shift<<<N, N>>>(g, R_d + N*N*k); // 0.95s for 100
		}

		std::cout << ms << "ms" << std::endl;

		// do some reduction on the images
		// ...

//		cudaEventRecord(stop);
//		cudaEventSynchronize(stop);
//		float ms = 0;
//		cudaEventElapsedTime(&ms, start, stop);
//		std::cout << ms << "ms" << std::endl;

		cudaMemcpy(R_h, R_d, num_zs*N*N*sizeof(float), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();
	}

	// free pointers
	cudaFree(g);
	cudaFree(h);
	cudaFree(R_d);

	for (int k = 0; k < num_zs; k++)
	{
		cv::Mat B(N, N, CV_32FC1, R_h + k*N*N);
		imshow(B);
	}

	cudaFree(R_h);

	return 0;
}

/*

#include <arrayfire.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>

#define NO_BATCH

typedef struct {
	int N;
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
void setup_vars(params_t params, af::array *grid, af::array *spacings)
{
	int n_step = (params.d_max - params.d_min) / params.d_step;
    // build (symmetric?) path-length grid
	af::array df = (af::iota(af::dim4(params.N, 1)) * (float)(params.N)/(float)(params.N-1)-params.N/2);
    *grid = af::tile(af::pow(df * params.del_x, 2), 1, params.N) + af::tile(af::pow(df * params.del_y, 2).T(), params.N, 1);
    *spacings = af::pow(af::range(af::dim4(n_step)) * params.d_step + params.d_min, 2);
}

// Expands input image into slices, performs a reduction, and writes the result out.
void process_image(params_t params, af::array &img, float *out_ptr, af::array &x_cube, af::array &grid, af::array &spacings)
{
	af::array x;

	int n_step = spacings.dims(0);
    af::cfloat k0 = {0., (float)(-2. * af::Pi / params.lambda0)};
    af::cfloat k1 = {0., (float)(1. / params.lambda0)};

	// FFT the input image
	af::array h_f = af::fft2(img);

	// process in batches to fit in memory
	// ... but this seems to entirely occupy the Tegra...
#ifdef NO_BATCH
	for (int i = 0; i < n_step; i++)
#else
	for (int j = 0; j < n_step; j += params.n_batch)
#endif
	{
#ifndef NO_BATCH
		gfor (af::seq i, j, min(n_step, j + params.n_batch) - 1)
#endif
		{
#ifdef NO_BATCH
			x = af::sqrt(grid + params.d_min + i * params.d_step); // ~0.15sec
#else
			x = af::sqrt(grid + spacings(i)); // r  // ~0.15sec
#endif
			x = k1 * af::exp(k0 * x) / x; // g
			af::fft2InPlace(x); // g_f // 0.3sec

			// note here: g is an even function
			// so F(g) is real valued and even
			// (i.e. can I just take FFT of half of g?)

			x = x * h_f; // gh_f
			af::ifft2InPlace(x); // h // 0.4sec
			x = af::abs(x); // |h|
			x = af::shift(x, x.dims(0)/2, x.dims(1)/2); // FFT shift

			x_cube(af::span, af::span, i) = x; // / af::max<float>(x) * 255.; // compression probably unnecessary?
		}
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
    params.N = 1024; // resolution in pixels
    params.lambda0 = 0.000488; // wavelength
    params.del_x = 5.32 / 1024; // horizontal frequency spacing
    params.del_y = 6.66 / 1280; // vertical frequency spacing
    params.d_max = 130; // max distance from CCD to object, mm
    params.d_min = 30; // min distance from CCD to object, mm
    params.d_step = 2; // step size in mm
    params.n_batch = 1; // number of frames per batch; best performance with 1!?!?
    params.n_frames = 3; // currently, just repeats the analysis

    // load in test image
    cv::Mat mat = cv::imread("test_square.bmp", CV_LOAD_IMAGE_GRAYSCALE);
    mat.convertTo(mat, CV_32FC1);

    // simulate a DMA buffer on the GPU, i.e. feeding in from video camera
    // will copy in the images as part of the main loop, simulate 'streaming'
    float *img_ptr;
    cudaMalloc((void **)&img_ptr, params.N * params.N * sizeof(float));

    // allocate the matrix using our predefined staging area
    af::array img(params.N, params.N, img_ptr, afDevice);
    af::eval(img);

    // pin buffer on the host
    float *h_ptr = af::pinned<float>(params.N * params.N * params.n_frames);

    af::array grid;
    af::array spacings;
    setup_vars(params, &grid, &spacings);

    // allocate this just once and reuse, it's huge
    int n_step = spacings.dims(0);
    af::array x_cube(params.N, params.N, n_step, f32);

    for (int k = 0; k < params.n_frames; k++)
    {
		// 'copy the image' - these would be successive frames in reality, and would probably live on GPU
		// i.e. this copy would not happen
		mat = mat + 0.; // no possibility of caching
		cudaMemcpy(img_ptr, mat.data, params.N * params.N * sizeof(float), cudaMemcpyHostToDevice);

		// expand the image into slices, do a reduction, save result to h_ptr
		af::timer::start();
		process_image(params, img, h_ptr + params.N * params.N * k, x_cube, grid, spacings);
		std::cout << af::timer::stop() << std::endl;
    }

//    cv::namedWindow("Display window", cv::WINDOW_NORMAL); // Create a window for display.
//    for (int i = 0; i < n_step; i++)
//    {
//        cv::Mat mat(cv::Size(1024, 1024), CV_32FC1, h_ptr + i * N * N);
//        cv::normalize(mat, mat, 1.0, 0.0, cv::NORM_MINMAX, -1);
//        cv::imshow("Display window", mat); // Show our image inside it.
//        cv::waitKey(0);
//    }

    return 0;
}

*/

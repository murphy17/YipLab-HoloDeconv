/*
 *
 * Proof-of-concept for GPU holographic deconvolution.
 * Michael Murphy, May 2017
 * Yip Lab
 *
 */

#include <arrayfire.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>

#define fftshift(in) af::shift(in, in.dims(0)/2, in.dims(1)/2)

int main(void)
{
    //cudaDeviceReset();

    int N = 1024; // resolution in pixels
    float lambda0 = 0.000488; // wavelength
    float del_x = 5.32 / 1024; // horizontal frequency spacing
    float del_y = 6.66 / 1280; // vertical frequency spacing
    float d_max = 130; // max distance from CCD to object, mm
    float d_min = 30; // min distance from CCD to object, mm
    float d_step = 1; // step size in mm
    int batch_size = 20;
    int n_step = (int)((d_max - d_min) / d_step + 0.5);

    af::cfloat k0 = {0., (float)(-2. * af::Pi / lambda0)};
    af::cfloat k1 = {0., (float)(1. / lambda0)};

    cv::Mat mat = cv::imread("test_square.bmp", CV_LOAD_IMAGE_GRAYSCALE);
    mat.convertTo(mat, CV_32FC1);

    // simulate a DMA buffer on the GPU, i.e. feeding in from video camera
    // will copy in the images as part of the main loop, simulate 'streaming'
    int sz = N * N * sizeof(float);
    const float *img_ptr;
    cudaMalloc((void **)&img_ptr, sz);

    // allocate the matrix using our predefined staging area
    af::array img(N, N, img_ptr, afDevice);

    // build (symmetric?) frequency grid
    af::array xx = (af::iota(af::dim4(N, 1)) * (float)(N)/(float)(N-1)-N/2) * del_x;
    af::array yy = (af::iota(af::dim4(1, N)) * (float)(N)/(float)(N-1)-N/2) * del_y;
    af::array grid = af::tile(af::pow(xx,2), 1, N) + af::tile(af::pow(yy,2), N, 1);

    // pin a buffer on the host for the whole cube
    float *h_ptr = af::pinned<float>(N * N * n_step);

    af::array x_batch(N, N, batch_size, f32);
    af::array d_batch(batch_size);
    af::array dd = af::pow(af::range(af::dim4(n_step)) * d_step + d_min, 2);

    for (int k = 0; k < 5; k++)
    {
		// 'copy the image' - these would be successive frames in reality, and would probably live on GPU
		// i.e. this copy would not happen
		mat = mat + 0.; // no possibility of caching, I hope
		cudaMemcpy((void *)img_ptr, (void *)mat.data, sz, cudaMemcpyHostToDevice);
		// af::array img(N, N, img_ptr, afDevice);

		af::timer t = af::timer::start();

		// FFT the input image
		af::array h_f = af::fft2(img);

		// make sure it's not just reusing the cache...
		h_f += 0.;

		for (int j = 0; j < n_step; j += batch_size)
		{
			int jj = min(n_step, j + batch_size);

			int n_batch = jj - j;

			d_batch = dd(af::seq(j, jj-1));

			gfor (af::seq i, 0, n_batch-1) // http://arrayfire.org/docs/page_gfor.htm
			{
				af::array x(N, N, c32);

				x = af::sqrt(grid + d_batch(i)); // r
				x = k1 * af::exp(k0 * x) / x; // g
				af::fft2InPlace(x); // g_f
				x = x * h_f; // gh_f
				af::ifft2InPlace(x); // h

				x = af::abs(x); // |h|
				x = fftshift(x);

				x_batch(af::span, af::span, i) = x; // / af::max<float>(x);
			}

			// this takes orders of magnitude longer than the computation
			// transfer batch to host
			// ... good news is that you probably won't need to do this!
			// (next stage is finding optimal d, then segmenting + 3D volume)
			x_batch.eval();
			// x_batch.host<float>(h_ptr + j * N * N);
			//cudaMemcpy( \
			//   h_ptr + j * N * N, \
			//   x_batch.device<float>(), \
			//   n_batch * sz, \
			//  cudaMemcpyDeviceToHost);
			//x_batch.unlock();
		}
		printf("elapsed seconds: %g\n", af::timer::stop(t));
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

/*
 * util.hpp
 *
 *  Created on: May 30, 2017
 *      Author: michaelmurphy
 */

#ifndef UTIL_HPP_
#define UTIL_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "half_math.cuh"

#if CV_MAJOR_VERSION == 3
#include <opencv2/core/cuda.hpp>
namespace cv_gpu = cv::cuda;
#else
#include <opencv2/gpu/gpu.hpp>
namespace cv_gpu = cv::gpu;
#endif

__global__
void h2_to_f2(half2 *h, float2 *f)
{
	f[blockIdx.x*blockDim.x+threadIdx.x] = h[blockIdx.x*blockDim.x+threadIdx.x];
}
__global__
void h_to_f(half *h, float *f)
{
	f[blockIdx.x*blockDim.x+threadIdx.x] = h[blockIdx.x*blockDim.x+threadIdx.x];
}

void view_gpu(half *x, int elements, bool log)
{
	checkCudaErrors( cudaDeviceSynchronize() );

	float *x_f;
	checkCudaErrors( cudaMalloc(&x_f, elements*sizeof(float)) );
	h_to_f<<<elements / 1024, 1024>>>(x, x_f);
	float *x_h = new float[elements];
	checkCudaErrors( cudaMemcpy(x_h, x_f, elements*sizeof(float), cudaMemcpyDeviceToHost) );

	cv::Mat A(1024, 1024, CV_32FC1, x_h);
	if (log)
		cv::log(A, A);
	cv::normalize(A, A, 1.0, 0.0, cv::NORM_MINMAX, -1);

	cv::namedWindow("Display window", cv::WINDOW_NORMAL); // Create a window for display.
	cv::imshow("Display window", A); // Show our image inside it.
	cv::waitKey(0);

	checkCudaErrors( cudaFree(x_f) );
	delete x_h;
}

void view_gpu(half2 *x, int elements, bool log)
{
	checkCudaErrors( cudaDeviceSynchronize() );

	float2 *x_f;
	checkCudaErrors( cudaMalloc(&x_f, elements*sizeof(float2)) );
	h2_to_f2<<<elements / 1024, 1024>>>(x, x_f);
	float2 *x_h = new float2[elements];
	checkCudaErrors( cudaMemcpy(x_h, x_f, elements*sizeof(float2), cudaMemcpyDeviceToHost) );

	cv::Mat A(1024, 1024, CV_32FC2, x_h);
	if (A.channels() == 2)
	{
		cv::Mat channels[2];
		cv::split(A, channels);
		cv::magnitude(channels[0], channels[1], A);
	}
	A.convertTo(A, CV_32FC1);
	if (log)
		cv::log(A, A);

	cv::normalize(A, A, 1.0, 0.0, cv::NORM_MINMAX, -1);

//	std::cout << A << std::endl;

	cv::namedWindow("Display window", cv::WINDOW_NORMAL); // Create a window for display.
	cv::imshow("Display window", A); // Show our image inside it.
	cv::waitKey(0);

	checkCudaErrors( cudaFree(x_f) );
	delete x_h;
}

// Convenience method for plotting
void imshow(cv::Mat in)
{
	cv::namedWindow("Display window", cv::WINDOW_NORMAL); // Create a window for display.
	cv::Mat out = in;
	cudaDeviceSynchronize();
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

void imshow(cv_gpu::GpuMat in) //, bool log=false)
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
//	if (log)
//		cv::log(out, out);
	cv::normalize(out, out, 1.0, 0.0, cv::NORM_MINMAX, -1);
	cv::imshow("Display window", out); // Show our image inside it.
	cv::waitKey(0);
}




#endif /* UTIL_HPP_ */

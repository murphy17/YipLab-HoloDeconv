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

#if CV_MAJOR_VERSION == 3
#include <opencv2/core/cuda.hpp>
namespace cv_gpu = cv::cuda;
#else
#include <opencv2/gpu/gpu.hpp>
namespace cv_gpu = cv::gpu;
#endif

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

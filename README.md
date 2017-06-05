## HoloDeconv

Holographic microscopy deconvolution via GPU. Written in native CUDA, CUFFT. Uses OpenCV for visualization.

Best times so far: 330ms/cube on Tegra (FP16), 36ms/cube on Titan (FP32). (100 1024x1024x1 slices per cube)

TODO: downsampling; FFT thresholding (denoising?); refactoring; review FP16 wrapper; de-hard-code params
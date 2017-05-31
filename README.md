## HoloDeconv

Holographic microscopy deconvolution via GPU. Written in native CUDA, CUFFT. Uses OpenCV for visualization.

Best times so far: 330ms / 100 slices on Tegra (FP16), 36ms / 100 slices on Titan (FP32).
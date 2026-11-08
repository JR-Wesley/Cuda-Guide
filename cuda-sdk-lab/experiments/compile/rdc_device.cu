// This device function is defined in a different CUDA translation unit.
extern "C" __device__ float rdc_scale(float x) { return 2.0f * x; }

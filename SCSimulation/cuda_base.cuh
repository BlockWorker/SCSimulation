#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <exception>

#define cu(call) do { cudaError_t ___err = (call); if (___err != cudaSuccess) throw CudaError(___err); } while (false)

class CudaError : public std::exception
{
public:
	const cudaError_t error;

	CudaError(cudaError_t error) : std::exception(cudaGetErrorString(error)), error(error) {

	}

};

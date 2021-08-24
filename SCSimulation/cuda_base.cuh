#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdexcept>

#define cu(call) do { cudaError_t ___err = (call); if (___err != cudaSuccess) throw CudaError(___err); } while (false)

class CudaError : public std::runtime_error
{
public:
	const cudaError_t error;

	CudaError(cudaError_t error) : std::runtime_error("CUDA error"), error(error) {

	}

	virtual ~CudaError() {

	}

	virtual const char* what() const noexcept override {
		return cudaGetErrorString(error);
	}

};

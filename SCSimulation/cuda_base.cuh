#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <exception>

//wrapper macro to detect CUDA errors
#define cu(call) do { cudaError_t ___err = (call); if (___err != cudaSuccess) throw scsim::CudaError(___err); } while (false)
#define cu_ignore_error(call) do { (call); cudaGetLastError(); } while (false)

namespace scsim {

	class CudaError : public std::exception
	{
	public:
		const cudaError_t error;

		CudaError(cudaError_t error) : std::exception(cudaGetErrorString(error)), error(error) {

		}

	};

}

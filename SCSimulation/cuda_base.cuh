#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <exception>

//wrapper macro to detect errors in CUDA API calls
#define cu(call) do { cudaError_t ___err = (call); if (___err != cudaSuccess) throw scsim::CudaError(___err); } while (false)

//wrapper macro for CUDA API calls that should ignore any errors generated
#define cu_ignore_error(call) do { (call); cudaGetLastError(); } while (false)

//macro to check for errors in a previously executed kernel and synchronize host and device
#define cu_kernel_errcheck() do { cu(cudaPeekAtLastError()); cu(cudaDeviceSynchronize()); } while (false);

//macro to check for errors in a previously executed kernel without host/device synchronization
#define cu_kernel_errcheck_nosync() cu(cudaPeekAtLastError())

namespace scsim {

	class CudaError : public std::exception
	{
	public:
		const cudaError_t error;

		CudaError(cudaError_t error) : std::exception(cudaGetErrorString(error)), error(error) {

		}

	};

}

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdexcept>
#include <stdio.h>

//wrapper macro to detect errors in CUDA API calls
#define cu(call) do { cudaError_t ___err = (call); if (___err != cudaSuccess) throw scsim::CudaError(___err, __FILE__, __LINE__); } while (false)

//wrapper macro for CUDA API calls that should ignore any errors generated
#define cu_ignore_error(call) do { (call); cudaGetLastError(); } while (false)

//macro to check for errors in a previously executed kernel and synchronize host and device
#define cu_kernel_errcheck() do { cu(cudaPeekAtLastError()); cu(cudaDeviceSynchronize()); } while (false);

//macro to check for errors in a previously executed kernel without host/device synchronization
#define cu_kernel_errcheck_nosync() cu(cudaPeekAtLastError())

namespace scsim {

	class CudaError : public std::runtime_error
	{
	public:
		const cudaError_t error;

		CudaError(cudaError_t error, const char* file, int line) : std::runtime_error("CUDA Error"), error(error) {
			char result[1024];
			snprintf(result, 1024, "CUDA error: %s (%s)\n  at %s, line %d", cudaGetErrorString(error), cudaGetErrorName(error), file, line);
			*this = std::runtime_error(result);
		}

		virtual ~CudaError() {

		}

		CudaError& operator=(runtime_error const& _Other) noexcept
		{
			return (CudaError&)std::runtime_error::operator=(_Other);
		}

	};

}

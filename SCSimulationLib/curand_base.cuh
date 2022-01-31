#pragma once

#include "cuda_base.cuh"
#include "curand.h"
#include <stdio.h>

//wrapper macro to detect errors in cuRAND API calls
#define cur(call) do { curandStatus_t ___stat = (call); if (___stat != CURAND_STATUS_SUCCESS) throw scsim::CurandError(___stat, __FILE__, __LINE__); } while (false)

namespace scsim {

	class CurandError : public std::runtime_error
	{
	public:
		const curandStatus_t error;

		CurandError(curandStatus_t error, const char* file, int line) : std::runtime_error("cuRAND Error"), error(error) {
			char result[1024];
			snprintf(result, 1024, "cuRAND error: %d\n  at %s, line %d", (int)error, file, line);
			*this = std::runtime_error(result);
		}

		virtual ~CurandError() {

		}

		CurandError& operator=(runtime_error const& _Other) noexcept
		{
			return (CurandError&)std::runtime_error::operator=(_Other);
		}

	};

}

#pragma once

#include "cuda_base.cuh"
#include "curand.h"
#include <stdio.h>

//wrapper macro to detect errors in cuRAND API calls
#define cur(call) do { curandStatus_t ___stat = (call); if (___stat != CURAND_STATUS_SUCCESS) throw scsim::CurandError(___stat, __FILE__, __LINE__); } while (false)

namespace scsim {

	class CurandError : public std::exception
	{
	public:
		const curandStatus_t error;

		CurandError(curandStatus_t error, const char* file, int line) : std::exception(), error(error) {
			char result[1024];
			snprintf(result, 1024, "cuRAND error: %d\n  at %s, line %d", (int)error, file, line);
			*this = std::exception(result);
		}

		virtual ~CurandError() {

		}

		CurandError& operator=(exception const& _Other) noexcept
		{
			return (CurandError&)std::exception::operator=(_Other);
		}

	};

}

#pragma once

#include "cuda_base.cuh"
#include "curand.h"
#include <stdio.h>

//wrapper macro to detect cuRAND errors
#define cur(call) do { curandStatus_t ___stat = (call); if (___stat != CURAND_STATUS_SUCCESS) throw scsim::CurandError(___stat); } while (false)

namespace scsim {

	class CurandError : public std::exception
	{
	public:
		const curandStatus_t error;

		CurandError(curandStatus_t error) : std::exception(), error(error) {
			char result[18];
			snprintf(result, 18, "cuRAND error: %d", (int)error);
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

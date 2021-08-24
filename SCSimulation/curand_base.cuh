#pragma once

#include "cuda_base.cuh"
#include "curand.h"

#define cur(call) do { curandStatus_t ___stat = (call); if (___stat != CURAND_STATUS_SUCCESS) throw CurandError(___stat); } while (false)

class CurandError : public std::runtime_error
{
public:
	const curandStatus_t error;

	CurandError(curandStatus_t error) : std::runtime_error("cuRAND error"), error(error) {

	}

	virtual ~CurandError() {

	}

	virtual const char* what() const noexcept override {
		char result[18];
		snprintf(result, 18, "cuRAND error: %d", (int)error);
		return result;
	}

};

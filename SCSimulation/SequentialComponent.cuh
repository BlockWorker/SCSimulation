#pragma once

#include <stdint.h>

#include "CircuitComponent.cuh"

#define SEQUENTIAL_FIXED_STATE_SIZE 1

#define takebit(x) (((x) & 0x80000000u) > 0)
#define putbit(x, v) ((x) |= (v) ? 1 : 0)

class SequentialComponent : public CircuitComponent
{
public:
	const size_t state_size;

	SequentialComponent(uint32_t num_inputs, uint32_t num_outputs, size_t state_size, uint32_t type, size_t size, size_t align);
	virtual ~SequentialComponent();

	virtual void reset_state() override;

protected:
	uint32_t fixed_state[SEQUENTIAL_FIXED_STATE_SIZE] = { 0 };
	uint32_t* var_state_host;
	uint32_t* var_state_dev;
	const bool state_fixed;

	__host__ __device__ uint32_t* state_ptr() const {
		if (state_fixed) return (uint32_t*)this->fixed_state;
#ifdef __CUDA_ARCH__
		else return var_state_dev;
#else
		else return var_state_host;
#endif
	}

	virtual void copy_state_host_to_device() override;
	virtual void copy_state_device_to_host() override;

	virtual void init_with_circuit(StochasticCircuit* circuit) override;

};
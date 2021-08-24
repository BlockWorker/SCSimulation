#pragma once

#include <stdint.h>

#include "CircuitComponent.cuh"

#define takebit(x) (((x) & 0x80000000u) > 0)
#define putbit(x, v) ((x) |= (v) ? 1 : 0)

class SequentialComponent : public CircuitComponent
{
public:
	const size_t state_size;

	SequentialComponent(uint32_t num_inputs, uint32_t num_outputs, size_t state_size, uint32_t type, size_t size);
	virtual ~SequentialComponent();

	virtual void reset_state() override;

protected:
	uint32_t* state_host;
	uint32_t* state_dev;

	virtual void copy_state_host_to_device() override;
	virtual void copy_state_device_to_host() override;

	virtual void init_with_circuit(StochasticCircuit* circuit) override;

};
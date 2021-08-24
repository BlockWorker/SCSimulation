#pragma once

#include <stdint.h>
#include <typeinfo>

#include "CircuitComponent.cuh"

class CombinatorialComponent : public CircuitComponent
{
public:
	CombinatorialComponent(uint32_t num_inputs, uint32_t num_outputs, uint32_t type, size_t size) : CircuitComponent(num_inputs, num_outputs, type & 0x7fffffffu, size) { }
	virtual ~CombinatorialComponent() { }

	virtual void reset_state() override final { }

protected:
	virtual void copy_state_host_to_device() override final { }
	virtual void copy_state_device_to_host() override final { }

};
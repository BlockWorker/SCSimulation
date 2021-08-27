#pragma once

#include <stdint.h>
#include "CircuitComponent.cuh"
#include "dll.h"

namespace scsim {

	class SCSIMAPI CombinatorialComponent : public CircuitComponent
	{
	public:
		CombinatorialComponent(uint32_t num_inputs, uint32_t num_outputs, uint32_t type, size_t size, size_t align);
		virtual ~CombinatorialComponent();

		virtual void reset_state() override final { }

		virtual void copy_state_host_to_device() override final { }
		virtual void copy_state_device_to_host() override final { }

	};

}

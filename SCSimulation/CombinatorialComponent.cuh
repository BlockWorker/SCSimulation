﻿#pragma once

#include <stdint.h>
#include "CircuitComponent.cuh"
#include "dll.h"

namespace scsim {

	class SCSIMAPI CombinatorialComponent : public CircuitComponent
	{
	public:
		/// <param name="type">Unique component type index/hash, use typehash(Type) macro in circuit_component_defines.h</param>
		/// <param name="size">Memory size of component, use sizeof(Type)</param>
		/// <param name="align">Memory alignment of component, use alignof(Type)</param>
		CombinatorialComponent(uint32_t num_inputs, uint32_t num_outputs, uint32_t type, size_t size, size_t align);

		virtual ~CombinatorialComponent();

		virtual void reset_state() override final { } //no state -> empty reset and copy functions

		virtual void copy_state_host_to_device() override final { }
		virtual void copy_state_device_to_host() override final { }

	};

}

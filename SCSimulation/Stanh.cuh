#pragma once

#include <stdint.h>
#include "SequentialComponent.cuh"
#include "dll.h"

namespace scsim {

	class SCSIMAPI Stanh : public SequentialComponent
	{
	public:
		Stanh(uint32_t input, uint32_t output, uint32_t k);

		virtual void reset_state() override;

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

	protected:
		virtual void link_devstep() override;

	private:
		const uint32_t k;

	};

}

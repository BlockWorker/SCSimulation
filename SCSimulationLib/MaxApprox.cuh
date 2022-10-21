#pragma once

#include <stdint.h>
#include <initializer_list>
#include "CombinatorialComponent.cuh"
#include "circuit_component_defines.cuh"

namespace scsim {

	/// <summary>
	/// Approximate maximum function, selecting the maximum bitstream word-by-word (32-bit sections)
	/// </summary>
	class SCSIMAPI MaxApprox : public CombinatorialComponent
	{
	public:
		/// <param name="inputs">pointer to array of input net indices</param>
		MaxApprox(uint32_t _num_inputs, uint32_t* inputs, uint32_t output, StochasticCircuitFactory* factory);

		/// <param name="first_input">first input net index, further inputs assigned consecutive indices</param>
		MaxApprox(uint32_t _num_inputs, uint32_t first_input, uint32_t output, StochasticCircuitFactory* factory);

		/// <param name="inputs">list of input net indices</param>
		MaxApprox(std::initializer_list<uint32_t> inputs, uint32_t output, StochasticCircuitFactory* factory);

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

		decl_device_statics()

	private:
		const uint32_t num_inputs;

	};

}

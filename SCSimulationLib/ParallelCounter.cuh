#pragma once

#include <stdint.h>
#include <initializer_list>
#include "CombinatorialComponent.cuh"
#include "circuit_component_defines.cuh"

namespace scsim {

	/// <summary>
	/// (Exact) Parallel Counter
	/// </summary>
	class SCSIMAPI ParallelCounter : public CombinatorialComponent
	{
	public:
		/// <param name="inputs">pointer to array of input net indices</param>
		/// <param name="outputs">pointer to array of output net indices (LSB first), must be sufficiently long for required number of outputs (floor(log2(num_inputs)) + 1)</param>
		ParallelCounter(uint32_t _num_inputs, uint32_t* inputs, uint32_t* outputs, StochasticCircuitFactory* factory);

		/// <param name="first_input">first input net index, further inputs assigned consecutive indices</param>
		/// <param name="first_output">first output net index (LSB), further outputs assigned consecutive indices, must have sufficient nets available (floor(log2(num_inputs)) + 1)</param>
		ParallelCounter(uint32_t _num_inputs, uint32_t first_input, uint32_t first_output, StochasticCircuitFactory* factory);

		/// <param name="inputs">list of input net indices</param>
		/// <param name="outputs">list of output net indices (LSB first), must be sufficiently long for required number of outputs (floor(log2(num_inputs)) + 1)</param>
		ParallelCounter(std::initializer_list<uint32_t> inputs, std::initializer_list<uint32_t> outputs, StochasticCircuitFactory* factory);

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

		decl_device_statics()

	private:
		const uint32_t num_inputs;
		const uint32_t num_outputs;

	};

}

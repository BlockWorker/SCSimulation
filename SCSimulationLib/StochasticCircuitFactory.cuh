#pragma once

#include <stdint.h>
#include <vector>
#include "library_export.h"

namespace scsim {

	class StochasticCircuit;
	class StochasticNumber;
	class CircuitComponent;
	class CombinatorialComponent;

	template<size_t state_size>
	class SequentialComponent;

	class SCSIMAPI StochasticCircuitFactory
	{
	public:
		const bool host_only;

		/// <param name="host_only">Whether the created circuits are only simulated on the host</param>
		StochasticCircuitFactory(bool host_only);
		~StochasticCircuitFactory();

		void reset();

		StochasticCircuit* create_circuit();

		/// <param name="sim_length">Desired simulation time span in bit times</param>
		void set_sim_length(uint32_t sim_length);

		/// <returns>Index of newly added net</returns>
		uint32_t add_net();

		/// <returns>Indices of first and last newly added nets</returns>
		std::pair<uint32_t, uint32_t> add_nets(uint32_t count);

		/// <summary>It is recommended to use the factory_add_component macro to add components instead of calling this function directly.</summary>
		uint32_t add_component(CombinatorialComponent* component);

		/// <summary>It is recommended to use the factory_add_component macro to add components instead of calling this function directly.</summary>
		template<size_t state_size>
		uint32_t add_component(SequentialComponent<state_size>* component) {
			uint32_t index = add_component_internal(component);
			num_seq_comp++;
			return index;
		}

	private:
		friend CircuitComponent;
		
		uint32_t sim_length;
		uint32_t num_nets;
		uint32_t num_comb_comp;
		uint32_t num_seq_comp;
		std::vector<CircuitComponent*> components;
		std::vector<bool> driven_nets;
		size_t max_component_size;
		size_t max_component_align;
		std::vector<uint32_t> component_io;

		uint32_t* net_values_host;
		uint32_t* net_values_dev;
		uint32_t* net_progress_host;
		uint32_t* net_progress_dev;
		uint32_t* components_host_index;
		CircuitComponent** components_host;
		CircuitComponent** components_dev;
		uint32_t* component_progress_host;
		uint32_t* component_progress_dev;
		char* component_array_host;
		char* component_array_dev;
		uint32_t* component_io_host;
		uint32_t* component_io_dev;
		size_t* component_io_offsets_host;
		size_t* component_io_offsets_dev;
		size_t* dev_offset_scratchpad;
		CircuitComponent** dev_pointers_scratchpad;

		StochasticNumber* net_numbers;

		uint32_t add_component_internal(CircuitComponent* component);

	};

}

#pragma once

#include <stdint.h>
#include <vector>
#include "dll.h"

namespace scsim {

	class StochasticCircuit;
	class CircuitComponent;
	class CombinatorialComponent;
	class SequentialComponent;

	class SCSIMAPI StochasticCircuitFactory
	{
	public:
		StochasticCircuitFactory();

		void reset();

		StochasticCircuit* create_circuit();

		/// <param name="host_only">Whether the created circuit should only be simulated on the host</param>
		void set_host_only(bool host_only);

		/// <param name="sim_length">Desired length of the simulation in bits</param>
		void set_sim_length(uint32_t sim_length);

		/// <returns>Index of newly added net</returns>
		uint32_t add_net();

		/// <returns>Indices of first and last newly added nets</returns>
		std::pair<uint32_t, uint32_t> add_nets(uint32_t count);

		void add_component(CombinatorialComponent* component);
		void add_component(SequentialComponent* component);

	private:
		bool host_only;
		uint32_t sim_length;
		uint32_t num_nets;
		uint32_t num_comb_comp;
		uint32_t num_seq_comp;
		std::vector<CircuitComponent*> components;
		std::vector<bool> driven_nets;
		size_t max_component_size;
		size_t max_component_align;

		uint32_t add_component_internal(CircuitComponent* component);

	};

}

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
		~StochasticCircuitFactory();

		void reset();

		StochasticCircuit* create_circuit();

		void set_host_only(bool host_only);
		void set_sim_length(uint32_t sim_length);

		uint32_t add_net();
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

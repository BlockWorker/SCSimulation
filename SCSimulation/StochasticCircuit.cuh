#pragma once

#include <stdint.h>
#include "dll.h"

namespace scsim {

	class CircuitComponent;
	class StochasticNumber;

	class SCSIMAPI StochasticCircuit
	{
	public:
		const bool host_only;

		const uint32_t sim_length;
		const uint32_t sim_length_words;

		const uint32_t num_nets;
		uint32_t* const net_values_host;
		const size_t net_values_host_pitch;
		uint32_t* const net_values_dev;
		const size_t net_values_dev_pitch;
		uint32_t* const net_progress_host;
		uint32_t* const net_progress_dev;

		const uint32_t num_components;
		const uint32_t num_components_comb;
		const uint32_t num_components_seq;
		CircuitComponent** const components_host;
		CircuitComponent** const components_dev;

		StochasticCircuit(uint32_t sim_length, uint32_t num_nets, uint32_t* net_values, uint32_t* net_progress, uint32_t num_components_comb, uint32_t num_components_seq, CircuitComponent** components);

		StochasticCircuit(uint32_t sim_length, uint32_t num_nets, uint32_t* net_values_host, uint32_t* net_values_dev, size_t net_values_dev_pitch, uint32_t* net_progress_host, uint32_t* net_progress_dev,
			uint32_t num_components_comb, uint32_t num_components_seq, CircuitComponent** components_host, CircuitComponent** components_dev, char* component_array_host, size_t component_array_host_pitch,
			char* component_array_dev, size_t component_array_dev_pitch);

		~StochasticCircuit();

		void reset_circuit();

		void set_net_value(uint32_t net, StochasticNumber* value);
		void set_net_value_unipolar(uint32_t net, double value);
		void set_net_value_bipolar(uint32_t net, double value);

		void copy_data_to_device();
		void copy_data_from_device();

		void simulate_circuit_host_only();
		void simulate_circuit();

		StochasticNumber* get_net_value(uint32_t net);
		double get_net_value_unipolar(uint32_t net);
		double get_net_value_bipolar(uint32_t net);

	private:
		bool simulation_finished = false;

		char* const component_array_host;
		const size_t component_array_host_pitch;
		char* const component_array_dev;
		const size_t component_array_dev_pitch;

	};

}

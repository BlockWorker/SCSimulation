﻿#pragma once

#include <stdint.h>
#include "dll.h"

namespace scsim {

	class CircuitComponent;
	class StochasticNumber;
	class StochasticCircuitFactory;

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

		~StochasticCircuit();

		StochasticCircuit(const StochasticCircuit&) = delete;
		StochasticCircuit& operator=(const StochasticCircuit&) = delete;
		StochasticCircuit(StochasticCircuit&&) = delete;
		StochasticCircuit& operator=(StochasticCircuit&&) = delete;

		void reset_circuit();

		void set_net_value(uint32_t net, StochasticNumber* value);
		void set_net_value_unipolar(uint32_t net, double value, uint32_t length);
		void set_net_value_unipolar(uint32_t net, double value);
		void set_net_value_bipolar(uint32_t net, double value, uint32_t length);
		void set_net_value_bipolar(uint32_t net, double value);

		/// <summary>
		/// Set the first [length] bits of the given net to all zeroes or all ones (chosen by "value")
		/// </summary>
		void set_net_value_constant(uint32_t net, bool value, uint32_t length);
		/// <summary>
		/// Set all bits of the given net to all zeroes or all ones (chosen by "value")
		/// </summary>
		void set_net_value_constant(uint32_t net, bool value);

		void copy_data_to_device();
		/// <param name="net_length">How many bits per net to copy to device</param>
		void copy_data_to_device(uint32_t net_length);
		void copy_data_from_device();
		/// <param name="net_length">How many bits per net to copy from device</param>
		void copy_data_from_device(uint32_t net_length);

		void simulate_circuit_host_only();
		/// <summary>
		/// Simulate circuit on the device without copying the circuit state back and forth, requires data to already be synchronized with device
		/// </summary>
		void simulate_circuit_dev_nocopy();
		void simulate_circuit();

		StochasticNumber* get_net_value(uint32_t net);
		double get_net_value_unipolar(uint32_t net);
		double get_net_value_bipolar(uint32_t net);

	private:
		friend StochasticCircuitFactory;
		friend CircuitComponent;

		bool simulation_finished = false;

		char* const component_array_host;
		const size_t component_array_host_pitch;
		char* const component_array_dev;
		const size_t component_array_dev_pitch;
		uint32_t* const component_progress_host;
		uint32_t* const component_progress_dev;
		const uint32_t num_component_types;
		uint32_t* const component_io_host;
		uint32_t* const component_io_dev;
		size_t* const component_io_offsets_host;
		size_t* const component_io_offsets_dev;

		//host-only circuit constructor
		StochasticCircuit(uint32_t sim_length, uint32_t num_nets, uint32_t* net_values, uint32_t* net_progress, uint32_t num_components_comb, uint32_t num_components_seq, CircuitComponent** components,
			uint32_t* component_progress, uint32_t num_component_types, uint32_t* component_io, size_t* component_io_offsets);

		//device-accelerated circuit constructor
		StochasticCircuit(uint32_t sim_length, uint32_t num_nets, uint32_t* net_values_host, uint32_t* net_values_dev, size_t net_values_dev_pitch, uint32_t* net_progress_host, uint32_t* net_progress_dev,
			uint32_t num_components_comb, uint32_t num_components_seq, CircuitComponent** components_host, CircuitComponent** components_dev, char* component_array_host, size_t component_array_host_pitch,
			char* component_array_dev, size_t component_array_dev_pitch, uint32_t* component_progress_host, uint32_t* component_progress_dev, uint32_t num_component_types, uint32_t* component_io_host,
			uint32_t* component_io_dev, size_t* component_io_offsets_host, size_t* component_io_offsets_dev);

		void copy_component_progress_from_device();

	};

}

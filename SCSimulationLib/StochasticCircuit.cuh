﻿#pragma once

#include <stdint.h>
#include "library_export.h"

namespace scsim {

	class CircuitComponent;
	class StochasticNumber;
	class StochasticCircuitFactory;
	class Scheduler;

	/// <summary>
	/// Class representing a stochastic circuit (essentially a graph of nets and components with their states).
	/// Don't try to instantiate this directly - use StochasticCircuitFactory to build circuits instead.
	/// </summary>
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
		uint32_t* const components_host_index;
		CircuitComponent** const components_host;
		CircuitComponent** const components_dev;
		const uint32_t num_component_types;

		StochasticNumber* const net_numbers;

		~StochasticCircuit();

		StochasticCircuit(const StochasticCircuit&) = delete;
		StochasticCircuit& operator=(const StochasticCircuit&) = delete;
		StochasticCircuit(StochasticCircuit&&) = delete;
		StochasticCircuit& operator=(StochasticCircuit&&) = delete;

		void reset_circuit();

		/// <summary>
		/// Set the given net's value to the given SN's value for as many bits as the number provides. The number may not be longer than the simulation time span.
		/// </summary>
		void set_net_value(uint32_t net, StochasticNumber& value);

		/// <summary>
		/// Set the value of the given net in unipolar encoding for the given number of bit times with guaranteed best accuracy.
		/// </summary>
		void set_net_value_unipolar(uint32_t net, double value, uint32_t length);

		/// <summary>
		/// Set the value of the given net in unipolar encoding for the entire simulation time span with guaranteed best accuracy.
		/// </summary>
		void set_net_value_unipolar(uint32_t net, double value);

		/// <summary>
		/// Set the value of the given net in bipolar encoding for the given number of bit times with guaranteed best accuracy.
		/// </summary>
		void set_net_value_bipolar(uint32_t net, double value, uint32_t length);

		/// <summary>
		/// Set the value of the given net in bipolar encoding for the entire simulation time span with guaranteed best accuracy.
		/// </summary>
		void set_net_value_bipolar(uint32_t net, double value);

		/// <summary>
		/// Set the values of multiple nets quickly for the given number of bit times without accuracy guarantees. Not efficient for small numbers of nets.
		/// </summary>
		/// <param name="nets">Array of net indices to set</param>
		/// <param name="values_unipolar">Array of net values to be encoded in unipolar encoding</param>
		/// <param name="count">How many net indices are given</param>
		/// <param name="length">How many bits to generate per net</param>
		/// <param name="copy">Whether to copy (synchronize) host and device data structures, defaults to true. If false, operates only on the device-side circuit state.</param>
		/// <param name="device_values">Whether given value array is located on the host. If false (as by default), assumes a host array and copies it to the device first.</param>
		void set_net_values_curand(uint32_t* nets, const double* values_unipolar, uint32_t count, uint32_t length, bool copy = true, bool device_values = false);

		/// <summary>
		/// Set the values of multiple nets quickly for the entire simulation time span without accuracy guarantees. Not efficient for small numbers of nets.
		/// </summary>
		/// <param name="nets">Array of net indices to set</param>
		/// <param name="values_unipolar">Array of net values to be encoded in unipolar encoding</param>
		/// <param name="count">How many net indices are given</param>
		/// <param name="copy">Whether to copy (synchronize) host and device data structures, defaults to true. If false, operates only on the device-side circuit state.</param>
		/// <param name="device_values">Whether given value array is located on the host. If false (as by default), assumes a host array and copies it to the device first.</param>
		void set_net_values_curand(uint32_t* nets, const double* values_unipolar, uint32_t count, bool copy = true, bool device_values = false);

		/// <summary>
		/// Set the values of multiple consecutive nets quickly for the given number of bit times without accuracy guarantees. Not efficient for small numbers of nets.
		/// </summary>
		/// <param name="first_net">First net index</param>
		/// <param name="values_unipolar">Array of net values to be encoded in unipolar encoding</param>
		/// <param name="count">How many nets to set starting at first_net</param>
		/// <param name="length">How many bits to generate per net</param>
		/// <param name="copy">Whether to copy (synchronize) host and device data structures, defaults to true. If false, operates only on the device-side circuit state.</param>
		/// <param name="device_values">Whether given value array is located on the host. If false (as by default), assumes a host array and copies it to the device first.</param>
		void set_net_values_curand(uint32_t first_net, const double* values_unipolar, uint32_t count, uint32_t length, bool copy = true, bool device_values = false);

		/// <summary>
		/// Set the values of multiple consecutive nets quickly for the entire simulation time span without accuracy guarantees. Not efficient for small numbers of nets.
		/// </summary>
		/// <param name="first_net">First net index</param>
		/// <param name="values_unipolar">Array of net values to be encoded in unipolar encoding</param>
		/// <param name="count">How many nets to set starting at first_net</param>
		/// <param name="copy">Whether to copy (synchronize) host and device data structures, defaults to true. If false, operates only on the device-side circuit state.</param>
		/// <param name="device_values">Whether given value array is located on the host. If false (as by default), assumes a host array and copies it to the device first.</param>
		void set_net_values_curand(uint32_t first_net, const double* values_unipolar, uint32_t count, bool copy = true, bool device_values = false);

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

		/// <returns>SN representing the given circuit net. Dynamically updates with circuit state, copy to preserve the current value.</returns>
		StochasticNumber& get_net_value(uint32_t net);
		/// <returns>SN representing the given circuit net. Dynamically updates with circuit state, copy to preserve the current value.</returns>
		const StochasticNumber& get_net_value(uint32_t net) const;
		double get_net_value_unipolar(uint32_t net) const;
		double get_net_value_bipolar(uint32_t net) const;

		/// <summary>
		/// Get the values of multiple nets quickly for the given number of bit times, without respecting individual net progress. Not efficient for small numbers of nets.
		/// </summary>
		/// <param name="nets">Array of net indices to evaluate</param>
		/// <param name="values_unipolar">Array to receive the unipolar net values (host-side)</param>
		/// <param name="count">How many net indices are given</param>
		/// <param name="length">How many bits to evaluate per net</param>
		/// <param name="copy">Whether to copy (synchronize) host and device data structures, defaults to true. If false, operates only on the device-side circuit state.</param>
		void get_net_values_cuda(uint32_t* nets, double* values_unipolar, uint32_t count, uint32_t length, bool copy = true);

		/// <summary>
		/// Get the values of multiple nets quickly for the entire simulation time span, without respecting individual net progress. Not efficient for small numbers of nets.
		/// </summary>
		/// <param name="nets">Array of net indices to evaluate</param>
		/// <param name="values_unipolar">Array to receive the unipolar net values (host-side)</param>
		/// <param name="count">How many net indices are given</param>
		/// <param name="copy">Whether to copy (synchronize) host and device data structures, defaults to true. If false, operates only on the device-side circuit state.</param>
		void get_net_values_cuda(uint32_t* nets, double* values_unipolar, uint32_t count, bool copy = true);

		/// <summary>
		/// Get the values of multiple consecutive nets quickly for the given number of bit times, without respecting individual net progress. Not efficient for small numbers of nets.
		/// </summary>
		/// <param name="first_net">First net index</param>
		/// <param name="values_unipolar">Array to receive the unipolar net values (host-side)</param>
		/// <param name="count">How many nets to evaluate starting at first_net</param>
		/// <param name="length">How many bits to evaluate per net</param>
		/// <param name="copy">Whether to copy (synchronize) host and device data structures, defaults to true. If false, operates only on the device-side circuit state.</param>
		void get_net_values_cuda(uint32_t first_net, double* values_unipolar, uint32_t count, uint32_t length, bool copy = true);

		/// <summary>
		/// Get the values of multiple consecutive nets quickly for the entire simulation time span, without respecting individual net progress. Not efficient for small numbers of nets.
		/// </summary>
		/// <param name="first_net">First net index</param>
		/// <param name="values_unipolar">Array to receive the unipolar net values (host-side)</param>
		/// <param name="count">How many nets to evaluate starting at first_net</param>
		/// <param name="copy">Whether to copy (synchronize) host and device data structures, defaults to true. If false, operates only on the device-side circuit state.</param>
		void get_net_values_cuda(uint32_t first_net, double* values_unipolar, uint32_t count, bool copy = true);

		/// <returns>Pointer to the component object with the given index in the circuit.</returns>
		CircuitComponent* get_component(uint32_t index);

		/// <summary>
		/// Makes this circuit use the given scheduler for simulation scheduling.
		/// </summary>
		/// <param name="scheduler">Scheduler to use. Must not be compiled yet. To use dynamic scheduling (default), set this to nullptr.</param>
		void set_scheduler(Scheduler* scheduler);
		const Scheduler* get_scheduler();

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
		uint32_t* const component_io_host;
		uint32_t* const component_io_dev;
		size_t* const component_io_offsets_host;
		size_t* const component_io_offsets_dev;

		Scheduler* scheduler;

		//host-only circuit constructor
		StochasticCircuit(uint32_t sim_length, uint32_t num_nets, uint32_t* net_values, uint32_t* net_progress, uint32_t num_components_comb, uint32_t num_components_seq, uint32_t* components_index,
			CircuitComponent** components, uint32_t* component_progress, uint32_t num_component_types, uint32_t* component_io, size_t* component_io_offsets, StochasticNumber* net_numbers);

		//device-accelerated circuit constructor
		StochasticCircuit(uint32_t sim_length, uint32_t num_nets, uint32_t* net_values_host, uint32_t* net_values_dev, size_t net_values_dev_pitch, uint32_t* net_progress_host, uint32_t* net_progress_dev,
			uint32_t num_components_comb, uint32_t num_components_seq, uint32_t* components_host_index, CircuitComponent** components_host, CircuitComponent** components_dev, char* component_array_host,
			size_t component_array_host_pitch, char* component_array_dev, size_t component_array_dev_pitch, uint32_t* component_progress_host, uint32_t* component_progress_dev, uint32_t num_component_types,
			uint32_t* component_io_host, uint32_t* component_io_dev, size_t* component_io_offsets_host, size_t* component_io_offsets_dev, StochasticNumber* net_numbers);

		void copy_component_progress_from_device();

	};

}

#pragma once

#include "StochasticCircuitFactory.cuh"
#include "BasicCombinatorial.cuh"
#include "ParallelCounter.cuh"
#include "Btanh.cuh"

using namespace scsim;

/// <summary>
/// Fully connected layer
/// Weight ordering (tensorflow/keras ordering): Input 1 outputs 1...n, input 2 outputs 1...n, ..., input n outputs 1...n
/// </summary>
class FCLayer
{
public:
	const uint32_t input_size;
	const uint32_t output_size;
	const uint32_t num_weights;
	const uint32_t first_input;
	const double scale_factor;
	const uint32_t btanh_r;
	
	/// <param name="first_input">first input net index, must have input_size consecutive nets already available</param>
	/// <param name="scale_factor">factor used in weight scaling, greater than or equal to 1</param>
	FCLayer(StochasticCircuitFactory& f, uint32_t input_size, uint32_t first_input, uint32_t output_size, double scale_factor, uint32_t btanh_r) : input_size(input_size), output_size(output_size),
		num_weights(input_size * output_size), first_input(first_input), scale_factor(scale_factor), btanh_r(btanh_r) {

		if (scale_factor < 1.) throw std::runtime_error("FCLayer: Invalid scale factor");

		auto count_width = (uint32_t)floor(log2(input_size + 1)) + 1; //width of parallel counter output for each neuron

		_first_weight = f.add_nets(num_weights).first;
		_first_bias = f.add_nets(output_size).first;
		auto _first_intermediate = f.add_nets(num_weights).first;
		auto _first_countline = f.add_nets(output_size * count_width).first;
		_first_output = f.add_nets(output_size).first;

		//offset multipliers for weights, to create correct dimension ordering
		auto weight_offset_per_input = output_size;
		constexpr auto weight_offset_per_output = 1u;

		uint32_t* counter_inputs = (uint32_t*)malloc((input_size + 1) * sizeof(uint32_t));
		uint32_t* counter_outputs = (uint32_t*)malloc(count_width * sizeof(uint32_t));
		if (counter_inputs == nullptr || counter_outputs == nullptr) throw std::runtime_error("FCLayer: Out of memory on initialization");

		for (uint32_t i = 0; i < output_size; i++) { //neuron loop
			auto weight_offset_o = i * weight_offset_per_output;
			auto counter_in_offset = i * input_size;
			auto count_offset = i * count_width;

			for (uint32_t j = 0; j < input_size; j++) { //weight multipliers / counter inputs
				auto weight_offset_oi = weight_offset_o + j * weight_offset_per_input;

				counter_inputs[j] = _first_intermediate + counter_in_offset + j;
				factory_add_component(f, XnorGate, first_input + j, _first_weight + weight_offset_oi, counter_inputs[j]);
			}

			counter_inputs[input_size] = _first_bias + i; //bias counter input

			for (uint32_t j = 0; j < count_width; j++) counter_outputs[j] = _first_countline + count_offset + j; //counter outputs

			factory_add_component(f, ParallelCounter, input_size + 1, counter_inputs, counter_outputs); //parallel counter (summation)
			factory_add_component(f, Btanh, input_size + 1, btanh_r, counter_outputs, _first_output + i); //activation function
		}
	}

	uint32_t first_output() const { return _first_output; }
	uint32_t first_weight() const { return _first_weight; }
	uint32_t first_bias() const { return _first_bias; }

private:
	uint32_t _first_output;
	uint32_t _first_weight;
	uint32_t _first_bias;

};
#pragma once

#include "StochasticCircuitFactory.cuh"
#include "BasicCombinatorial.cuh"
#include "MaxApprox.cuh"

using namespace scsim;

/// <summary>
/// Max pooling layer
/// </summary>
class MaxPoolLayer
{
public:
	const uint32_t input_width;
	const uint32_t input_height;
	const uint32_t layers;
	const uint32_t pool_width;
	const uint32_t pool_height;
	const uint32_t output_width;
	const uint32_t output_height;
	const uint32_t first_input;

	/// <param name="first_input">first input net index, must have input_size consecutive nets already available</param>
	MaxPoolLayer(StochasticCircuitFactory& f, uint32_t input_width, uint32_t input_height, uint32_t layers, uint32_t first_input, uint32_t pool_width, uint32_t pool_height) :
		input_width(input_width), input_height(input_height), layers(layers), pool_width(pool_width), pool_height(pool_height), first_input(first_input), output_width(input_width / pool_width),
		output_height(input_height / pool_height) {

		if (input_width % pool_width != 0 || input_height % pool_height != 0) throw std::runtime_error("MaxPoolLayer: Input size must be multiple of pool size");

		auto pool_size = pool_width * pool_height;
		auto input_layer_size = input_width * input_height;
		auto output_layer_size = output_width * output_height;

		_first_output = f.add_nets(output_layer_size * layers).first;

		uint32_t* max_inputs = (uint32_t*)malloc(pool_size * sizeof(uint32_t));
		if (max_inputs == nullptr) throw std::runtime_error("MaxPoolLayer: Out of memory on initialization");

		for (uint32_t l = 0; l < layers; l++) { //layer loop
			auto input_offset_l = input_layer_size * l;
			auto output_offset_l = output_layer_size * l;

			for (uint32_t j = 0; j < output_height; j++) { //output loop: row
				auto input_offset_ly = input_offset_l + input_width * pool_height * j;
				auto output_offset_ly = output_offset_l + output_width * j;

				for (uint32_t i = 0; i < output_width; i++) { //output loop: column
					auto input_offset_lyx = input_offset_ly + pool_width * i;

					for (uint32_t t = 0; t < pool_height; t++) { //pool loop: row
						auto max_offset_y = pool_width * t;
						auto input_offset_lyxy = input_offset_lyx + input_width * t;

						for (uint32_t s = 0; s < pool_width; s++) { //pool loop: column
							max_inputs[max_offset_y + s] = first_input + input_offset_lyxy + s;
						}
					}

					factory_add_component(f, MaxApprox, pool_size, max_inputs, _first_output + output_offset_ly + i);
				}
			}
		}
	}

	uint32_t first_output() const { return _first_output; }

private:
	uint32_t _first_output;

};
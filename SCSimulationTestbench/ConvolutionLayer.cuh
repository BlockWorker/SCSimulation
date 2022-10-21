#pragma once

#include "StochasticCircuitFactory.cuh"
#include "BasicCombinatorial.cuh"
#include "ParallelCounter.cuh"
#include "Btanh.cuh"

using namespace scsim;

/// <summary>
/// Convolution layer
/// Weight ordering: Stencil 1 layer 1 row 1 column 1...k, stencil 1 layer 1 row 2 column 1...k, ..., stencil 1 layer m row l column 1...k, stencil 2 layer 1 row 1 column 1...k, ..., stencil n layer m row l column 1...k
/// </summary>
class ConvolutionLayer
{
public:
	const uint32_t total_width;
	const uint32_t total_height;
	const uint32_t input_depth;
	const uint32_t output_depth;
	const uint32_t stencil_width;
	const uint32_t stencil_height;
	const uint32_t num_weights;
	const uint32_t first_input;
	const uint32_t padding_input;
	const double scale_factor;

	ConvolutionLayer(StochasticCircuitFactory& f, uint32_t total_width, uint32_t total_height, uint32_t input_depth, uint32_t first_input, uint32_t output_depth, uint32_t stencil_width,
		uint32_t stencil_height, uint32_t padding_input, double scale_factor) : total_width(total_width), total_height(total_height), input_depth(input_depth), first_input(first_input),
		output_depth(output_depth), stencil_width(stencil_width), stencil_height(stencil_height), padding_input(padding_input), scale_factor(scale_factor),
		num_weights(stencil_width * stencil_height * input_depth * output_depth) {

		auto total_layer_size = total_width * total_height;
		auto stencil_layer_size = stencil_width * stencil_height;
		auto neuron_inputs = stencil_layer_size * input_depth; //number of inputs per neuron
		auto count_width = (uint32_t)floor(log2(neuron_inputs + 1)) + 1; //width of parallel counter output for each neuron
		auto btanh_r = Btanh::calculate_r(neuron_inputs + 1, scale_factor);

		auto stencil_x_offset = (stencil_width + 1) / 2 - 1;
		auto stencil_y_offset = (stencil_height + 1) / 2 - 1;
		

		_first_weight = f.add_nets(num_weights).first;
		_first_bias = f.add_nets(output_depth).first;
		auto _first_countline = f.add_nets(total_layer_size * output_depth * count_width).first;
		_first_output = f.add_nets(total_layer_size * output_depth).first;

		uint32_t* counter_inputs = (uint32_t*)malloc((neuron_inputs + 1) * sizeof(uint32_t));
		uint32_t* counter_outputs = (uint32_t*)malloc(count_width * sizeof(uint32_t));
		if (counter_inputs == nullptr || counter_outputs == nullptr) throw std::runtime_error("ConvolutionLayer: Out of memory on initialization");

		for (uint32_t l = 0; l < output_depth; l++) { //neuron loop: layer
			auto pixel_offset_l = total_layer_size * l;
			auto weight_offset_l = neuron_inputs * l;

			for (uint32_t j = 0; j < total_height; j++) { //neuron loop: row
				auto pixel_offset_ly = pixel_offset_l + total_width * j;

				for (uint32_t i = 0; i < total_width; i++) { //neuron loop: column
					auto pixel_offset_lyx = pixel_offset_ly + i;

					for (uint32_t t = 0; t < stencil_height; t++) { //stencil loop: row
						auto input_y = (int64_t)j + t - stencil_y_offset;
						uint32_t input_offset_y = total_width * input_y;
						auto weight_offset_ly = weight_offset_l + stencil_width * t;
						auto counter_in_offset_y = t * stencil_width * input_depth;

						for (uint32_t s = 0; s < stencil_width; s++) { //stencil loop: column
							auto input_x = (int64_t)i + s - stencil_x_offset;
							uint32_t input_offset_xy = input_offset_y + input_x;
							auto weight_offset_lxy = weight_offset_ly + s;
							auto counter_in_offset_xy = counter_in_offset_y + s * input_depth;

							if (input_y < 0 || input_y >= total_height || input_x < 0 || input_x >= total_width) {
								for (uint32_t u = 0; u < input_depth; u++) { //stencil loop: layer, out of bounds -> using zero padding for this pixel
									counter_inputs[counter_in_offset_xy + u] = padding_input;
								}
							} else {
								auto first_intermediate = f.add_nets(input_depth).first;

								for (uint32_t u = 0; u < input_depth; u++) { //stencil loop: layer, in bounds -> using data for this pixel
									factory_add_component(f, XnorGate, first_input + input_offset_xy + total_layer_size * u, _first_weight + weight_offset_lxy + stencil_layer_size * u, first_intermediate + u);
									counter_inputs[counter_in_offset_xy + u] = first_intermediate + u;
								}
							}
						}
					}

					counter_inputs[neuron_inputs] = _first_bias + l; //bias counter input

					auto countline_offset = pixel_offset_lyx * count_width;
					for (uint32_t k = 0; k < count_width; k++) counter_outputs[k] = _first_countline + countline_offset + k; //counter outputs

					factory_add_component(f, ParallelCounter, neuron_inputs + 1, counter_inputs, counter_outputs); //parallel counter (summation)
					factory_add_component(f, Btanh, neuron_inputs + 1, btanh_r, counter_outputs, _first_output + pixel_offset_lyx); //activation function
				}
			}
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
#pragma once

#include "StochasticCircuitFactory.cuh"
#include "BasicCombinatorial.cuh"
#include "ParallelCounter.cuh"
#include "Btanh.cuh"

#undef CONVOLUTION_TEST_OUTPUTS
//#define CONVOLUTION_TEST_OUTPUTS

using namespace scsim;

/// <summary>
/// Convolution layer
/// Weight ordering (tensorflow/keras 'channel_last'): Row 1 column 1 layer 1 stencil 1...k, row 1 column 1 layer 2 stencil 1...k, ..., row 1 column m layer l stencil 1...k, row 2 column 1 layer 1 stencil 1...k, ..., row n column m layer l stencil 1...k
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
	const uint32_t first_padding_input;
	const uint32_t padding_input_count;
	const double scale_factor;
	const uint32_t btanh_r;

#ifdef CONVOLUTION_TEST_OUTPUTS
	uint32_t test_neuron_indices[27 + 1 + 5 + 1] = { 0 };
#endif

	ConvolutionLayer(StochasticCircuitFactory& f, uint32_t total_width, uint32_t total_height, uint32_t input_depth, uint32_t first_input, uint32_t output_depth, uint32_t stencil_width,
		uint32_t stencil_height, uint32_t first_padding_input, uint32_t padding_input_count, double scale_factor, uint32_t btanh_r) : total_width(total_width), total_height(total_height),
		input_depth(input_depth), first_input(first_input),	output_depth(output_depth), stencil_width(stencil_width), stencil_height(stencil_height), first_padding_input(first_padding_input),
		padding_input_count(padding_input_count), scale_factor(scale_factor), num_weights(stencil_width * stencil_height * input_depth * output_depth), btanh_r(btanh_r) {

		auto total_layer_size = total_width * total_height;
		auto stencil_layer_size = stencil_width * stencil_height;
		auto neuron_inputs = stencil_layer_size * input_depth; //number of inputs per neuron, excluding bias
		auto count_width = (uint32_t)floor(log2(neuron_inputs + 1.)) + 1; //width of parallel counter output for each neuron

		padding_input_offset = 0;

		auto stencil_x_offset = (stencil_width + 1) / 2 - 1;
		auto stencil_y_offset = (stencil_height + 1) / 2 - 1;

		//offset multipliers for input/output pixels, to create correct dimension ordering
		auto input_offset_per_row = total_width * input_depth;
		auto output_offset_per_row = total_width * output_depth;
		auto input_offset_per_column = input_depth;
		auto output_offset_per_column = output_depth;
		constexpr auto pixel_offset_per_layer = 1u;
		//offset multipliers for weights, to create correct dimension ordering
		auto weight_offset_per_row = stencil_width * input_depth * output_depth;
		auto weight_offset_per_column = input_depth * output_depth;
		auto weight_offset_per_layer = output_depth;
		constexpr auto weight_offset_per_stencil = 1u;

		_first_weight = f.add_nets(num_weights).first;
		_first_bias = f.add_nets(output_depth).first;
		auto _first_countline = f.add_nets(total_layer_size * output_depth * count_width).first;
		_first_output = f.add_nets(total_layer_size * output_depth).first;

		uint32_t* counter_inputs = (uint32_t*)malloc((neuron_inputs + 1ull) * sizeof(uint32_t));
		uint32_t* counter_outputs = (uint32_t*)malloc(count_width * sizeof(uint32_t));
		if (counter_inputs == nullptr || counter_outputs == nullptr) throw std::runtime_error("ConvolutionLayer: Out of memory on initialization");


		for (uint32_t j = 0; j < total_height; j++) { //neuron loop: row
			auto output_offset_y = output_offset_per_row * j;

			for (uint32_t i = 0; i < total_width; i++) { //neuron loop: column
				auto output_offset_yx = output_offset_y + output_offset_per_column * i;

				for (uint32_t l = 0; l < output_depth; l++) { //neuron loop: layer
					auto output_offset_yxs = output_offset_yx + pixel_offset_per_layer * l;
					auto weight_offset_s = weight_offset_per_stencil * l;

					for (uint32_t t = 0; t < stencil_height; t++) { //stencil loop: row
						auto input_y = (int64_t)j + t - stencil_y_offset;
						uint32_t input_offset_y = input_offset_per_row * input_y;
						auto weight_offset_sy = weight_offset_s + weight_offset_per_row * t;
						auto counter_in_offset_y = t * stencil_width * input_depth;

						for (uint32_t s = 0; s < stencil_width; s++) { //stencil loop: column
							auto input_x = (int64_t)i + s - stencil_x_offset;
							uint32_t input_offset_xy = input_offset_y + input_offset_per_column * input_x;
							auto weight_offset_syx = weight_offset_sy + weight_offset_per_column * s;
							auto counter_in_offset_xy = counter_in_offset_y + s * input_depth;

							if (input_y < 0 || input_y >= total_height || input_x < 0 || input_x >= total_width) {
								for (uint32_t u = 0; u < input_depth; u++) { //stencil loop: layer, out of bounds -> using zero padding for this pixel, loop through all given padding inputs
									counter_inputs[counter_in_offset_xy + u] = first_padding_input + padding_input_offset++;
									padding_input_offset %= padding_input_count;
								}
							} else {
								auto first_intermediate = f.add_nets(input_depth).first;

								for (uint32_t u = 0; u < input_depth; u++) { //stencil loop: layer, in bounds -> using data for this pixel
									auto input_offset_xyl = input_offset_xy + pixel_offset_per_layer * u;
									auto weight_offset_sxyl = weight_offset_syx + weight_offset_per_layer * u;

									factory_add_component(f, XnorGate, first_input + input_offset_xyl, _first_weight + weight_offset_sxyl, first_intermediate + u);
									counter_inputs[counter_in_offset_xy + u] = first_intermediate + u;
								}
							}
						}
					}

					counter_inputs[neuron_inputs] = _first_bias + l; //bias counter input

					auto countline_offset = output_offset_yxs * count_width;
					for (uint32_t k = 0; k < count_width; k++) counter_outputs[k] = _first_countline + countline_offset + k; //counter outputs

					factory_add_component(f, ParallelCounter, neuron_inputs + 1, counter_inputs, counter_outputs); //parallel counter (summation)
					factory_add_component(f, Btanh, neuron_inputs + 1, btanh_r, counter_outputs, _first_output + output_offset_yxs); //activation function

#ifdef CONVOLUTION_TEST_OUTPUTS
					if (i == 1 && j == 1 && l == 0) {
						memcpy(test_neuron_indices, counter_inputs, (27 + 1) * sizeof(uint32_t));
						memcpy(test_neuron_indices + (27 + 1), counter_outputs, 5 * sizeof(uint32_t));
						test_neuron_indices[27 + 1 + 5] = _first_output + output_offset_yxs;
					}
#endif
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

	uint32_t padding_input_offset;

};
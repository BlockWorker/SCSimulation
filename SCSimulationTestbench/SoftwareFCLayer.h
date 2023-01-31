#pragma once

#include <stdint.h>
#include <memory>
#include <functional>

/// <summary>
/// Fully connected layer in software
/// Weight ordering (tensorflow/keras ordering): Input 1 outputs 1...n, input 2 outputs 1...n, ..., input n outputs 1...n
/// </summary>
class SoftwareFCLayer
{
public:
	const uint32_t input_size;
	const uint32_t output_size;
	const uint32_t num_weights;
	const std::function<void(double*, size_t)> activation_func;

	const double* weights;
	const double* biases;

	/// <param name="activation_func">Activation function, given pointer to values and number of values - returns in same given array. Defaults to no-op (identity activation function).</param>
	SoftwareFCLayer(uint32_t input_size, uint32_t output_size, const double* weights, const double* biases, std::function<void(double*, size_t)> activation_func = [](double* vals, size_t count) { }) :
		input_size(input_size), output_size(output_size), num_weights(input_size * output_size), activation_func(activation_func) {
		this->weights = weights;
		this->biases = biases;
	}

	void calculate(const double* inputs, double* outputs) {
		memset(outputs, 0, output_size * sizeof(double));

		//offset multipliers for weights, to create correct dimension ordering
		auto weight_offset_per_input = output_size;
		constexpr auto weight_offset_per_output = 1u;

		for (uint32_t i = 0; i < output_size; i++) {
			auto weight_offset_o = i * weight_offset_per_output;
			for (uint32_t j = 0; j < input_size; j++) { //sum weighted inputs
				auto weight_offset_oi = weight_offset_o + j * weight_offset_per_input;
				outputs[i] += inputs[j] * weights[weight_offset_oi];
			}
			outputs[i] += biases[i];
		}
		activation_func(outputs, output_size);
	}
	
};

#pragma once

#include <stdint.h>
#include <memory>

/// <summary>
/// Fully connected layer in software
/// Weight ordering: Output 1 inputs 1...n, output 2 inputs 1...n, ..., output n inputs 1...n
/// </summary>
class SoftwareFCLayer
{
public:
	const uint32_t input_size;
	const uint32_t output_size;
	const uint32_t num_weights;
	const double scale_factor;
	double* weights;

	SoftwareFCLayer(uint32_t input_size, uint32_t output_size, double scale_factor, double* weights) : input_size(input_size), output_size(output_size), num_weights(input_size * output_size),
		scale_factor(scale_factor) {
		this->weights = weights;
	}

	void calculate(double* inputs, double* outputs) {
		memset(outputs, 0, output_size * sizeof(double));

		for (uint32_t i = 0; i < output_size; i++) {
			auto weight_offset = i * input_size;
			for (uint32_t j = 0; j < input_size; j++) {
				outputs[i] += inputs[j] * weights[weight_offset + j];
			}
		}
	}
	
};

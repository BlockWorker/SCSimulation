#pragma once

#include <stdint.h>
#include "dll.h"

namespace scsim {

	class SCSIMAPI StochasticNumber
	{
	public:
		const uint32_t length;
		const size_t word_length;

		StochasticNumber(uint32_t length);
		StochasticNumber(uint32_t length, uint32_t* data, bool device_data = false);
		~StochasticNumber();

		static StochasticNumber* generate_unipolar(uint32_t length, double value);
		static StochasticNumber* generate_bipolar(uint32_t length, double value);
		static void generate_multiple_curand(StochasticNumber** numbers, uint32_t length, double* values_unipolar, size_t count);
		const uint32_t* get_data() const;
		double get_value_unipolar() const;
		double get_value_bipolar() const;
		void print_unipolar(uint32_t max_bits) const;
		void print_unipolar() const;
		void print_bipolar(uint32_t max_bits) const;
		void print_bipolar() const;

	private:
		uint32_t* data;
		void print_internal(double value, const char* ident, uint32_t max_bits) const;

	};

}


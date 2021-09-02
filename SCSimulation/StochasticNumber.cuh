#pragma once

#include <stdint.h>
#include "dll.h"

namespace scsim {

	class SCSIMAPI StochasticNumber
	{
	public:
		const uint32_t length;
		const size_t word_length;

		/// <summary>
		/// Create SN initialized to zero
		/// </summary>
		StochasticNumber(uint32_t length);

		/// <summary>
		/// Create SN initialized with given data
		/// </summary>
		/// <param name="device_data">Whether the data is located in device memory instead of host memory</param>
		StochasticNumber(uint32_t length, uint32_t* data, bool device_data = false);

		~StochasticNumber();
		StochasticNumber(const StochasticNumber& other);
		StochasticNumber& operator=(const StochasticNumber& other);

		/// <returns>SN with given value in unipolar encoding, guaranteed best possible accuracy for given length</returns>
		static StochasticNumber* generate_unipolar(uint32_t length, double value);

		/// <returns>SN with given value in bipolar encoding, guaranteed best possible accuracy for given length</returns>
		static StochasticNumber* generate_bipolar(uint32_t length, double value);

		/// <summary>
		/// Generate large number of SNs quickly, but without accuracy guarantees
		/// </summary>
		/// <param name="length">Length of each SN in bits</param>
		/// <param name="numbers">Array of number pointers to populate (only allocation necessary in advance)</param>
		/// <param name="values_unipolar">Array of values to be encoded assuming unipolar encoding</param>
		static void generate_multiple_curand(StochasticNumber** numbers, uint32_t length, double* values_unipolar, size_t count);

		const uint32_t* get_data() const;

		double get_value_unipolar() const;
		double get_value_bipolar() const;

		/// <param name="max_bits">Maximum number of bits to be printed to console</param>
		void print_unipolar(uint32_t max_bits) const;
		void print_unipolar() const;

		/// <param name="max_bits">Maximum number of bits to be printed to console</param>
		void print_bipolar(uint32_t max_bits) const;
		void print_bipolar() const;

		/// <returns>Correlation between this SN and other (SCC metric)</returns>
		double get_correlation(StochasticNumber* other);

		/// <returns>Autocorrelation of this SN w.r.t. the given offset (Box-Jenkins-Function)</returns>
		double get_autocorrelation(uint32_t offset);

		/// <returns>Correlation between this x and y (SCC metric)</returns>
		static double get_correlation(StochasticNumber* x, StochasticNumber* y);

	private:
		uint32_t* data;
		void print_internal(double value, const char* ident, uint32_t max_bits) const;

	};

}


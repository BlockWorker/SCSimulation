#pragma once

#include <stdint.h>
#include "dll.h"

namespace scsim {

	//maximum number of SN words to be generated in a single batch - limited to prevent out-of-memory errors
	constexpr size_t MAX_CURAND_BATCH_WORDS = 1 << 19;

	class StochasticCircuit;
	class StochasticCircuitFactory;

	class SCSIMAPI StochasticNumber
	{
	public:
		/// <summary>
		/// Create empty SN (length zero)
		/// </summary>
		StochasticNumber();

		/// <summary>
		/// Create SN initialized to all zeroes
		/// </summary>
		StochasticNumber(uint32_t length);

		/// <summary>
		/// Create SN initialized with given data
		/// </summary>
		/// <param name="device_data">Whether the data is located in device memory instead of host memory</param>
		StochasticNumber(uint32_t length, const uint32_t* data, bool device_data = false);

		~StochasticNumber();

		/// <summary>
		/// Creates a standalone copy of the given SN, with the same length and value bit-by-bit.
		/// </summary>
		StochasticNumber(const StochasticNumber& other);

		/// <summary>
		/// Copies the exact length and bit-by-bit value of the given SN to this SN, if possible.
		/// </summary>
		StochasticNumber& operator=(const StochasticNumber& other);

		/// <returns>SN with given value in unipolar encoding, guaranteed best possible accuracy for given length</returns>
		static StochasticNumber* generate_unipolar(uint32_t length, double value);

		/// <returns>SN with given value in bipolar encoding, guaranteed best possible accuracy for given length</returns>
		static StochasticNumber* generate_bipolar(uint32_t length, double value);

		/// <returns>SN with given constant bit stream value (all zeroes or all ones)</returns>
		static StochasticNumber* generate_constant(uint32_t length, bool value);

		/// <summary>
		/// Generate large number of SNs quickly, but without accuracy guarantees
		/// </summary>
		/// <param name="length">Length of each SN in bits</param>
		/// <param name="numbers">Array of number pointers to populate (only allocation necessary in advance)</param>
		/// <param name="values_unipolar">Array of values to be encoded assuming unipolar encoding</param>
		static void generate_multiple_curand(StochasticNumber** numbers, uint32_t length, const double* values_unipolar, size_t count);

		uint32_t length() const;
		uint32_t word_length() const;

		/// <summary>
		/// Sets this SN's length to the given value, if possible.
		/// If the SN is shortened, bits are removed from the end of the SN.
		/// If the SN is prolonged, uninitialized bits are added to the end of the SN.
		/// <b>Warning:</b> Both shortening and prolonging may change the SN's value, so if the value needs to be preserved, it is recommended to set the value again after calling this function.
		/// </summary>
		void set_length(uint32_t length);

		uint32_t* data();
		const uint32_t* data() const;

		double get_value_unipolar() const;
		double get_value_bipolar() const;

		/// <summary>Sets this SN to the given value in unipolar encoding, guaranteed best possible accuracy for the current SN length</summary>
		void set_value_unipolar(double value);

		/// <summary>Sets this SN to the given value in bipolar encoding, guaranteed best possible accuracy for the current length</summary>
		void set_value_bipolar(double value);

		/// <summary>Set all bits of this SN to all zeroes or all ones (chosen by "value")</summary>
		void set_value_constant(bool value);

		/// <param name="max_bits">Maximum number of bits to be printed to console</param>
		void print_unipolar(uint32_t max_bits) const;
		void print_unipolar() const;

		/// <param name="max_bits">Maximum number of bits to be printed to console</param>
		void print_bipolar(uint32_t max_bits) const;
		void print_bipolar() const;

		/// <returns>Correlation between this SN and other (SCC metric)</returns>
		double get_correlation(const StochasticNumber* other) const;

		/// <returns>Correlation between x and y (SCC metric)</returns>
		static double get_correlation(const StochasticNumber* x, const StochasticNumber* y);

		/// <returns>Autocorrelation of this SN w.r.t. the given offset (Box-Jenkins-Function)</returns>
		double get_autocorrelation(uint32_t offset) const;

	private:
		friend StochasticCircuit;
		friend StochasticCircuitFactory;

		const bool data_is_internal;

		uint32_t* const _length_ptr;
		uint32_t _internal_length;
		const uint32_t _external_max_length;
		uint32_t* _data;

		StochasticNumber(uint32_t* data_ptr, uint32_t* length_ptr, uint32_t max_length);

		void print_internal(double value, const char* ident, uint32_t max_bits) const;

		static void generate_bitstreams_curand(uint32_t** outputs, uint32_t length, const double* values_unipolar, size_t count);
		static void generate_bitstreams_curand(uint32_t* output, size_t output_pitch, uint32_t length, const double* values_unipolar, size_t count);

	};

}


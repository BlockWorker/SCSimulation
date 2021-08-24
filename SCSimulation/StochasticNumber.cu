#include "curand_base.cuh"

#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <random>
#include "Range.h"
#include <iostream>

#include "StochasticNumber.cuh"
#include "libpopcnt.h"

class gen_iterator : public std::iterator<std::output_iterator_tag, uint32_t>
{
public:

	class proxy
	{
	public:

		proxy(uint32_t* _data) : data(_data) { }

		void operator=(uint32_t val) {
			auto word = val / 32;
			auto bit = val % 32;
			data[word] |= 1 << (31 - bit);
		}

	private:

		uint32_t* const data;

	};

	explicit gen_iterator(uint32_t* _data) : data(_data) {}

	gen_iterator& operator++() { return *this; }
	gen_iterator operator++(int) { return *this; }

	proxy operator*() {
		return proxy(data);
	}

	gen_iterator& operator=(const gen_iterator&) { return *this; }

private:

	uint32_t* const data;

};

StochasticNumber::StochasticNumber(uint32_t length) : length(length), word_length((length + 31) / 32) {
	data = (uint32_t*)calloc(word_length, sizeof(uint32_t));
}

StochasticNumber::StochasticNumber(uint32_t length, uint32_t* data, bool device_data) : length(length), word_length((length + 31) / 32) {
	this->data = (uint32_t*)malloc(word_length * sizeof(uint32_t));
	if (this->data != nullptr) {
		if (device_data) cu(cudaMemcpy(this->data, data, word_length * sizeof(uint32_t), cudaMemcpyDeviceToHost));
		else memcpy(this->data, data, word_length * sizeof(uint32_t));
	}
}

StochasticNumber::~StochasticNumber() {
	free(data);
}

StochasticNumber* StochasticNumber::generate_unipolar(uint32_t length, double value) {
	if (value < 0. || value > 1.) throw;

	auto ret = new StochasticNumber(length);

	auto one_count = (uint32_t)round(value * length);

	auto range = Range(0, length - 1);
	std::sample(range.start(), range.end(), gen_iterator(ret->data), one_count, std::default_random_engine(std::random_device()()));

	return ret;
}

StochasticNumber* StochasticNumber::generate_bipolar(uint32_t length, double value) {
	return generate_unipolar(length, (value + 1.) / 2.);
}

__global__ void sngen_kern(double* rand, double* values, uint32_t* result, size_t result_pitch, uint32_t length, uint32_t word_length) {
	auto num = blockIdx.x;
	auto word_index = blockIdx.y * blockDim.x + threadIdx.x;

	if (word_index < word_length) {
		auto result_ptr = (uint32_t*)((char*)result + (num * result_pitch)) + word_index;
		auto rand_ptr = rand + (num * length) + (32 * word_index);
		auto val = values[num];

		uint32_t word = 0;
		for (uint32_t i = 0; i < 32; i++) {
			word <<= 1;
			word |= (rand_ptr[i] <= val) ? 1 : 0;
		}

		*result_ptr = word;
	}
}

void StochasticNumber::generate_multiple_curand(StochasticNumber** numbers, uint32_t length, double* values_unipolar, size_t count) {
	if (length % 32 != 0) throw;
	auto word_length = length / 32;

	double* rand_dev, *val_dev;
	uint32_t* sn_dev;
	size_t sn_dev_pitch;

	cu(cudaMalloc(&rand_dev, count * length * sizeof(double)));
	cu(cudaMalloc(&val_dev, count * sizeof(double)));
	cu(cudaMallocPitch(&sn_dev, &sn_dev_pitch, word_length * sizeof(uint32_t), count));

	cu(cudaMemcpy(val_dev, values_unipolar, count * sizeof(double), cudaMemcpyHostToDevice));

	curandGenerator_t gen;
	cur(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	cur(curandSetPseudoRandomGeneratorSeed(gen, std::random_device()()));
	cur(curandGenerateUniformDouble(gen, rand_dev, count * length));

	auto block_size = __min(word_length, 256);
	dim3 grid_size(count, (word_length + block_size - 1) / block_size);

	sngen_kern<<<grid_size, block_size>>>(rand_dev, val_dev, sn_dev, sn_dev_pitch, length, word_length);

	for (size_t i = 0; i < count; i++) {
		auto data_ptr = (uint32_t*)((char*)sn_dev + (i * sn_dev_pitch));
		numbers[i] = new StochasticNumber(length, data_ptr, true);
	}

	cu(cudaFree(rand_dev));
	cu(cudaFree(val_dev));
	cu(cudaFree(sn_dev));
}

const uint32_t* StochasticNumber::get_data() const {
	return data;
}

double StochasticNumber::get_value_unipolar() const {
	auto one_count = popcnt(data, word_length * sizeof(uint32_t));
	return (double)one_count / (double)length;
}

double StochasticNumber::get_value_bipolar() const {
	return get_value_unipolar() * 2. - 1.;
}

void StochasticNumber::print_unipolar(uint32_t max_bits) const {
	print_internal(get_value_unipolar(), "up", max_bits);
}

void StochasticNumber::print_unipolar() const {
	print_unipolar(length);
}

void StochasticNumber::print_bipolar(uint32_t max_bits) const {
	print_internal(get_value_bipolar(), "bp", max_bits);
}

void StochasticNumber::print_bipolar() const {
	print_bipolar(length);
}

void StochasticNumber::print_internal(double value, const char* ident, uint32_t max_bits) const {
	uint32_t bits = __min(max_bits, length);
	uint32_t words = (bits + 31) / 32;
	printf("% .6f", value);
	std::cout << " [" << ident << "] | ";
	for (uint32_t i = 0; i < words; i++) {
		auto word = data[i];
		for (uint32_t j = 0; j < 32; j++) {
			if (32 * i + j >= bits) break;
			std::cout << ((word & 0x80000000u) > 0 ? '1' : '0');
			word <<= 1;
		}
	}
	if (bits < length) std::cout << "... (" << length << ")";
	std::cout << std::endl;
}

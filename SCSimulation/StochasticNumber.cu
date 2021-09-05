#include "curand_base.cuh"

#include <algorithm>
#include <random>
#include "Range.h"
#include <iostream>

#include "StochasticNumber.cuh"
#include "libpopcnt.h"

#define takebitval(x) (((x) & 0x80000000u) > 0 ? 1 : 0)

namespace scsim {

	//output iterator used for exact SN generation
	//writing an integer to this iterator sets the bit at that position in the SN
	class gen_iterator
	{
	public:
		using iterator_category = std::output_iterator_tag;
		using value_type = uint32_t;
		using difference_type = ptrdiff_t;
		using pointer = uint32_t*;
		using reference = uint32_t&;

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

	StochasticNumber::StochasticNumber(const StochasticNumber& other) : StochasticNumber(other.length, other.data, false) {

	}

	StochasticNumber& StochasticNumber::operator=(const StochasticNumber& other) {
		free(data);
		return *this = StochasticNumber(other);
	}

	StochasticNumber* StochasticNumber::generate_unipolar(uint32_t length, double value) {
		if (value < 0. || value > 1.) throw;

		auto ret = new StochasticNumber(length);

		auto one_count = (uint32_t)round(value * length); //exact number of high bits required for best accuracy

		//set exact number of bits in random positions
		auto range = Range(0, length - 1);
		std::sample(range.start(), range.end(), gen_iterator(ret->data), one_count, std::default_random_engine(std::random_device()()));

		return ret;
	}

	StochasticNumber* StochasticNumber::generate_bipolar(uint32_t length, double value) {
		return generate_unipolar(length, (value + 1.) / 2.);
	}

	//generate one word of one result number per thread, by comparing random doubles to desired value (essentially parallelized simulated SNG)
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

	//maximum number of SN words to be generated in a single batch - limited to prevent out-of-memory errors
	constexpr uint32_t MAX_CURAND_BATCH_WORDS = 1 << 20;

	void StochasticNumber::generate_multiple_curand(StochasticNumber** numbers, uint32_t length, double* values_unipolar, size_t count) {
		auto word_length = (length + 31) / 32;

		auto gen_length = word_length * 32; //device code only produces entire words for efficiency -> this is the true number of bits generated per number

		double* rand_dev, * val_dev;
		uint32_t* sn_dev;
		size_t sn_dev_pitch;

		uint32_t max_batch = MAX_CURAND_BATCH_WORDS / word_length; //number of SNs in one batch

		curandGenerator_t gen;
		cur(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
		cur(curandSetPseudoRandomGeneratorSeed(gen, std::random_device()()));

		for (uint32_t batch_offset = 0; batch_offset < count; batch_offset += max_batch) {
			uint32_t batch_size = __min(count - batch_offset, max_batch);

			cu(cudaMalloc(&rand_dev, batch_size * gen_length * sizeof(double)));
			cu(cudaMalloc(&val_dev, batch_size * sizeof(double)));
			cu(cudaMallocPitch(&sn_dev, &sn_dev_pitch, word_length * sizeof(uint32_t), batch_size));

			cu(cudaMemcpy(val_dev, values_unipolar + batch_offset, batch_size * sizeof(double), cudaMemcpyHostToDevice));

			cur(curandGenerateUniformDouble(gen, rand_dev, batch_size * gen_length)); //generate one random double for each bit to be generated

			auto block_size = __min(word_length, 256);
			dim3 grid_size(batch_size, (word_length + block_size - 1) / block_size);

			sngen_kern<<<grid_size, block_size>>>(rand_dev, val_dev, sn_dev, sn_dev_pitch, gen_length, word_length); //generate actual SNs on device

			//create SN objects based on device data
			for (size_t i = 0; i < batch_size; i++) {
				auto data_ptr = (uint32_t*)((char*)sn_dev + (i * sn_dev_pitch));
				numbers[batch_offset + i] = new StochasticNumber(length, data_ptr, true);
			}

			cu(cudaFree(rand_dev));
			cu(cudaFree(val_dev));
			cu(cudaFree(sn_dev));
		}

		cur(curandDestroyGenerator(gen));
	}

	const uint32_t* StochasticNumber::get_data() const {
		return data;
	}

	double StochasticNumber::get_value_unipolar() const {
		auto one_count = popcnt(data, word_length * sizeof(uint32_t)); //fast population count to calculate value

		if (length % 32 != 0) { //if the number length is not a multiple of a word: subtract excess counted bits
			uint32_t last_word = data[word_length - 1];
			uint32_t excess_bits = 32 - length % 32;
			uint32_t excess_ones = 0;

			for (uint32_t i = 0; i < excess_bits; i++) {
				excess_ones += (last_word & 0x1);
				last_word >>= 1;
			}

			one_count -= excess_ones;
		}

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
				std::cout << takebitval(word);
				word <<= 1;
			}
		}
		if (bits < length) std::cout << "... (" << length << ")";
		std::cout << std::endl;
	}

	double StochasticNumber::get_correlation(StochasticNumber* other) {
		return get_correlation(this, other);
	}

	//see box-jenkins-function for more information
	double StochasticNumber::get_autocorrelation(uint32_t offset) {
		auto expected = get_value_unipolar();

		double denom_sum = 0;
		double num_sum = 0;

		uint32_t word1 = data[0];
		uint32_t word2 = data[offset / 32] << (offset % 32);
		for (uint32_t i = 0; i < length; i++) {
			uint32_t j = i + offset;

			double diff1 = takebitval(word1) - expected;
			word1 <<= 1;
			if (i % 32 == 31) {
				auto word1_index = (i + 1) / 32;
				if (word1_index < word_length) word1 = data[word1_index];
			}
			
			denom_sum += diff1 * diff1;

			if (j < length) {
				double diff2 = takebitval(word2) - expected;
				word2 <<= 1;
				if (j % 32 == 31) {
					auto word2_index = (j + 1) / 32;
					if (word2_index < word_length) word2 = data[word2_index];
				}

				num_sum += diff1 * diff2;
			}
		}

		if (denom_sum == 0.0) return 0.0;

		return num_sum / denom_sum;
	}

	//see SCC metric for more information
	double StochasticNumber::get_correlation(StochasticNumber* x, StochasticNumber* y) {
		if (x->length != y->length) throw;

		auto length = x->length;
		auto words = x->word_length;
		
		uint32_t* one_one = (uint32_t*)malloc(words * sizeof(uint32_t)); //overlapping ones
		uint32_t* one_zero = (uint32_t*)malloc(words * sizeof(uint32_t)); //ones in X where Y has zeroes
		uint32_t* zero_one = (uint32_t*)malloc(words * sizeof(uint32_t)); //zeroes in X where Y has ones
		uint32_t* zero_zero = (uint32_t*)malloc(words * sizeof(uint32_t)); //overlapping zeroes

		for (uint32_t i = 0; i < words; i++) {
			uint32_t mask; //calculate mask to avoid interference from bits beyond the limits of the SN
			if (i * 32 + 31 < length) mask = 0xffffffff;
			else {
				auto idlebits = (i + 1) * 32 - length;
				mask = ~((1 << idlebits) - 1);
			}

			auto xw = x->data[i];
			auto yw = y->data[i];

			auto diff = (xw ^ yw) & mask;

			one_one[i] = xw & yw & mask;
			one_zero[i] = diff & xw;
			zero_one[i] = diff & yw;
			zero_zero[i] = ~(xw | yw) & mask;
		}

		int32_t a = popcnt(one_one, words * sizeof(uint32_t));
		int32_t b = popcnt(one_zero, words * sizeof(uint32_t));
		int32_t c = popcnt(zero_one, words * sizeof(uint32_t));
		int32_t d = popcnt(zero_zero, words * sizeof(uint32_t));

		free(one_one);
		free(one_zero);
		free(zero_one);
		free(zero_zero);

		auto ad = (int64_t)a * d;
		auto bc = (int64_t)b * c;
		auto apb = a + b;
		auto apc = a + c;
		auto apbapc = (int64_t)apb * apc;

		double denom;

		if (ad > bc) {
			denom = (double)((int64_t)length * __min(apb, apc) - apbapc);
		} else {
			denom = (double)(apbapc - (int64_t)length * __max(a - d, 0));
		}

		if (denom == 0.0) return 0.0;

		return (double)(ad - bc) / denom;
	}

}

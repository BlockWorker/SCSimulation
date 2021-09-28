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

	StochasticNumber::StochasticNumber() : StochasticNumber(0) {

	}

	StochasticNumber::StochasticNumber(uint32_t length) : data_is_internal(true), _length_ptr(&_internal_length), _external_max_length(0) {
		_internal_length = length;
		if (length > 0) {
			_data = (uint32_t*)calloc(word_length(), sizeof(uint32_t));
			if (_data == nullptr) throw std::exception("StochasticNumber: Data allocation failed");
		} else {
			_data = nullptr;
		}
	}

	StochasticNumber::StochasticNumber(uint32_t length, const uint32_t* data, bool device_data) : data_is_internal(true), _length_ptr(&_internal_length), _external_max_length(0) {
		if (length == 0) throw std::exception("StochasticNumber: Data-based constructor may not be used to create empty SNs");
		_internal_length = length;
		auto my_word_length = word_length();
		_data = (uint32_t*)malloc(my_word_length * sizeof(uint32_t));
		if (_data == nullptr) throw std::exception("StochasticNumber: Data allocation failed");
		if (device_data) cu(cudaMemcpy(_data, data, my_word_length * sizeof(uint32_t), cudaMemcpyDeviceToHost));
		else memcpy(_data, data, my_word_length * sizeof(uint32_t));
	}

	StochasticNumber::StochasticNumber(uint32_t* data_ptr, uint32_t* length_ptr, uint32_t max_length) : data_is_internal(false), _length_ptr(length_ptr), _external_max_length(max_length) {
		_internal_length = 0;
		_data = data_ptr;
	}

	StochasticNumber::~StochasticNumber() {
		if (data_is_internal) free(_data);
	}

	StochasticNumber::StochasticNumber(const StochasticNumber& other) : StochasticNumber(other.length(), other.data(), false) {

	}

	StochasticNumber& StochasticNumber::operator=(const StochasticNumber& other) {
		set_length(other.length());
		memcpy(data(), other.data(), other.word_length() * sizeof(uint32_t));
		return *this;
	}

	StochasticNumber* StochasticNumber::generate_unipolar(uint32_t length, double value) {
		if (length == 0) throw std::exception("StochasticNumber generate: Length must be greater than zero.");
		if (value < 0. || value > 1.) throw std::exception("StochasticNumber generate: Given value is outside the encodable value range.");

		auto ret = new StochasticNumber(length);

		ret->set_value_unipolar(value);

		return ret;
	}

	StochasticNumber* StochasticNumber::generate_bipolar(uint32_t length, double value) {
		return generate_unipolar(length, (value + 1.) / 2.);
	}

	StochasticNumber* StochasticNumber::generate_constant(uint32_t length, bool value) {
		if (length == 0) throw std::exception("StochasticNumber generate: Length must be greater than zero.");

		auto ret = new StochasticNumber(length);

		if (value) ret->set_value_constant(true); //set to all ones if required, setting to all zeroes is not necessary as the SN is already initialized that way

		return ret;
	}

	void StochasticNumber::generate_multiple_curand(StochasticNumber** numbers, uint32_t length, const double* values_unipolar, size_t count) {
		if (length == 0) throw std::exception("StochasticNumber generate: Length must be greater than zero.");
		if (count == 0) throw std::exception("StochasticNumber generate: Count must be greater than zero.");

		auto word_length = (length + 31) / 32;

		uint32_t* sn_dev = nullptr;
		size_t sn_dev_pitch = 0;

		uint32_t max_batch = MAX_CURAND_BATCH_WORDS / word_length; //number of SNs in one batch

		for (uint32_t batch_offset = 0; batch_offset < count; batch_offset += max_batch) {
			uint32_t batch_size = __min(count - batch_offset, max_batch);

			try {
				cu(cudaMallocPitch(&sn_dev, &sn_dev_pitch, word_length * sizeof(uint32_t), batch_size));

				generate_bitstreams_curand(sn_dev, sn_dev_pitch, length, values_unipolar + batch_offset, batch_size);

				//create SN objects based on device data
				for (size_t i = 0; i < batch_size; i++) {
					auto data_ptr = (uint32_t*)((char*)sn_dev + (i * sn_dev_pitch));
					numbers[batch_offset + i] = new StochasticNumber(length, data_ptr, true);
				}
			} catch (CudaError& error) {
				cu_ignore_error(cudaFree(sn_dev));

				throw error;
			}

			cu_ignore_error(cudaFree(sn_dev));
		}
	}

	uint32_t StochasticNumber::length() const {
		return *_length_ptr;
	}

	uint32_t StochasticNumber::word_length() const {
		return (*_length_ptr + 31) / 32;
	}

	void StochasticNumber::set_length(uint32_t length) {
		if (length == this->length()) return;
		if (data_is_internal) { //internal data: change data array for new size
			if (length == 0) { //make SN empty for length 0
				free(_data);
				_data = nullptr;
				_internal_length = 0;
			} else { //resize data for length > 0
				uint32_t new_word_length = (length + 31) / 32;
				if (new_word_length != word_length()) {
					auto new_dataptr = (uint32_t*)realloc(_data, new_word_length * sizeof(uint32_t));
					if (new_dataptr == nullptr) throw std::exception("set_length: Data reallocation failed");
					_data = new_dataptr;
					_internal_length = length;
				}
			}
		} else { //external data: simply adjust length value unless maximum length is exceeded
			if (length > _external_max_length) throw std::exception("set_length: New length exceeds maximum length");
			*_length_ptr = length;
		}
	}

	uint32_t* StochasticNumber::data() {
		return _data;
	}

	const uint32_t* StochasticNumber::data() const {
		return _data;
	}

	double StochasticNumber::get_value_unipolar() const {
		auto my_length = length();

		if (my_length == 0) throw std::exception("get_value_unipolar: Value of an empty SN is undefined.");

		auto my_word_length = word_length();
		auto my_data = data();

		auto one_count = popcnt(my_data, my_word_length * sizeof(uint32_t)); //fast population count to calculate value

		if (my_length % 32 != 0) { //if the number length is not a multiple of a word: subtract excess counted bits
			uint32_t excess_mask = (1u << (32u - my_length % 32u)) - 1u;

			uint32_t excess_data = my_data[my_word_length - 1] & excess_mask;

			one_count -= popcnt(&excess_data, sizeof(uint32_t));
		}

		return (double)one_count / (double)my_length;
	}

	double StochasticNumber::get_value_bipolar() const {
		return get_value_unipolar() * 2. - 1.;
	}

	void StochasticNumber::set_value_unipolar(double value) {
		auto my_length = length();
		auto my_data = data();

		if (my_length == 0) throw std::exception("set_value_unipolar: SN length must be greater than zero.");
		if (value < 0. || value > 1.) throw std::exception("set_value_unipolar: Given value is outside the encodable value range.");

		memset(my_data, 0, word_length() * sizeof(uint32_t));

		auto one_count = (uint32_t)round(value * my_length); //exact number of high bits required for best accuracy

		//set exact number of bits in random positions
		auto range = Range(0, my_length - 1);
		std::sample(range.start(), range.end(), gen_iterator(my_data), one_count, std::default_random_engine(std::random_device()()));
	}

	void StochasticNumber::set_value_bipolar(double value) {
		set_value_unipolar((value + 1.) / 2.);
	}

	void StochasticNumber::set_value_constant(bool value) {
		auto my_length = length();
		auto my_data = data();

		if (my_length == 0) throw std::exception("set_value_constant: SN length must be greater than zero.");

		auto word_length = my_length / 32;
		if (word_length > 0) memset(my_data, value ? 0xff : 0x00, word_length * sizeof(uint32_t));

		auto extra_length = my_length % 32;
		if (extra_length > 0) {
			if (value) my_data[word_length] |= (0xffffffff << (32 - extra_length));
			else my_data[word_length] &= 0xffffffff >> extra_length;
		}
	}

	void StochasticNumber::print_unipolar(uint32_t max_bits) const {
		if (length() == 0) {
			std::cout << "Empty stochastic number" << std::endl;
			return;
		}
		print_internal(get_value_unipolar(), "up", max_bits);
	}

	void StochasticNumber::print_unipolar() const {
		print_unipolar(length());
	}

	void StochasticNumber::print_bipolar(uint32_t max_bits) const {
		if (length() == 0) {
			std::cout << "Empty stochastic number" << std::endl;
			return;
		}
		print_internal(get_value_bipolar(), "bp", max_bits);
	}

	void StochasticNumber::print_bipolar() const {
		print_bipolar(length());
	}

	void StochasticNumber::print_internal(double value, const char* ident, uint32_t max_bits) const {
		auto my_length = length();
		uint32_t bits = __min(max_bits, my_length);
		uint32_t words = (bits + 31) / 32;
		auto my_data = data();
		printf("% .6f", value);
		std::cout << " [" << ident << "] | ";
		for (uint32_t i = 0; i < words; i++) {
			auto word = my_data[i];
			for (uint32_t j = 0; j < 32; j++) {
				if (32 * i + j >= bits) break;
				std::cout << takebitval(word);
				word <<= 1;
			}
		}
		if (bits < my_length) std::cout << "... (" << my_length << ")";
		std::cout << std::endl;
	}

	double StochasticNumber::get_correlation(const StochasticNumber* other) const {
		return get_correlation(this, other);
	}

	//see box-jenkins-function for more information
	double StochasticNumber::get_autocorrelation(uint32_t offset) const {
		auto my_length = length();

		if (my_length == 0) throw std::exception("get_autocorrelation: Autocorrelation of an empty SN is undefined.");

		auto my_word_length = word_length();
		auto my_data = data();

		auto expected = get_value_unipolar();

		double denom_sum = 0;
		double num_sum = 0;

		uint32_t word1 = my_data[0];
		uint32_t word2 = my_data[offset / 32] << (offset % 32);
		for (uint32_t i = 0; i < my_length; i++) {
			uint32_t j = i + offset;

			double diff1 = takebitval(word1) - expected;
			word1 <<= 1;
			if (i % 32 == 31) {
				auto word1_index = (i + 1) / 32;
				if (word1_index < my_word_length) word1 = my_data[word1_index];
			}
			
			denom_sum += diff1 * diff1;

			if (j < my_length) {
				double diff2 = takebitval(word2) - expected;
				word2 <<= 1;
				if (j % 32 == 31) {
					auto word2_index = (j + 1) / 32;
					if (word2_index < my_word_length) word2 = my_data[word2_index];
				}

				num_sum += diff1 * diff2;
			}
		}

		if (denom_sum == 0.0) return 0.0;

		return num_sum / denom_sum;
	}

	//see SCC metric for more information
	double StochasticNumber::get_correlation(const StochasticNumber* x, const StochasticNumber* y) {
		if (x->length() != y->length()) throw std::exception("get_correlation: Both numbers must have the same length.");

		auto length = x->length();
		
		if (length == 0) throw std::exception("get_correlation: Correlation of empty SNs is undefined.");

		auto words = x->word_length();

		auto x_data = x->data();
		auto y_data = y->data();
		
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

			auto xw = x_data[i];
			auto yw = y_data[i];

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

	//generate one word of one result number per thread, by comparing random doubles to desired value (essentially parallelized simulated SNG)
	__global__ void sngen_kern(double* rand, double* values, uint32_t** results, uint32_t length, uint32_t word_length) {
		auto num = blockIdx.x;
		auto word_index = blockIdx.y * blockDim.x + threadIdx.x;

		if (word_index < word_length) {
			auto result_ptr = results[num] + word_index;
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

	void StochasticNumber::generate_bitstreams_curand(uint32_t** outputs, uint32_t length, const double* values_unipolar, size_t count) {
		if (length == 0) throw std::exception("generate_bitstreams_curand: Length must be greater than zero.");
		if (count == 0) throw std::exception("generate_bitstreams_curand: Count must be greater than zero.");

		auto word_length = (length + 31) / 32;

		if (count * word_length > MAX_CURAND_BATCH_WORDS) throw std::exception("generate_bitstreams_curand: Amount of data to be generated exceeds MAX_CURAND_BATCH_WORDS.");

		auto gen_length = word_length * 32; //device code only produces entire words for efficiency -> this is the true number of bits generated per number

		double* rand_dev = nullptr;
		double* val_dev = nullptr;
		uint32_t** outputs_dev = nullptr;

		curandGenerator_t gen;
		cur(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A));
		cur(curandSetPseudoRandomGeneratorSeed(gen, std::random_device()()));

		try {
			cu(cudaMalloc(&rand_dev, count * gen_length * sizeof(double)));
			cu(cudaMalloc(&val_dev, count * sizeof(double)));
			cu(cudaMalloc(&outputs_dev, count * sizeof(uint32_t*)));

			cu(cudaMemcpy(val_dev, values_unipolar, count * sizeof(double), cudaMemcpyHostToDevice));
			cu(cudaMemcpy(outputs_dev, outputs, count * sizeof(uint32_t*), cudaMemcpyHostToDevice));

			cur(curandGenerateUniformDouble(gen, rand_dev, count * gen_length)); //generate one random double for each bit to be generated

			auto block_size = __min(word_length, 256);
			dim3 grid_size(count, (word_length + block_size - 1) / block_size);

			sngen_kern<<<grid_size, block_size>>>(rand_dev, val_dev, outputs_dev, gen_length, word_length); //generate actual SNs on device
			cu_kernel_errcheck();
		} catch (CudaError& error) {
			cu_ignore_error(cudaFree(rand_dev));
			cu_ignore_error(cudaFree(val_dev));
			cu_ignore_error(cudaFree(outputs_dev));

			throw error;
		}

		cu_ignore_error(cudaFree(rand_dev));
		cu_ignore_error(cudaFree(val_dev));
		cu_ignore_error(cudaFree(outputs_dev));

		cur(curandDestroyGenerator(gen));
	}

	void StochasticNumber::generate_bitstreams_curand(uint32_t* output, size_t output_pitch, uint32_t length, const double* values_unipolar, size_t count) {
		if (length == 0) throw std::exception("generate_bitstreams_curand: Length must be greater than zero.");
		if (count == 0) throw std::exception("generate_bitstreams_curand: Count must be greater than zero.");

		auto word_length = (length + 31) / 32;

		if (count * word_length > MAX_CURAND_BATCH_WORDS) throw std::exception("generate_bitstreams_curand: Amount of data to be generated exceeds MAX_CURAND_BATCH_WORDS.");

		auto outputs = (uint32_t**)malloc(count * sizeof(uint32_t*));
		for (size_t i = 0; i < count; i++) {
			outputs[i] = (uint32_t*)((char*)output + (i * output_pitch));
		}

		generate_bitstreams_curand(outputs, length, values_unipolar, count);

		free(outputs);
	}

}

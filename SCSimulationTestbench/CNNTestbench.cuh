#pragma once

#include "Testbench.cuh"
#include "StochasticNumber.cuh"

#include "BinaryDoubleReader.h"
#include "NumericCSVReader.h"

#include "ConvolutionLayer.cuh"
#include "FCLayer.cuh"
#include "MaxPoolLayer.cuh"
#include "SoftwareFCLayer.h"


static constexpr uint32_t image_width = 32;
static constexpr uint32_t image_height = 32;
static constexpr uint32_t image_channels = 3;
static constexpr uint32_t image_size = image_width * image_height * image_channels;
static constexpr uint32_t total_image_count = 10000;

static constexpr uint32_t zero_pad_count = 1024;

static constexpr char* weight_path = "weights.csv";
static constexpr char* image_path = "images.dat";


static void softmax(double* vals, size_t count) {
	double denom = 0.;
	for (size_t i = 0; i < count; i++) denom += exp(vals[i]);
	for (size_t i = 0; i < count; i++) vals[i] = exp(vals[i]) / denom;
}


class CNNTestbench : public Testbench
{
public:
	CNNTestbench(uint32_t min_sim_length, uint32_t max_iter_runs, uint32_t image_count) : Testbench(1, max_iter_runs, true), min_sim_length(min_sim_length),
		max_iter_runs(max_iter_runs), max_sim_length(min_sim_length << (max_iter_runs - 1)), max_sim_length_words((max_sim_length + 31) / 32), image_count(image_count) {
		cu(cudaMalloc(&image_vals_device, image_count * image_size * sizeof(double)));
		first_in = 0;
		first_out = 0;
		param_reader = new NumericCSVReader(weight_path);
		software_layer = new SoftwareFCLayer(64, 10, param_reader->get_row(10), param_reader->get_row(11), &softmax);

		BinaryDoubleReader image_reader(image_path, total_image_count + image_count * image_size);

		image_labels = new uint32_t[image_count];
		for (uint32_t i = 0; i < image_count; i++) {
			image_labels[i] = (uint32_t)image_reader.get_value(i);
		}

		cu(cudaMemcpy(image_vals_device, image_reader.get_data() + total_image_count, image_count * image_size * sizeof(double), cudaMemcpyHostToDevice));
	}

	virtual ~CNNTestbench() {
		cu_ignore_error(cudaFree(image_vals_device));
		cu_ignore_error(cudaFree(parameter_storage_device[0]));
		cu_ignore_error(cudaFree(zero_pad_storage_dev));

		free(parameter_storage_host[0]);
		free(zero_pad_storage_host);

		delete[] image_labels;
		delete param_reader;
		delete software_layer;
	}

protected:
	const uint32_t min_sim_length;
	const uint32_t max_sim_length;
	const uint32_t max_sim_length_words;
	const uint32_t max_iter_runs;
	const uint32_t image_count;

	uint32_t first_zero_pad;
	uint32_t first_in;
	uint32_t first_out;

	uint32_t* zero_pad_storage_host;
	uint32_t* zero_pad_storage_dev;

#ifdef CONVOLUTION_TEST_OUTPUTS
	uint32_t first_outs_intermediate[7] = { 0 };
	uint32_t out_sizes_intermediate[7] = { 0 };
	uint32_t test_neuron_indices[27 + 1 + 5 + 1] = { 0 };
#endif

	double* image_vals_device;
	uint32_t param_start_indices[10] = { 0 };
	size_t param_counts[10] = { 0 };
	uint32_t* parameter_storage_host[10] = { nullptr }; //storage spaces for weights + biases of 5 parametrized layers
	uint32_t* parameter_storage_device[10] = { nullptr };
	bool params_initialized = false;

	NumericCSVReader* param_reader;
	uint32_t* image_labels;

	SoftwareFCLayer* software_layer;

	uint32_t host_correct = 0, dev_correct = 0;
	std::vector<double> accs;


	virtual uint32_t build_circuit(uint32_t setup) override {
		factory.set_sim_length(max_sim_length);

		//build neural network structure
		first_zero_pad = factory.add_nets(zero_pad_count).first;

		first_in = factory.add_nets(image_size).first;

		ConvolutionLayer conv1(factory, image_width, image_height, image_channels, first_in, 20, 3, 3, first_zero_pad, zero_pad_count,
			param_reader->get_value(0, 0), (uint32_t)param_reader->get_value(0, 1));
		MaxPoolLayer pool1(factory, conv1.total_width, conv1.total_height, conv1.output_depth, conv1.first_output(), 2, 2);
		ConvolutionLayer conv2(factory, pool1.output_width, pool1.output_height, pool1.layers, pool1.first_output(), 40, 3, 3, first_zero_pad, zero_pad_count,
			param_reader->get_value(2, 0), (uint32_t)param_reader->get_value(2, 1));
		MaxPoolLayer pool2(factory, conv2.total_width, conv2.total_height, conv2.output_depth, conv2.first_output(), 2, 2);
		ConvolutionLayer conv3(factory, pool2.output_width, pool2.output_height, pool2.layers, pool2.first_output(), 60, 3, 3, first_zero_pad, zero_pad_count,
			param_reader->get_value(4, 0), (uint32_t)param_reader->get_value(4, 1));
		ConvolutionLayer conv4(factory, conv3.total_width, conv3.total_height, conv3.output_depth, conv3.first_output(), 60, 3, 3, first_zero_pad, zero_pad_count,
			param_reader->get_value(6, 0), (uint32_t)param_reader->get_value(6, 1));
		MaxPoolLayer pool3(factory, conv4.total_width, conv4.total_height, conv4.output_depth, conv4.first_output(), 2, 2);
		FCLayer fc1(factory, pool3.output_width * pool3.output_height * pool3.layers, pool3.first_output(), 64, param_reader->get_value(8, 0), (uint32_t)param_reader->get_value(8, 1));

		first_out = fc1.first_output();

#ifdef CONVOLUTION_TEST_OUTPUTS
		first_outs_intermediate[0] = conv1.first_output();
		first_outs_intermediate[1] = pool1.first_output();
		first_outs_intermediate[2] = conv2.first_output();
		first_outs_intermediate[3] = pool2.first_output();
		first_outs_intermediate[4] = conv3.first_output();
		first_outs_intermediate[5] = conv4.first_output();
		first_outs_intermediate[6] = pool3.first_output();

		out_sizes_intermediate[0] = conv1.total_width * conv1.total_height * conv1.output_depth;
		out_sizes_intermediate[1] = pool1.output_width * pool1.output_height * pool1.layers;
		out_sizes_intermediate[2] = conv2.total_width * conv2.total_height * conv2.output_depth;
		out_sizes_intermediate[3] = pool2.output_width * pool2.output_height * pool2.layers;
		out_sizes_intermediate[4] = conv3.total_width * conv3.total_height * conv3.output_depth;
		out_sizes_intermediate[5] = conv4.total_width * conv4.total_height * conv4.output_depth;
		out_sizes_intermediate[6] = pool3.output_width * pool3.output_height * pool3.layers;

		memcpy(test_neuron_indices, conv1.test_neuron_indices, (27 + 1 + 5 + 1) * sizeof(uint32_t));
#endif

		//prepare parameter storage
		param_start_indices[0] = conv1.first_weight();
		param_start_indices[1] = conv1.first_bias();
		param_start_indices[2] = conv2.first_weight();
		param_start_indices[3] = conv2.first_bias();
		param_start_indices[4] = conv3.first_weight();
		param_start_indices[5] = conv3.first_bias();
		param_start_indices[6] = conv4.first_weight();
		param_start_indices[7] = conv4.first_bias();
		param_start_indices[8] = fc1.first_weight();
		param_start_indices[9] = fc1.first_bias();

		param_counts[0] = conv1.num_weights;
		param_counts[1] = conv1.output_depth;
		param_counts[2] = conv2.num_weights;
		param_counts[3] = conv2.output_depth;
		param_counts[4] = conv3.num_weights;
		param_counts[5] = conv3.output_depth;
		param_counts[6] = conv4.num_weights;
		param_counts[7] = conv4.output_depth;
		param_counts[8] = fc1.num_weights;
		param_counts[9] = fc1.output_size;

		size_t total_param_count = 0;
		for (uint32_t i = 0; i < 10; i++) total_param_count += param_counts[i];

		parameter_storage_host[0] = (uint32_t*)malloc(total_param_count * max_sim_length_words * sizeof(uint32_t));
		zero_pad_storage_host = (uint32_t*)malloc(zero_pad_count * max_sim_length_words * sizeof(uint32_t));
		if (parameter_storage_host[0] == nullptr || zero_pad_storage_host == nullptr) throw std::runtime_error("build_circuit: host allocation failed");
		cu(cudaMalloc(parameter_storage_device, total_param_count * max_sim_length_words * sizeof(uint32_t)));
		cu(cudaMalloc(&zero_pad_storage_dev, zero_pad_count * max_sim_length_words * sizeof(uint32_t)));
		
		for (uint32_t i = 1; i < 10; i++) {
			parameter_storage_host[i] = parameter_storage_host[i - 1] + param_counts[i - 1] * max_sim_length_words;
			parameter_storage_device[i] = parameter_storage_device[i - 1] + param_counts[i - 1] * max_sim_length_words;
		}

		return max_iter_runs;
	}

	void save_parameters() {
		for (uint32_t i = 0; i < 10; i++) {
			cu(cudaMemcpy2D(parameter_storage_host[i], circuit->sim_length_words * sizeof(uint32_t), (char*)circuit->net_values_host + (circuit->net_values_host_pitch * param_start_indices[i]),
				circuit->net_values_host_pitch, circuit->sim_length_words * sizeof(uint32_t), param_counts[i], cudaMemcpyHostToHost));
			cu(cudaMemcpy2D(parameter_storage_device[i], circuit->sim_length_words * sizeof(uint32_t), (char*)circuit->net_values_dev + (circuit->net_values_dev_pitch * param_start_indices[i]),
				circuit->net_values_dev_pitch, circuit->sim_length_words * sizeof(uint32_t), param_counts[i], cudaMemcpyDeviceToDevice));
		}

		cu(cudaMemcpy2D(zero_pad_storage_host, circuit->sim_length_words * sizeof(uint32_t), (char*)circuit->net_values_host + (circuit->net_values_host_pitch * first_zero_pad),
			circuit->net_values_host_pitch, circuit->sim_length_words * sizeof(uint32_t), zero_pad_count, cudaMemcpyHostToHost));
		cu(cudaMemcpy2D(zero_pad_storage_dev, circuit->sim_length_words * sizeof(uint32_t), (char*)circuit->net_values_dev + (circuit->net_values_dev_pitch * first_zero_pad),
			circuit->net_values_dev_pitch, circuit->sim_length_words * sizeof(uint32_t), zero_pad_count, cudaMemcpyDeviceToDevice));
	}

	void load_parameters(bool device) {
		//restore parameters
		for (uint32_t i = 0; i < 10; i++) {
			for (uint32_t j = 0; j < param_counts[i]; j++) circuit->net_progress_host[param_start_indices[i] + j] = circuit->sim_length;

			if (device) {
				cu(cudaMemcpy2D((char*)circuit->net_values_dev + (circuit->net_values_dev_pitch * param_start_indices[i]), circuit->net_values_dev_pitch, parameter_storage_device[i],
					circuit->sim_length_words * sizeof(uint32_t), circuit->sim_length_words * sizeof(uint32_t), param_counts[i], cudaMemcpyDeviceToDevice));
			} else {
				cu(cudaMemcpy2D((char*)circuit->net_values_host + (circuit->net_values_host_pitch * param_start_indices[i]), circuit->net_values_host_pitch, parameter_storage_host[i],
					circuit->sim_length_words * sizeof(uint32_t), circuit->sim_length_words * sizeof(uint32_t), param_counts[i], cudaMemcpyHostToHost));
			}
		}

		//restore zero padding
		for (uint32_t j = 0; j < zero_pad_count; j++) circuit->net_progress_host[first_zero_pad + j] = circuit->sim_length;
		if (device) {
			cu(cudaMemcpy2D((char*)circuit->net_values_dev + (circuit->net_values_dev_pitch * first_zero_pad), circuit->net_values_dev_pitch, zero_pad_storage_dev,
				circuit->sim_length_words * sizeof(uint32_t), circuit->sim_length_words * sizeof(uint32_t), zero_pad_count, cudaMemcpyDeviceToDevice));
		} else {
			cu(cudaMemcpy2D((char*)circuit->net_values_host + (circuit->net_values_host_pitch * first_zero_pad), circuit->net_values_host_pitch, zero_pad_storage_host,
				circuit->sim_length_words * sizeof(uint32_t), circuit->sim_length_words * sizeof(uint32_t), zero_pad_count, cudaMemcpyHostToHost));
		}

		//copy net progress to device if needed
		if (device) cu(cudaMemcpy(circuit->net_progress_dev, circuit->net_progress_host, circuit->num_nets * sizeof(uint32_t), cudaMemcpyHostToDevice));
	}

	virtual void config_circuit(uint32_t setup, uint32_t scheduler, uint32_t iteration, uint32_t input, bool device) override {
		uint32_t iter_sim_length = min_sim_length << iteration;
		uint32_t iter_sim_length_words = (iter_sim_length + 31) / 32;

		auto param_start = std::chrono::steady_clock::now();
		if (!params_initialized) { //first run overall: generate parameter and zero padding SNs
			//generate parameters
			for (uint32_t i = 0; i < 10; i++) {
				circuit->set_net_values_curand(param_start_indices[i], param_reader->get_row(i) + 2, param_counts[i], false);
			}

			//generate zero padding
			double zero_pad_values[zero_pad_count];
			for (uint32_t i = 0; i < zero_pad_count; i++) zero_pad_values[i] = .5; //unipolar .5 = bipolar 0
			circuit->set_net_values_curand(first_zero_pad, zero_pad_values, zero_pad_count, false);

			//copy net values
			cu(cudaMemcpy2D(circuit->net_values_host, circuit->net_values_host_pitch, circuit->net_values_dev, circuit->net_values_dev_pitch, circuit->sim_length_words * sizeof(uint32_t),
				circuit->num_nets, cudaMemcpyDeviceToHost));

			//save parameters and zero padding, and mark as initialized for future runs
			save_parameters();
			params_initialized = true;
		} else { //later runs: copy already generated parameter SNs, on corresponding side
			load_parameters(device);
		}
		auto param_end = std::chrono::steady_clock::now();
		std::cout << "Parameter gen/load time: " << std::chrono::duration_cast<std::chrono::microseconds>(param_end - param_start).count() / 1000.0 << "ms" << std::endl;

		//generate image SNs
		circuit->set_net_values_curand(first_in, image_vals_device + (image_size * input), image_size, iter_sim_length, false, true);

		//copy image SNs back to host for host-side sim
		if (!device) cu(cudaMemcpy2D((char*)circuit->net_values_host + (circuit->net_values_host_pitch * first_in), circuit->net_values_host_pitch,
			(char*)circuit->net_values_dev + (circuit->net_values_dev_pitch * first_in), circuit->net_values_dev_pitch,
			iter_sim_length_words * sizeof(uint32_t), image_size, cudaMemcpyDeviceToHost));
	}

	virtual uint32_t get_iter_length(uint32_t setup, uint32_t scheduler, uint32_t iteration) override {
		return min_sim_length << iteration;
	}

	virtual uint32_t get_iter_inputs(uint32_t setup, uint32_t scheduler, uint32_t iteration, bool device) override {
		if (scheduler > 0 && device) return image_count;
		else return __max(image_count / 10, 1);
	}

	virtual void write_additional_column_titles(std::stringstream& ss) override {
		for (uint32_t i = 0; i < num_iterations; i++) {
			ss << CSV_SEPARATOR << i << "AccH" << CSV_SEPARATOR << i << "AccD";
		}
		ss << CSV_SEPARATOR << min_sim_length;
	}

	virtual void post_input(uint32_t setup, uint32_t scheduler, uint32_t iteration, uint32_t input, bool device) override {
		double sc_out[64] = { 0. };
		double final_out[10] = { 0. };

		circuit->get_net_values_cuda(first_out, sc_out, 64, get_iter_length(setup, scheduler, iteration), !device); //get outputs of SC network
		for (uint32_t i = 0; i < 64; i++) sc_out[i] = 2. * sc_out[i] - 1.; //convert to bipolar values, as they are supposed to be
		software_layer->calculate(sc_out, final_out); //calculate result of final dense layer

#ifdef CONVOLUTION_TEST_OUTPUTS
		std::stringstream ss;
		for (uint32_t j = 0; j < image_size; j++) {
			ss << circuit->get_net_value_bipolar(first_in + j) << ',';
		}
		ss << '\n';
		for (uint32_t i = 0; i < 7; i++) {
			for (uint32_t j = 0; j < out_sizes_intermediate[i]; j++) {
				ss << circuit->get_net_value_bipolar(first_outs_intermediate[i] + j) << ',';
			}
			ss << '\n';
		}
		for (uint32_t j = 0; j < 64; j++) {
			ss << sc_out[j] << ',';
		}
		ss << '\n';
		for (uint32_t j = 0; j < 200; j++) { //intermediate products within first layer, before counter
			ss << circuit->get_net_value_bipolar(first_outs_intermediate[0] + out_sizes_intermediate[0] + j) << ',';
		}
		ss << '\n';
		auto first_cl = param_start_indices[1] + param_counts[1];
		for (uint32_t i = 0; i < 40; i += 5) {
			double sum = 0.;
			for (uint32_t j = 0; j < 5; j++) {
				sum += circuit->get_net_value_unipolar(first_cl + i + j) * (double)(1 << j);
			}
			ss << (sum * 2. - 28.) << ',';
		}
		std::ofstream file;
		file.open("sim_intermediates.csv");
		file << ss.str();
		file.close();

		std::stringstream ss2;
		for (uint32_t i = 0; i < (27 + 1 + 5 + 1); i++) {
			if (i == 27 || i == 27 + 1 || i == 27 + 1 + 5) ss2 << std::endl;

			auto data = (uint32_t*)((char*)circuit->net_values_host + (circuit->net_values_host_pitch * test_neuron_indices[i]));

			for (uint32_t i = 0; i < __min(circuit->sim_length_words, 8); i++) {
				auto word = data[i];
				for (uint32_t j = 0; j < 32; j++) {
					ss2 << ((word & 0x80000000u) > 0 ? 1 : 0) << ',';
					word <<= 1;
				}
			}
			ss2 << std::endl;
		}
		std::ofstream file2;
		file2.open("neuron_bitstreams.csv");
		file2 << ss2.str();
		file2.close();
#endif

		uint32_t prediction = 0;
		double prediction_confidence = 0.;
		for (uint32_t i = 0; i < 10; i++) {
			if (final_out[i] > prediction_confidence) {
				prediction_confidence = final_out[i];
				prediction = i;
			}
		}

		if (prediction == image_labels[input]) {
			if (device) dev_correct++;
			else host_correct++;
		}

		std::cout << "Predicting class " << prediction << " with confidence " << prediction_confidence << ", which is " << (prediction == image_labels[input] ? "RIGHT" : "WRONG") << std::endl;
	}

	virtual void post_iter(uint32_t setup, uint32_t scheduler, uint32_t iteration) override {
		accs.push_back((double)host_correct / (double)get_iter_inputs(setup, scheduler, iteration, false));
		accs.push_back((double)dev_correct / (double)get_iter_inputs(setup, scheduler, iteration, true));
		host_correct = 0;
		dev_correct = 0;
	}

	virtual void post_setup(uint32_t setup, uint32_t scheduler, std::stringstream& ss) override {
		for (auto acc : accs) ss << CSV_SEPARATOR << acc;
		accs.clear();
	}

};

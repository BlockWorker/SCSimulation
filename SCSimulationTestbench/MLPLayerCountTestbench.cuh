#pragma once

#include <random>

#include "Testbench.cuh"
#include "StochasticNumber.cuh"
#include "BasicCombinatorial.cuh"
#include "Stanh.cuh"

class MLPLayerCountTestbench : public Testbench
{
public:
	/// <param name="layersize">Number of neurons per layer</param>
	/// <param name="setups">Number of setups, setup i contains i+1 layers</param>
	MLPLayerCountTestbench(uint32_t min_sim_length, uint32_t layersize, uint32_t setups, uint32_t iter_runs) : Testbench(setups, iter_runs), min_sim_length(min_sim_length), layersize(layersize),
		iter_runs(iter_runs), selectcount((uint32_t)ceil(log2(layersize))), max_sim_length(min_sim_length << (iter_runs - 1)) {
		numbers = nullptr;
		vals = nullptr;
		auto logn = log2(layersize);
		states = 2 * (uint32_t)round(logn + (log2(max_sim_length) * layersize) / (33.27 * logn));
		num_count = 0;
		first_in = 0;
		random_num_count = 0;
		select_num_count = 0;
	}

	virtual ~MLPLayerCountTestbench() {
		if (numbers != nullptr) for (uint32_t i = 0; i < num_count; i++) delete numbers[i];
		free(numbers);
		free(vals);
	}

protected:
	const uint32_t min_sim_length;
	const uint32_t max_sim_length;
	const uint32_t iter_runs;
	const uint32_t layersize;

	uint32_t first_in;
	uint32_t states;
	const uint32_t selectcount;
	uint32_t select_num_count;
	uint32_t random_num_count;
	uint32_t num_count;

	StochasticNumber** numbers;
	double* vals;

	virtual uint32_t build_circuit(uint32_t setup) override {
		select_num_count = (setup + 1) * selectcount;
		random_num_count = (setup + 1) * layersize * layersize + layersize;
		num_count = random_num_count + select_num_count;

		factory.set_sim_length(max_sim_length);

		first_in = factory.add_nets(random_num_count).first;
		auto first_weight = first_in + layersize;
		auto first_sel = factory.add_nets(select_num_count).first;
		auto first_int = factory.add_nets((setup + 1) * layersize).first;

		build_layer(first_in, first_weight, first_sel, first_int);
		for (uint32_t i = 1; i <= setup; i++) {
			build_layer(first_int + (i - 1) * layersize, first_weight + i * layersize * layersize, first_sel + i * selectcount, first_int + i * layersize);
		}

		free(vals);
		vals = (double*)malloc(num_count * sizeof(double));
		std::default_random_engine eng{ std::random_device()() };
		std::uniform_real_distribution<double> dist(0.0, 1.0);
		for (uint32_t i = 0; i < random_num_count; i++) {
			vals[i] = dist(eng);
		}
		for (uint32_t i = 0; i < select_num_count; i++) {
			vals[random_num_count + i] = 0.5;
		}

		return num_iterations;
	}

	virtual uint32_t config_circuit(uint32_t setup, uint32_t iteration, bool device) override {
		uint32_t iter_sim_length = min_sim_length;
		for (uint32_t i = 0; i < iteration; i++) {
			iter_sim_length *= 2;
		}

		if (!device) {
			numbers = (StochasticNumber**)calloc(num_count, sizeof(StochasticNumber*));
			StochasticNumber::generate_multiple_curand(numbers, iter_sim_length, vals, num_count);
		}

		for (uint32_t i = 0; i < num_count; i++) {
			circuit->set_net_value(first_in + i, *numbers[i]);
		}

		if (device) {
			for (uint32_t i = 0; i < num_count; i++) delete numbers[i];
			free(numbers);
			numbers = nullptr;
		}

		return iter_sim_length;
	}

	virtual void write_additional_column_titles(std::stringstream& ss) override {
		ss << CSV_SEPARATOR << min_sim_length;
	}

private:
	void build_layer(uint32_t first_layer_in, uint32_t first_layer_weight, uint32_t first_layer_sel, uint32_t first_layer_out) {
		for (uint32_t i = 0; i < layersize; i++) {
			auto added_nets = factory.add_nets(layersize + 1);
			auto first_mux_in = added_nets.first;
			auto mux_stanh_int = added_nets.second;

			auto first_neuron_weight = first_layer_weight + i * layersize;

			for (uint32_t j = 0; j < layersize; j++) {
				factory_add_component(factory, XnorGate, first_layer_in + j, first_neuron_weight + j, first_mux_in + j);
			}

			factory_add_component(factory, MultiplexerN, layersize, first_mux_in, first_layer_sel, mux_stanh_int);
			factory_add_component(factory, Stanh, mux_stanh_int, first_layer_out + i, states);
		}
	}

};

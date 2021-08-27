#pragma once

#include "Testbench.cuh"
#include "StochasticNumber.cuh"
#include "BasicCombinatorial.cuh"

class MuxNTestbench : public Testbench
{
public:
	MuxNTestbench(uint32_t sim_length, uint32_t setups) : Testbench(setups, 2), sim_length(sim_length) {
		numbers = nullptr;
		vals = nullptr;
	}

	virtual ~MuxNTestbench() {
		if (numbers != nullptr) for (uint32_t i = 0; i <= count; i++) delete numbers[i];
		free(numbers);
		free(vals);
	}

protected:
	const uint32_t sim_length;

	uint32_t first_in;
	uint32_t first_sel;
	uint32_t first_out;
	uint32_t count;

	StochasticNumber** numbers;
	double* vals;

	virtual void build_circuit(uint32_t setup) override {
		factory->set_sim_length(sim_length);

		count = 16;
		for (uint32_t i = 0; i < setup; i++) count *= 2;

		first_in = factory->add_nets(8 * (count + 1)).first;
		first_sel = factory->add_nets(3).first;
		first_out = factory->add_nets(count + 1).first;

		for (uint32_t i = 0; i <= count; i++) {
			auto in = 8 * i;
			factory->add_component(new MultiplexerN({ in, in + 1, in + 2, in + 3, in + 4, in + 5, in + 6, in + 7 }, { first_sel, first_sel + 1, first_sel + 2 }, first_out + i));
		}
	}

	virtual bool config_circuit(uint32_t iteration) override {
		if (iteration == 0) {
			numbers = (StochasticNumber**)malloc((count + 1) * sizeof(StochasticNumber*));
			vals = (double*)malloc((count + 1) * sizeof(double));
			for (uint32_t i = 0; i <= count; i++) {
				vals[i] = (double)i / (double)count;
			}

			StochasticNumber::generate_multiple_curand(numbers, sim_length, vals, count + 1);
		}

		for (uint32_t i = 0; i < 8 * (count + 1); i++) {
			circuit->set_net_value(first_in + i, numbers[i % (count + 1)]);
		}
		for (uint32_t i = 0; i < 3; i++) {
			circuit->set_net_value_unipolar(first_sel + i, 0.5);
		}

		if (iteration == 1) {
			for (uint32_t i = 0; i <= count; i++) delete numbers[i];
			free(numbers);
			free(vals);
			numbers = nullptr;
			vals = nullptr;
		}

		return iteration == 0;
	}

};

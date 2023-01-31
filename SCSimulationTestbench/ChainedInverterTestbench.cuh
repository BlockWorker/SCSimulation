#pragma once

#include "Testbench.cuh"
#include "StochasticNumber.cuh"
#include "BasicCombinatorial.cuh"

class ChainedInverterTestbench : public Testbench
{
public:
	/// <param name="setups">Number of setups, setup i contains 2^(i+4) inverters</param>
	ChainedInverterTestbench(uint32_t min_sim_length, uint32_t setups, uint32_t max_iter_runs) : Testbench(2 * setups, max_iter_runs), min_sim_length(min_sim_length), max_iter_runs(max_iter_runs), setups(setups) {
		count = 0;
		in = 0;
		out = 0;
		curr_max_sim_length = min_sim_length;
		number = nullptr;
	}

protected:
	const uint32_t min_sim_length;
	const uint32_t max_iter_runs;
	const uint32_t setups;

	uint32_t curr_max_sim_length;

	uint32_t in;
	uint32_t out;
	uint32_t count;

	StochasticNumber* number;

	virtual uint32_t build_circuit(uint32_t setup) override {
		auto setup_id = setup / 2;
		auto setup_kind = setup % 2;
		auto num_runs = __min(setups - setup_id, max_iter_runs);
		if (setup_id >= 2 && num_runs >= 2 && num_runs < max_iter_runs) num_runs = 2; //reduce simulation runs - only max length for small setups, then restrict to 2 runs each

		curr_max_sim_length = min_sim_length << (num_runs - 1);
		count = 16 << setup_id;

		factory.set_sim_length(curr_max_sim_length);

		auto netids = factory.add_nets(count + 1);
		in = netids.first;
		out = netids.second;

		if (setup_kind == 0) {
			for (uint32_t i = 0; i < count; i++) {
				factory_add_component(factory, Inverter, in + i, in + i + 1);
			}
		} else {
			for (uint32_t i = 0; i < count; i++) {
				factory_add_component(factory, Inverter, out - i - 1, out - i);
			}
		}

		return num_runs;
	}

	virtual void config_circuit(uint32_t setup, uint32_t iteration, uint32_t input, bool device) override {
		uint32_t iter_sim_length = min_sim_length << iteration;

		if (!device) number = StochasticNumber::generate_unipolar(iter_sim_length, 0.25);

		circuit->set_net_value(in, *number);

		if (device) delete number;
	}

	virtual uint32_t get_iter_length(uint32_t setup, uint32_t iteration) {
		return min_sim_length << iteration;
	}

	virtual void write_additional_column_titles(std::stringstream& ss) override {
		ss << CSV_SEPARATOR << "Order" << CSV_SEPARATOR << min_sim_length;
	}

	virtual void post_setup(uint32_t setup, std::stringstream& ss) override {
		ss << CSV_SEPARATOR << ((setup % 2 == 0) ? "Forward" : "Backward");
	}

};

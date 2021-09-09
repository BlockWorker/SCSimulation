#pragma once

#include "circuit_component_defines.cuh"
#include "BasicCombinatorial.cuh"

using namespace scsim;

bool sim_bitwise = false;

class BitwiseInverter : public Inverter
{
public:
	BitwiseInverter(uint32_t input, uint32_t output) : Inverter(input, output) {

	}

	virtual void simulate_step_host() override {
		if (!sim_bitwise) {
			Inverter::simulate_step_host();
			return;
		}

		auto out_offset = output_offsets_host[0];
		auto in_offset = input_offsets_host[0];
		auto current_progress = current_sim_progress();
		auto current_progress_word = current_sim_progress_word();
		auto next_step_progress = next_sim_progress();
		auto next_step_progress_word = next_sim_progress_word();

		uint32_t out_word = circuit->net_values_host[out_offset + current_progress_word] >> (32 - (current_progress % 32)); //present output, shifted for seamless continuation

		for (uint32_t i = current_progress_word; i < next_step_progress_word; i++) {
			auto in_word = circuit->net_values_host[in_offset + i];
			for (uint32_t j = 0; j < 32; j++) {
				auto bit = 32 * i + j;
				if (bit < current_progress) { //ignore bits that have already been processed
					in_word <<= 1;
					continue;
				}
				if (bit >= next_step_progress) { //last word ends before being full: shift for correct bit position
					out_word <<= (32 - j);
					break;
				}

				out_word <<= 1;
				putbit(out_word, !takebit(in_word)); //process bit

				in_word <<= 1;
			}
			circuit->net_values_host[out_offset + i] = out_word;
			out_word = 0;
		}
	}

};

class BitwiseAndGate : public AndGate
{
public:
	BitwiseAndGate(uint32_t input1, uint32_t input2, uint32_t output) : AndGate(input1, input2, output) {

	}

	virtual void simulate_step_host() override {
		if (!sim_bitwise) {
			AndGate::simulate_step_host();
			return;
		}

		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];
		auto current_progress = current_sim_progress();
		auto current_progress_word = current_sim_progress_word();
		auto next_step_progress = next_sim_progress();
		auto next_step_progress_word = next_sim_progress_word();

		uint32_t out_word = circuit->net_values_host[out_offset + current_progress_word] >> (32 - (current_progress % 32)); //present output, shifted for seamless continuation

		for (uint32_t i = current_progress_word; i < next_step_progress_word; i++) {
			auto in_word_1 = circuit->net_values_host[in_offset_1 + i];
			auto in_word_2 = circuit->net_values_host[in_offset_2 + i];
			for (uint32_t j = 0; j < 32; j++) {
				auto bit = 32 * i + j;
				if (bit < current_progress) { //ignore bits that have already been processed
					in_word_1 <<= 1;
					in_word_2 <<= 1;
					continue;
				}
				if (bit >= next_step_progress) { //last word ends before being full: shift for correct bit position
					out_word <<= (32 - j);
					break;
				}

				out_word <<= 1;
				putbit(out_word, takebit(in_word_1) && takebit(in_word_2)); //process bit
				
				in_word_1 <<= 1;
				in_word_2 <<= 1;
			}
			circuit->net_values_host[out_offset + i] = out_word;
			out_word = 0;
		}
	}

};

class BitwiseDelay : public Delay
{
public:
	BitwiseDelay(uint32_t input, uint32_t output) : Delay(input, output) {

	}

	virtual void simulate_step_host() override {
		if (!sim_bitwise) {
			Delay::simulate_step_host();
			return;
		}

		auto out_offset = output_offsets_host[0];
		auto in_offset = input_offsets_host[0];
		auto current_progress = current_sim_progress();
		auto current_progress_word = current_sim_progress_word();
		auto next_step_progress = next_sim_progress();
		auto next_step_progress_word = next_sim_progress_word();

		uint32_t out_word = circuit->net_values_host[out_offset + current_progress_word] >> (32 - (current_progress % 32)); //present output, shifted for seamless continuation

		bool last = false;
		if (current_progress > 0) {
			auto word = circuit->net_values_host[in_offset + (current_progress - 1) / 32];
			word <<= (current_progress - 1) % 32;
			last = takebit(word);
		}

		for (uint32_t i = current_progress_word; i < next_step_progress_word; i++) {
			auto in_word = circuit->net_values_host[in_offset + i];
			for (uint32_t j = 0; j < 32; j++) {
				auto bit = 32 * i + j;
				if (bit < current_progress) { //ignore bits that have already been processed
					in_word <<= 1;
					continue;
				}
				if (bit >= next_step_progress) { //last word ends before being full: shift for correct bit position
					out_word <<= (32 - j);
					break;
				}

				out_word <<= 1;
				putbit(out_word, last); //process bit

				last = takebit(in_word);
				in_word <<= 1;
			}
			circuit->net_values_host[out_offset + i] = out_word;
			out_word = 0;
		}
	}

};



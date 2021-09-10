#include "cuda_base.cuh"

#include <vector>
#include <algorithm>

#include "StochasticCircuit.cuh"
#include "CircuitComponent.cuh"
#include "StochasticNumber.cuh"

namespace scsim {

	StochasticCircuit::StochasticCircuit(uint32_t sim_length, uint32_t num_nets, uint32_t* net_values, uint32_t* net_progress, uint32_t num_components_comb, uint32_t num_components_seq,
		CircuitComponent** components, uint32_t* component_progress, uint32_t num_component_types) :
		host_only(true), sim_length(sim_length), sim_length_words((sim_length + 31) / 32), num_nets(num_nets), net_values_host(net_values), net_values_host_pitch((sim_length + 31) / 32 * sizeof(uint32_t)),
		net_values_dev(nullptr), net_values_dev_pitch(0), net_progress_host(net_progress), net_progress_dev(nullptr), num_components_comb(num_components_comb), num_components_seq(num_components_seq),
		num_components(num_components_comb + num_components_seq), components_host(components), components_dev(nullptr), component_array_host(nullptr), component_array_host_pitch(0),
		component_array_dev(nullptr), component_array_dev_pitch(0), component_progress_host(component_progress), component_progress_dev(nullptr), num_component_types(num_component_types) {

	}

	StochasticCircuit::StochasticCircuit(uint32_t sim_length, uint32_t num_nets, uint32_t* net_values_host, uint32_t* net_values_dev, size_t net_values_dev_pitch, uint32_t* net_progress_host, uint32_t* net_progress_dev,
		uint32_t num_components_comb, uint32_t num_components_seq, CircuitComponent** components_host, CircuitComponent** components_dev, char* component_array_host, size_t component_array_host_pitch,
		char* component_array_dev, size_t component_array_dev_pitch, uint32_t* component_progress_host, uint32_t* component_progress_dev, uint32_t num_component_types) :
		host_only(false), sim_length(sim_length), sim_length_words((sim_length + 31) / 32), num_nets(num_nets), net_values_host(net_values_host),
		net_values_host_pitch((sim_length + 31) / 32 * sizeof(uint32_t)), net_values_dev(net_values_dev), net_values_dev_pitch(net_values_dev_pitch), net_progress_host(net_progress_host),
		net_progress_dev(net_progress_dev), num_components_comb(num_components_comb), num_components_seq(num_components_seq), num_components(num_components_comb + num_components_seq),
		components_host(components_host), components_dev(components_dev), component_array_host(component_array_host), component_array_host_pitch(component_array_host_pitch),
		component_array_dev(component_array_dev), component_array_dev_pitch(component_array_dev_pitch), component_progress_host(component_progress_host),
		component_progress_dev(component_progress_dev), num_component_types(num_component_types) {

	}

	StochasticCircuit::~StochasticCircuit() {
		if (host_only) {
			for (uint32_t i = 0; i < num_components; i++) {
				delete components_host[i]; //host-only: components are in originally allocated positions, delete
			}
		}
		else {
			for (uint32_t i = 0; i < num_components; i++) {
				components_host[i]->~CircuitComponent(); //device-accelerated: components are in component array, only deconstruct
			}
		}

		if (!host_only) {
			cu_ignore_error(cudaFree(net_values_dev));
			cu_ignore_error(cudaFree(net_progress_dev));
			cu_ignore_error(cudaFree(components_dev));
			cu_ignore_error(cudaFree(component_array_dev));
			cu_ignore_error(cudaFree(component_progress_dev));

			cu_ignore_error(cudaHostUnregister(component_array_host));
			cu_ignore_error(cudaHostUnregister(net_values_host));
			cu_ignore_error(cudaHostUnregister(net_progress_host));
			cu_ignore_error(cudaHostUnregister(component_progress_host));

			free(component_array_host);
		}

		free(net_values_host);
		free(net_progress_host);
		free(components_host);
		free(component_progress_host);
	}

	void StochasticCircuit::reset_circuit() {
		simulation_finished = false;
		memset(net_progress_host, 0, num_nets * sizeof(uint32_t)); //clear progress on all nets

		for (uint32_t i = 0; i < num_components_seq; i++) {
			components_host[num_components_comb + i]->reset_state(); //reset sequential component states
		}
	}

	void StochasticCircuit::set_net_value(uint32_t net, StochasticNumber* value) {
		memcpy((net_values_host + (sim_length_words * net)), value->get_data(), __min(sim_length_words, value->word_length) * sizeof(uint32_t)); //copy data from SN
		net_progress_host[net] = __min(sim_length, value->length);
	}

	void StochasticCircuit::set_net_value_unipolar(uint32_t net, double value, uint32_t length) {
		auto tempnum = StochasticNumber::generate_unipolar(length, value);
		set_net_value(net, tempnum);
		delete tempnum;
	}

	void StochasticCircuit::set_net_value_unipolar(uint32_t net, double value) {
		set_net_value_unipolar(net, value, sim_length);
	}

	void StochasticCircuit::set_net_value_bipolar(uint32_t net, double value, uint32_t length) {
		auto tempnum = StochasticNumber::generate_bipolar(length, value);
		set_net_value(net, tempnum);
		delete tempnum;
	}

	void StochasticCircuit::set_net_value_bipolar(uint32_t net, double value) {
		set_net_value_bipolar(net, value, sim_length);
	}

	void StochasticCircuit::set_net_value_constant(uint32_t net, bool value, uint32_t length) {
		auto actual_length = __min(length, sim_length);
		auto net_value = net_values_host + (sim_length_words * net);

		auto word_length = actual_length / 32;
		if (word_length > 0) memset(net_value, value ? 0xff : 0x00, word_length * sizeof(uint32_t));

		auto extra_length = actual_length % 32;
		if (extra_length > 0) {
			if (value) net_value[word_length] |= (0xffffffff << (32 - extra_length));
			else net_value[word_length] &= 0xffffffff >> extra_length;
		}
	}

	void StochasticCircuit::set_net_value_constant(uint32_t net, bool value) {
		set_net_value_constant(net, value, sim_length);
	}

	void StochasticCircuit::copy_data_to_device() {
		copy_data_to_device(sim_length);
	}

	void StochasticCircuit::copy_data_to_device(uint32_t net_length) {
		if (host_only) return;

		size_t width = (__min(net_length, sim_length) + 31) / 32 * sizeof(uint32_t);

		cu(cudaMemcpy2D(net_values_dev, net_values_dev_pitch, net_values_host, net_values_host_pitch, width, num_nets, cudaMemcpyHostToDevice)); //copy net values
		cu(cudaMemcpy(net_progress_dev, net_progress_host, num_nets * sizeof(uint32_t), cudaMemcpyHostToDevice)); //copy net progress
		cu(cudaMemcpy2D(component_array_dev, component_array_dev_pitch, component_array_host, component_array_host_pitch, component_array_host_pitch, num_components, cudaMemcpyHostToDevice)); //copy component array
		cu(cudaMemcpy(component_progress_dev, component_progress_host, 2 * num_components * sizeof(uint32_t), cudaMemcpyHostToDevice));
	}

	void StochasticCircuit::copy_data_from_device() {
		copy_data_from_device(sim_length);
	}

	void StochasticCircuit::copy_data_from_device(uint32_t net_length) {
		if (host_only) return;

		size_t width = (__min(net_length, sim_length) + 31) / 32 * sizeof(uint32_t);

		cu(cudaMemcpy2D(net_values_host, net_values_host_pitch, net_values_dev, net_values_dev_pitch, width, num_nets, cudaMemcpyDeviceToHost)); //copy net values
		cu(cudaMemcpy(net_progress_host, net_progress_dev, num_nets * sizeof(uint32_t), cudaMemcpyDeviceToHost)); //copy net progress
		cu(cudaMemcpy2D(component_array_host, component_array_host_pitch, component_array_dev, component_array_dev_pitch, component_array_host_pitch, num_components, cudaMemcpyDeviceToHost)); //copy component array
		cu(cudaMemcpy(component_progress_host, component_progress_dev, 2 * num_components * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	}

	void StochasticCircuit::simulate_circuit_host_only() {
		std::vector<uint32_t> last_round_possible_progress(num_components, 0);

		while (!simulation_finished) { //iterate over all components until simulation is finished
			int finished_components = 0;

			for (uint32_t i = 0; i < num_components; i++) { //sequentially go through components
				CircuitComponent* comp = components_host[i];

				comp->calculate_simulation_progress_host();

				if (comp->next_sim_progress() == last_round_possible_progress[i]) { //check and mark finished components
					finished_components++;
					continue;
				}
				last_round_possible_progress[i] = comp->next_sim_progress();

				comp->simulate_step_host(); //simulate next step
				comp->sim_step_finished_host();
			}

			if (finished_components == num_components) simulation_finished = true; //done if all components finished
		}
	}

	void StochasticCircuit::simulate_circuit() {
		if (host_only) {
			simulate_circuit_host_only();
			return;
		}

		copy_data_to_device();
		simulate_circuit_dev_nocopy();
		copy_data_from_device();
	}

	__global__ void calc_sim_progress(CircuitComponent** components, uint32_t count) {
		auto comp = blockIdx.x * blockDim.x + threadIdx.x;
		if (comp < count) components[comp]->calculate_simulation_progress_dev();
	}

	__global__ void exec_sim_step(CircuitComponent** components, uint32_t* comp_indices, uint32_t* comp_counts, uint32_t* comp_offsets) {
		auto type = blockIdx.z;
		auto comp = blockIdx.x * blockDim.x + threadIdx.x;
		if (comp < comp_counts[type]) components[comp_indices[comp_offsets[type] + comp]]->simulate_step_dev();
	}

	__global__ void finish_sim_step(CircuitComponent** components, uint32_t* comp_indices, uint32_t count) {
		auto comp = blockIdx.x * blockDim.x + threadIdx.x;
		if (comp < count) components[comp_indices[comp]]->sim_step_finished_dev();
	}

	void StochasticCircuit::simulate_circuit_dev_nocopy() {
		uint32_t block_size_calcp = __min(num_components, 256);
		uint32_t num_blocks_calcp = block_size_calcp == 0 ? 0 : (num_components + block_size_calcp - 1) / block_size_calcp;

		std::vector<uint32_t> last_round_possible_progress(num_components, 0);

		std::vector<uint32_t> sim_comb;
		std::vector<uint32_t> comb_type_counts;
		std::vector<uint32_t> comb_type_offsets;
		std::vector<uint32_t> sim_seq;
		std::vector<uint32_t> seq_type_counts;
		std::vector<uint32_t> seq_type_offsets;

		uint32_t* sim_comb_dev;
		uint32_t* comb_type_counts_dev;
		uint32_t* comb_type_offsets_dev;
		uint32_t* sim_seq_dev;
		uint32_t* seq_type_counts_dev;
		uint32_t* seq_type_offsets_dev;

		cu(cudaMalloc(&sim_comb_dev, num_components_comb * sizeof(uint32_t)));
		cu(cudaMalloc(&comb_type_counts_dev, num_component_types * sizeof(uint32_t)));
		cu(cudaMalloc(&comb_type_offsets_dev, num_component_types * sizeof(uint32_t)));
		cu(cudaMalloc(&sim_seq_dev, num_components_seq * sizeof(uint32_t)));
		cu(cudaMalloc(&seq_type_counts_dev, num_component_types * sizeof(uint32_t)));
		cu(cudaMalloc(&seq_type_offsets_dev, num_component_types * sizeof(uint32_t)));

		while (!simulation_finished) { //run simulation rounds until simulation is finished
			int finished_components = 0;

			calc_sim_progress<<<num_blocks_calcp, block_size_calcp>>>(components_dev, num_components); //calculate progress for components
			copy_component_progress_from_device(); //copy component progress to host

			//combinatorial components
			sim_comb.clear();
			comb_type_counts.clear();
			comb_type_offsets.clear();
			comb_type_offsets.push_back(0);
			uint32_t last_type = 0;
			uint32_t comb_sim_words = 0;
			for (uint32_t i = 0; i < num_components_comb; i++) {
				CircuitComponent* comp = components_host[i];
				uint32_t ctype = comp->component_type;

				if (comp->next_sim_progress() == last_round_possible_progress[i]) { //check and mark finished components
					finished_components++;
					continue;
				}
				last_round_possible_progress[i] = comp->next_sim_progress();

				auto words = comp->next_sim_progress_word() - comp->current_sim_progress_word();
				if (words > comb_sim_words) comb_sim_words = words; //remember largest number of words to be simulated

				//arrange components by types and remember counts and offsets
				if (ctype == last_type) {
					comb_type_counts.back()++;
				}
				else {
					if (!comb_type_counts.empty()) comb_type_offsets.push_back(comb_type_offsets.back() + comb_type_counts.back());
					comb_type_counts.push_back(1);
					last_type = ctype;
				}
				sim_comb.push_back(i);
			}

			if (!sim_comb.empty()) { //if combinatorial components need to be simulated
				//copy component pointers, counts, offsets to device
				cu(cudaMemcpy(sim_comb_dev, sim_comb.data(), sim_comb.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
				cu(cudaMemcpy(comb_type_counts_dev, comb_type_counts.data(), comb_type_counts.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
				cu(cudaMemcpy(comb_type_offsets_dev, comb_type_offsets.data(), comb_type_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

				uint32_t block_size_y = __min(comb_sim_words, 64); //block size y: words to simulate

				auto num_threads_comp = *std::max_element(comb_type_counts.begin(), comb_type_counts.end());
				uint32_t block_size_x = __min(num_threads_comp, 256 / block_size_y); //block size x: components of same type to simulate

				dim3 block_size(block_size_x, block_size_y);

				//grid size x and y: component/word count split into multiple blocks
				//grid size z: component types
				dim3 grid_size((num_threads_comp + block_size_x - 1) / block_size_x, (comb_sim_words + block_size_y - 1) / block_size_y, comb_type_counts.size());

				//simulate
				exec_sim_step<<<grid_size, block_size>>>(components_dev, sim_comb_dev, comb_type_counts_dev, comb_type_offsets_dev);

				//mark step as finished
				uint32_t block_size_fin = __min(sim_comb.size(), 256);
				uint32_t num_blocks_fin = (sim_comb.size() + block_size_fin - 1) / block_size_fin;
				finish_sim_step<<<num_blocks_fin, block_size_fin>>>(components_dev, sim_comb_dev, sim_comb.size());
			}

			//sequential components, similar to combinatorial as shown above
			sim_seq.clear();
			seq_type_counts.clear();
			seq_type_offsets.clear();
			seq_type_offsets.push_back(0);
			for (uint32_t i = num_components_comb; i < num_components; i++) {
				CircuitComponent* comp = components_host[i];
				uint32_t ctype = comp->component_type;

				if (comp->next_sim_progress() == last_round_possible_progress[i]) { //check and mark finished components
					finished_components++;
					continue;
				}
				last_round_possible_progress[i] = comp->next_sim_progress();

				if (ctype == last_type) {
					seq_type_counts.back()++;
				}
				else {
					if (!seq_type_counts.empty()) seq_type_offsets.push_back(seq_type_offsets.back() + seq_type_counts.back());
					seq_type_counts.push_back(1);
					last_type = ctype;
				}
				sim_seq.push_back(i);
			}

			if (!sim_seq.empty()) {
				cu(cudaMemcpy(sim_seq_dev, sim_seq.data(), sim_seq.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
				cu(cudaMemcpy(seq_type_counts_dev, seq_type_counts.data(), seq_type_counts.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
				cu(cudaMemcpy(seq_type_offsets_dev, seq_type_offsets.data(), seq_type_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

				auto max_num_threads = *std::max_element(seq_type_counts.begin(), seq_type_counts.end());
				uint32_t block_size = __min(max_num_threads, 256); //only one thread per component -> block size y = 1, grid size y = 1
				dim3 grid_size((max_num_threads + block_size - 1) / block_size, 1, seq_type_counts.size());

				exec_sim_step<<<grid_size, block_size>>>(components_dev, sim_seq_dev, seq_type_counts_dev, seq_type_offsets_dev);

				uint32_t block_size_fin = __min(sim_seq.size(), 256);
				uint32_t num_blocks_fin = (sim_seq.size() + block_size_fin - 1) / block_size_fin;
				finish_sim_step<<<num_blocks_fin, block_size_fin>>>(components_dev, sim_seq_dev, sim_seq.size());
			}

			if (finished_components == num_components) simulation_finished = true; //done if all components finished
		}

		cu(cudaFree(sim_comb_dev));
		cu(cudaFree(comb_type_counts_dev));
		cu(cudaFree(comb_type_offsets_dev));
		cu(cudaFree(sim_seq_dev));
		cu(cudaFree(seq_type_counts_dev));
		cu(cudaFree(seq_type_offsets_dev));
	}

	StochasticNumber* StochasticCircuit::get_net_value(uint32_t net) {
		auto progress = net_progress_host[net];
		if (progress == 0) throw;
		return new StochasticNumber(progress, net_values_host + (sim_length_words * net));
	}

	double StochasticCircuit::get_net_value_unipolar(uint32_t net) {
		auto num = get_net_value(net);
		auto ret = num->get_value_unipolar();
		delete num;
		return ret;
	}

	double StochasticCircuit::get_net_value_bipolar(uint32_t net) {
		auto num = get_net_value(net);
		auto ret = num->get_value_bipolar();
		delete num;
		return ret;
	}

	void StochasticCircuit::copy_component_progress_from_device() {
		cu(cudaMemcpy(component_progress_host, component_progress_dev, 2 * num_components * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	}

}

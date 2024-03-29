﻿#include "cuda_base.cuh"

#include <algorithm>
#include <numeric>

#include "StochasticCircuitFactory.cuh"
#include "StochasticCircuit.cuh"
#include "StochasticNumber.cuh"
#include "CircuitComponent.cuh"
#include "CombinatorialComponent.cuh"
#include "SequentialComponent.cuh"

namespace scsim {

	StochasticCircuitFactory::StochasticCircuitFactory(bool host_only, bool force_fast_mempitch_nets, bool force_fast_mempitch_components) : host_only(host_only),
		force_fast_mempitch_nets(force_fast_mempitch_nets), force_fast_mempitch_components(force_fast_mempitch_components) {
		reset();
	}

	StochasticCircuitFactory::~StochasticCircuitFactory() {
		for (size_t i = 0; i < components.size(); i++) {
			delete components[i];
		}

		if (net_numbers != nullptr) {
			for (uint32_t i = 0; i < num_nets; i++) {
				net_numbers[i].~StochasticNumber();
			}
		}

		if (!host_only) {
			cu_ignore_error(cudaHostUnregister(net_values_host));
			cu_ignore_error(cudaHostUnregister(net_progress_host));
			cu_ignore_error(cudaHostUnregister(component_progress_host));
			cu_ignore_error(cudaHostUnregister(component_array_host));

			cu_ignore_error(cudaFree(net_values_dev));
			cu_ignore_error(cudaFree(net_progress_dev));
			cu_ignore_error(cudaFree(components_dev));
			cu_ignore_error(cudaFree(component_progress_dev));
			cu_ignore_error(cudaFree(component_array_dev));
			cu_ignore_error(cudaFree(component_io_dev));
			cu_ignore_error(cudaFree(component_io_offsets_dev));

			cu_ignore_error(cudaFreeHost(dev_offset_scratchpad));
			cu_ignore_error(cudaFreeHost(dev_pointers_scratchpad));
		}

		free(net_values_host);
		free(net_progress_host);
		free(components_host);
		free(component_progress_host);
		free(component_array_host);
		free(component_io_host);
		free(component_io_offsets_host);
		free(net_numbers);
	}

	void StochasticCircuitFactory::reset() {
		sim_length = 0;
		num_nets = 0;
		num_comb_comp = 0;
		num_seq_comp = 0;
		components.clear();
		driven_nets.clear();
		max_component_size = 0;
		max_component_align = 0;
		component_io.clear();

		net_values_host = nullptr;
		net_values_dev = nullptr;
		net_progress_host = nullptr;
		net_progress_dev = nullptr;
		components_host = nullptr;
		components_dev = nullptr;
		component_progress_host = nullptr;
		component_progress_dev = nullptr;
		component_array_host = nullptr;
		component_array_dev = nullptr;
		component_io_host = nullptr;
		component_io_dev = nullptr;
		component_io_offsets_host = nullptr;
		component_io_offsets_dev = nullptr;
		dev_offset_scratchpad = nullptr;
		dev_pointers_scratchpad = nullptr;
		net_numbers = nullptr;
	}

	StochasticCircuit* StochasticCircuitFactory::create_circuit() {
		if (sim_length == 0) throw std::runtime_error("create_circuit: Simulation time span has not been set.");
		if (num_nets == 0 || components.size() == 0) throw std::runtime_error("create_circuit: Factory is empty, so circuit creation is impossible.");

		StochasticCircuit* circuit = nullptr;

		size_t sim_length_words = (sim_length + 31) / 32;

		//host-side circuit state
#ifndef ALLOCSIZETEST
		net_values_host = (uint32_t*)malloc(sim_length_words * num_nets * sizeof(uint32_t));
		net_progress_host = (uint32_t*)calloc(num_nets, sizeof(uint32_t));
		components_host_index = (uint32_t*)malloc(components.size() * sizeof(uint32_t));
		components_host = (CircuitComponent**)malloc(components.size() * sizeof(CircuitComponent*));
		components_dev = nullptr;
		component_progress_host = (uint32_t*)calloc(2 * components.size(), sizeof(uint32_t));
		component_progress_dev = nullptr;
		component_io_host = (uint32_t*)malloc(component_io.size() * sizeof(uint32_t));
		component_io_dev = nullptr;
		component_io_offsets_host = (size_t*)malloc(component_io.size() * sizeof(size_t));
		component_io_offsets_dev = nullptr;
		net_numbers = (StochasticNumber*)calloc(num_nets, sizeof(StochasticNumber));

		if (net_values_host == nullptr || net_progress_host == nullptr || components_host == nullptr || component_progress_host == nullptr ||
			component_io_host == nullptr || component_io_offsets_host == nullptr || net_numbers == nullptr) throw std::runtime_error("create_circuit: Host-side state allocation failed.");
#endif

		size_t component_pitch;
		size_t component_array_dev_pitch;

		//sort components by type for automatic grouping during simulation
		std::vector<uint32_t> sort_indices(components.size());
		std::iota(sort_indices.begin(), sort_indices.end(), 0);
		std::stable_sort(sort_indices.begin(), sort_indices.end(), [&](auto a, auto b) { return components[a]->component_type < components[b]->component_type; });

#ifndef ALLOCSIZETEST
		for (uint32_t i = 0; i < components.size(); i++) {
			auto sort_index = sort_indices[i];
			components_host[i] = components[sort_index];
			components_host_index[sort_index] = i;
		}

		memcpy(component_io_host, component_io.data(), component_io.size() * sizeof(uint32_t));
#endif

		uint32_t num_component_types = 0;
		uint32_t last_type = 0;
		for (uint32_t i = 0; i < components.size(); i++) { //count component types
			uint32_t type = components[i]->component_type;
			if (type != last_type) {
				num_component_types++;
				last_type = type;
			}
		}

#ifndef ALLOCSIZETEST
		for (uint32_t i = 0; i < num_nets; i++) { //create stochastic numbers corresponding to nets
			new (net_numbers + i) StochasticNumber(net_values_host + (sim_length_words * i), net_progress_host + i, sim_length);
		}
#endif

		if (host_only) { //host-only circuit: create circuit and initialize components, that's all
#ifndef ALLOCSIZETEST
			circuit = new StochasticCircuit(sim_length, num_nets, net_values_host, net_progress_host, num_comb_comp, num_seq_comp, components_host_index, components_host, component_progress_host,
				num_component_types, component_io_host, component_io_offsets_host, net_numbers);
			for (uint32_t i = 0; i < components.size(); i++) {
				components_host[i]->init_with_circuit(circuit, component_progress_host + (2ull * i), nullptr, nullptr);
			}
#endif
		}
		else { //device-assisted circuit
			size_t net_pitch_dev;

#ifndef ALLOCSIZETEST
			//page-lock host-side circuit state if possible
			cu_ignore_error(cudaHostRegister(net_values_host, sim_length_words * num_nets * sizeof(uint32_t), cudaHostRegisterDefault));
			cu_ignore_error(cudaHostRegister(net_progress_host, num_nets * sizeof(uint32_t), cudaHostRegisterDefault));
			cu_ignore_error(cudaHostRegister(component_progress_host, 2 * components.size() * sizeof(uint32_t), cudaHostRegisterDefault));
#endif

			//device-side circuit state
#ifdef ALLOCSIZETEST
			size_t theoreticalsize = sim_length_words * sizeof(uint32_t) * num_nets;
			if (sim_length_words * sizeof(uint32_t) >= FAST_MEMPITCH_THRESHOLD || force_fast_mempitch_nets) { //simulation length long enough, or forced: use fast memory pitching
				cu(cudaMallocPitch(&net_values_dev, &net_pitch_dev, sim_length_words * sizeof(uint32_t), 10));
				cu(cudaFree(net_values_dev));
			} else { //otherwise, use simple memory pitching, maybe slower
				net_pitch_dev = sim_length_words * sizeof(uint32_t);
			}
			size_t allocsize = net_pitch_dev * num_nets
				+ num_nets * sizeof(uint32_t)
				+ components.size() * sizeof(CircuitComponent*)
				+ 2 * components.size() * sizeof(uint32_t)
				+ component_io.size() * sizeof(uint32_t)
				+ component_io.size() * sizeof(size_t);
#else
			if (sim_length_words * sizeof(uint32_t) >= FAST_MEMPITCH_THRESHOLD || force_fast_mempitch_nets) { //simulation length long enough, or forced: use fast memory pitching
				cu(cudaMallocPitch(&net_values_dev, &net_pitch_dev, sim_length_words * sizeof(uint32_t), num_nets));
			} else { //otherwise, use simple memory pitching, maybe slower
				net_pitch_dev = sim_length_words * sizeof(uint32_t);
				cu(cudaMalloc(&net_values_dev, net_pitch_dev * num_nets));
			}
			cu(cudaMalloc(&net_progress_dev, num_nets * sizeof(uint32_t)));
			cu(cudaMalloc(&components_dev, components.size() * sizeof(CircuitComponent*)));
			cu(cudaMalloc(&component_progress_dev, 2 * components.size() * sizeof(uint32_t)));
			cu(cudaMalloc(&component_io_dev, component_io.size() * sizeof(uint32_t)));
			cu(cudaMalloc(&component_io_offsets_dev, component_io.size() * sizeof(size_t)));
#endif

			auto max_comp_size_misalignment = max_component_size % max_component_align;
			if (max_comp_size_misalignment == 0) component_pitch = max_component_size;
			else component_pitch = max_component_size - max_comp_size_misalignment + max_component_align;

			//component array, host side, page-locked if possible
#ifndef ALLOCSIZETEST
			component_array_host = (char*)malloc(components.size() * component_pitch);
			if (component_array_host == nullptr) throw std::runtime_error("create_circuit: Host-side state allocation failed.");
			cu_ignore_error(cudaHostRegister(component_array_host, components.size() * component_pitch, cudaHostRegisterDefault));
#endif

			//component array, device side
#ifdef ALLOCSIZETEST
			theoreticalsize += component_pitch * components.size();
			if (component_pitch >= FAST_MEMPITCH_THRESHOLD || force_fast_mempitch_components) { //component pitch long enough, or forced: use fast memory pitching
				cu(cudaMallocPitch(&component_array_dev, &component_array_dev_pitch, component_pitch, 10));
				cu(cudaFree(component_array_dev));
			} else { //otherwise, use simple memory pitching, maybe slower
				component_array_dev_pitch = component_pitch;
			}
			allocsize += component_array_dev_pitch * components.size();

			printf("Nets: %u, Components: %zu\n", num_nets, components.size());
			printf("Theoretical size: %f MB, Allocated size: %f MB\n", (double)theoreticalsize / 1048576.0, (double)allocsize / 1048576.0);
			printf("Net pitch: %zu/%zu, Comp pitch: %zu/%zu\n", sim_length_words * sizeof(uint32_t), net_pitch_dev, component_pitch, component_array_dev_pitch);

			for (size_t i = 0; i < components.size(); i++) { //cleanup components
				delete components[i];
			}
#else
			if (component_pitch >= FAST_MEMPITCH_THRESHOLD || force_fast_mempitch_components) { //component pitch long enough, or forced: use fast memory pitching
				cu(cudaMallocPitch(&component_array_dev, &component_array_dev_pitch, component_pitch, components.size()));
			} else { //otherwise, use simple memory pitching, maybe slower
				component_array_dev_pitch = component_pitch;
				cu(cudaMalloc(&component_array_dev, component_array_dev_pitch * components.size()));
			}
			
			circuit = new StochasticCircuit(sim_length, num_nets, net_values_host, net_values_dev, net_pitch_dev, net_progress_host, net_progress_dev, num_comb_comp, num_seq_comp,
				components_host_index, components_host, components_dev, component_array_host, component_pitch, component_array_dev, component_array_dev_pitch, component_progress_host,
				component_progress_dev,	num_component_types, component_io_host, component_io_dev, component_io_offsets_host, component_io_offsets_dev, net_numbers);

			cu(cudaMallocHost(&dev_offset_scratchpad, component_io.size() * sizeof(size_t)));
			cu(cudaMallocHost(&dev_pointers_scratchpad, components.size() * sizeof(CircuitComponent*)));

			for (uint32_t i = 0; i < components.size(); i++) {
				auto comp = components_host[i];

				components_host[i] = (CircuitComponent*)(component_array_host + i * component_pitch); //redirect component pointers to component array

				comp->dev_ptr = (CircuitComponent*)(component_array_dev + i * component_array_dev_pitch); //link components's own device pointer to component array
				dev_pointers_scratchpad[i] = comp->dev_ptr; //direct device-side pointers to component array

				comp->init_with_circuit(circuit, component_progress_host + (2ull * i), component_progress_dev + (2ull * i), dev_offset_scratchpad);
				
				memcpy(components_host[i], comp, comp->mem_obj_size); //copy component to component array
			}

			cu(cudaMemcpy(component_io_dev, component_io_host, component_io.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
			cu(cudaMemcpy(component_io_offsets_dev, dev_offset_scratchpad, component_io.size() * sizeof(size_t), cudaMemcpyHostToDevice));
			cu(cudaMemcpy(components_dev, dev_pointers_scratchpad, components.size() * sizeof(CircuitComponent*), cudaMemcpyHostToDevice));

			cu_ignore_error(cudaFreeHost(dev_offset_scratchpad));
			cu_ignore_error(cudaFreeHost(dev_pointers_scratchpad));
			dev_offset_scratchpad = nullptr;
			dev_pointers_scratchpad = nullptr;

			for (auto comp : components) operator delete(comp); //free original allocation for components (without destroying the actual components)

			circuit->reset_circuit();
#endif
		}

		reset();

		return circuit;
	}

	void StochasticCircuitFactory::set_sim_length(uint32_t sim_length) {
		if (sim_length == 0) throw std::runtime_error("set_sim_length: Simulation time span must be greater than zero.");
		this->sim_length = sim_length;
	}

	uint32_t StochasticCircuitFactory::add_net() {
		driven_nets.push_back(false); //new net initially undriven
		return num_nets++;
	}

	std::pair<uint32_t, uint32_t> StochasticCircuitFactory::add_nets(uint32_t count) {
		if (count == 0) throw std::runtime_error("add_nets: Number of added nets must be greater than zero.");
		driven_nets.resize(driven_nets.size() + count, false); //new nets initially undriven
		auto first = num_nets;
		num_nets += count;
		return std::make_pair(first, num_nets - 1);
	}

	uint32_t StochasticCircuitFactory::add_component(CombinatorialComponent* component) {
		uint32_t index = add_component_internal(component);
		num_comb_comp++;
		return index;
	}

	uint32_t StochasticCircuitFactory::get_num_nets() const {
		return num_nets;
	}

	uint32_t StochasticCircuitFactory::get_num_components() const {
		return components.size();
	}

	uint32_t StochasticCircuitFactory::add_component_internal(CircuitComponent* component) {
		size_t index = components.size();

		for (size_t i = 0; i < component->num_inputs; i++) {
			//disallow invalid/nonexistent nets
			if (component->inputs_host[i] >= num_nets) throw std::runtime_error("add_component: Component references a net that does not exist.");
		}

		for (size_t i = 0; i < component->num_outputs; i++) {
			auto net = component->outputs_host[i];
			//disallow invalid/nonexistent nets and multiple outputs per net
			if (net >= num_nets) throw std::runtime_error("add_component: Component references a net that does not exist.");
			if (driven_nets[net]) throw std::runtime_error("add_component: An output net is already used as another component's output.");
		}

		for (size_t i = 0; i < component->num_outputs; i++) {
			auto net = component->outputs_host[i];
			driven_nets[net] = true; //mark all output nets as driven
		}

		//remember largest component memory size and alignment in circuit (minimum component array pitch)
		if (component->mem_obj_size > max_component_size) max_component_size = component->mem_obj_size;
		if (component->mem_align > max_component_align) max_component_align = component->mem_align;

		component->index = index;

		components.push_back(component);

		return index;
	}

}

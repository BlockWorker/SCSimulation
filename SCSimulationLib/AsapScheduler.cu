#include "AsapScheduler.cuh"
#include "HostAsapScheduler.h"
#include "CircuitComponent.cuh"
#include "StochasticCircuit.cuh"

namespace scsim {

	typedef struct {
		size_t sim_comb_offset;
		size_t comb_type_counts_offset;
		size_t comb_type_offsets_offset;
		size_t sim_seq_offset;
		size_t seq_type_counts_offset;
		size_t seq_type_offsets_offset;
		size_t combinatorial_count;
		size_t total_count;
		uint32_t comb_type_max_count;
		uint32_t seq_type_max_count;
		uint32_t number_comb_types;
		uint32_t number_seq_types;
	} bucket_info_t;

	AsapScheduler::AsapScheduler() : Scheduler("AsapScheduler") {
		circuit = nullptr;
		host_scheduler = new HostAsapScheduler();
		cuda_graph = nullptr;
		cuda_graph_exec = nullptr;
		sim_comb_dev = nullptr;
		comb_type_counts_dev = nullptr;
		comb_type_offsets_dev = nullptr;
		sim_seq_dev = nullptr;
		seq_type_counts_dev = nullptr;
		seq_type_offsets_dev = nullptr;
	}
	
	AsapScheduler::~AsapScheduler() {
		delete host_scheduler;

		cudaFree(sim_comb_dev);
		cudaFree(comb_type_counts_dev);
		cudaFree(comb_type_offsets_dev);
		cudaFree(sim_seq_dev);
		cudaFree(seq_type_counts_dev);
		cudaFree(seq_type_offsets_dev);

		cudaGraphDestroy(cuda_graph);
		cudaGraphExecDestroy(cuda_graph_exec);
	}

	__global__ void asap_calc_sim_progress(CircuitComponent** components, uint32_t* indices_comb, uint32_t count_comb, uint32_t* indices_seq, uint32_t count_seq) {
		auto comp = blockIdx.x * blockDim.x + threadIdx.x;
		if (comp < count_comb) components[indices_comb[comp]]->calculate_simulation_progress_dev();
		else {
			comp -= count_comb;
			if (comp < count_seq) components[indices_seq[comp]]->calculate_simulation_progress_dev();
		}
	}

	__global__ void exec_sim_step(CircuitComponent** components, uint32_t* comp_indices, uint32_t* comp_counts, uint32_t* comp_offsets);

	__global__ void finish_sim_step(CircuitComponent** components, uint32_t* comp_indices, uint32_t count);

	void AsapScheduler::compile(StochasticCircuit* circuit) {
		this->circuit = circuit;
		host_scheduler->compile(circuit);

		std::vector<uint32_t> sim_comb;
		std::vector<uint32_t> comb_type_counts;
		std::vector<uint32_t> comb_type_offsets;
		std::vector<uint32_t> sim_seq;
		std::vector<uint32_t> seq_type_counts;
		std::vector<uint32_t> seq_type_offsets;

		std::vector<bucket_info_t> bucket_info;

		for (auto bucket : host_scheduler->schedule_buckets) { //process buckets, first round: generate kernel data structures
			//add bucket info with offsets in corresponding vectors, and initial 0 combinatorial count
			bucket_info.push_back({ sim_comb.size(), comb_type_counts.size(), comb_type_offsets.size(), sim_seq.size(), seq_type_counts.size(), seq_type_offsets.size(), 0, bucket.size(), 0, 0, 0, 0 });
			auto& info = bucket_info.back();

			uint32_t last_type = 0;

			for (auto index : bucket) { //process all components in bucket
				auto comp = circuit->components_host[index];
				
				if (is_combinatorial(comp)) { //same processing as in StochasticCircuit: divide into component types, specified by counts and offsets
					if (comp->component_type == last_type) {
						comb_type_counts.back()++;
					} else {
						if (comb_type_counts.empty()) {
							comb_type_offsets.push_back(0);
						} else {
							auto count = comb_type_counts.back();
							comb_type_offsets.push_back(comb_type_offsets.back() + count);
							if (count > info.comb_type_max_count) info.comb_type_max_count = count;
						}
						comb_type_counts.push_back(1);
						last_type = comp->component_type;
						info.number_comb_types++;
					}
					sim_comb.push_back(index);
					info.combinatorial_count++;
				} else {
					if (comp->component_type == last_type) {
						seq_type_counts.back()++;
					} else {
						if (seq_type_counts.empty()) {
							seq_type_offsets.push_back(0);
						} else {
							auto count = seq_type_counts.back();
							seq_type_offsets.push_back(seq_type_offsets.back() + count);
							if (count > info.seq_type_max_count) info.seq_type_max_count = count;
						}
						seq_type_counts.push_back(1);
						last_type = comp->component_type;
						info.number_seq_types++;
					}
					sim_seq.push_back(index);
				}
			}

			if (comb_type_counts.size() > 0 && comb_type_counts.back() > info.comb_type_max_count) info.comb_type_max_count = comb_type_counts.back();
			if (seq_type_counts.size() > 0 && seq_type_counts.back() > info.seq_type_max_count) info.seq_type_max_count = seq_type_counts.back();
		}

		cu(cudaMalloc(&sim_comb_dev, sim_comb.size() * sizeof(uint32_t)));
		cu(cudaMalloc(&comb_type_counts_dev, comb_type_counts.size() * sizeof(uint32_t)));
		cu(cudaMalloc(&comb_type_offsets_dev, comb_type_offsets.size() * sizeof(uint32_t)));
		cu(cudaMalloc(&sim_seq_dev, sim_seq.size() * sizeof(uint32_t)));
		cu(cudaMalloc(&seq_type_counts_dev, seq_type_counts.size() * sizeof(uint32_t)));
		cu(cudaMalloc(&seq_type_offsets_dev, seq_type_offsets.size() * sizeof(uint32_t)));

		cu(cudaMemcpy(sim_comb_dev, sim_comb.data(), sim_comb.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
		cu(cudaMemcpy(comb_type_counts_dev, comb_type_counts.data(), comb_type_counts.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
		cu(cudaMemcpy(comb_type_offsets_dev, comb_type_offsets.data(), comb_type_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
		cu(cudaMemcpy(sim_seq_dev, sim_seq.data(), sim_seq.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
		cu(cudaMemcpy(seq_type_counts_dev, seq_type_counts.data(), seq_type_counts.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
		cu(cudaMemcpy(seq_type_offsets_dev, seq_type_offsets.data(), seq_type_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));


		//create cuda graph
		cu(cudaGraphCreate(&cuda_graph, 0));

		cudaGraphNode_t prev_nodes[2] = { nullptr, nullptr }; //nodes of previous bucket, combinatorial and sequential
		uint8_t prev_nodes_existence = 0; //existence flags of the above: bit 0 (value 1) = combinatorial, bit 1 (value 2) = sequential

		cudaGraphNode_t calcp_node = nullptr; //node for progress calculation of current bucket
		cudaGraphNode_t temp_node = nullptr; //temporary node for simulation nodes, preceding finish nodes

		cudaKernelNodeParams node_params = { 0 };
		void* param_pointers[10] = { 0 };

		node_params.kernelParams = param_pointers;
		node_params.extra = NULL;
		node_params.sharedMemBytes = 0;

		uint32_t block_size_y = __min(circuit->sim_length_words, 64); //block size y: words to simulate
		uint32_t grid_size_y = (circuit->sim_length_words + block_size_y - 1) / block_size_y;

		for (auto& info : bucket_info) { //process buckets, second round: generate cuda graph
			auto sequential_count = info.total_count - info.combinatorial_count;

			uint32_t block_size_calcp = __min(info.total_count, 256);
			node_params.blockDim = dim3(block_size_calcp);
			node_params.gridDim = dim3((info.total_count + block_size_calcp - 1) / block_size_calcp);
			node_params.func = &asap_calc_sim_progress;

			uint32_t* comb_indices = sim_comb_dev + info.sim_comb_offset;
			uint32_t* seq_indices = sim_seq_dev + info.sim_seq_offset;
			param_pointers[0] = (void*)&circuit->components_dev;
			param_pointers[1] = (void*)&comb_indices;
			param_pointers[2] = (void*)&info.combinatorial_count;
			param_pointers[3] = (void*)&seq_indices;
			param_pointers[4] = (void*)&sequential_count;

			//determine predecessors
			cudaGraphNode_t* pred_ptr = prev_nodes;
			size_t pred_count;
			switch (prev_nodes_existence) {
				case 0:
					pred_count = 0;
					break;
				case 2:
					pred_ptr++;
					[[fallthrough]];
				case 1:
					pred_count = 1;
					break;
				case 3:
					pred_count = 2;
					break;
			}

			//create node for progress calculation with the above parameters, predecessors are previous finish nodes, their count is given by the sum of the two existence bits
			cu(cudaGraphAddKernelNode(&calcp_node, cuda_graph, prev_nodes, ((prev_nodes_existence & 1) + (prev_nodes_existence >> 1)), &node_params));

			//reset existence flags for nodes
			prev_nodes_existence = 0;

			if (info.number_comb_types > 0) { //combinatorial components if present
				//make simulation node
				uint32_t block_size_x = __min(info.comb_type_max_count, 256 / block_size_y); //block size x: components of same type to simulate
				node_params.blockDim = dim3(block_size_x, block_size_y);
				//grid size x and y: component/word count split into multiple blocks
				//grid size z: component types
				node_params.gridDim = dim3((info.comb_type_max_count + block_size_x - 1) / block_size_x, grid_size_y, info.number_comb_types);
				node_params.func = &exec_sim_step;

				uint32_t* counts = comb_type_counts_dev + info.comb_type_counts_offset;
				uint32_t* offsets = comb_type_offsets_dev + info.comb_type_offsets_offset;
				param_pointers[0] = (void*)&circuit->components_dev;
				param_pointers[1] = (void*)&sim_comb_dev;
				param_pointers[2] = (void*)&counts;
				param_pointers[3] = (void*)&offsets;

				cu(cudaGraphAddKernelNode(&temp_node, cuda_graph, &calcp_node, 1, &node_params));

				//make finish node
				uint32_t block_size_fin = __min(info.combinatorial_count, 256);
				node_params.blockDim = dim3(block_size_fin);
				node_params.gridDim = dim3((info.combinatorial_count + block_size_fin - 1) / block_size_fin);
				node_params.func = &finish_sim_step;

				param_pointers[0] = (void*)&circuit->components_dev;
				param_pointers[1] = (void*)&comb_indices;
				param_pointers[2] = (void*)&info.combinatorial_count;

				cu(cudaGraphAddKernelNode(prev_nodes, cuda_graph, &temp_node, 1, &node_params));

				//mark combinatorial node as present
				prev_nodes_existence |= 1;
			}

			if (info.number_seq_types > 0) { //sequential components if present
				//make simulation node
				uint32_t block_size_x = __min(info.seq_type_max_count, 256 / block_size_y); //block size x: components of same type to simulate
				node_params.blockDim = dim3(block_size_x, 1);
				//grid size x: component count split into multiple blocks
				//grid size z: component types
				node_params.gridDim = dim3((info.seq_type_max_count + block_size_x - 1) / block_size_x, 1, info.number_seq_types);
				node_params.func = &exec_sim_step;

				uint32_t* counts = seq_type_counts_dev + info.seq_type_counts_offset;
				uint32_t* offsets = seq_type_offsets_dev + info.seq_type_offsets_offset;
				param_pointers[0] = (void*)&circuit->components_dev;
				param_pointers[1] = (void*)&sim_seq_dev;
				param_pointers[2] = (void*)&counts;
				param_pointers[3] = (void*)&offsets;

				cu(cudaGraphAddKernelNode(&temp_node, cuda_graph, &calcp_node, 1, &node_params));

				//make finish node
				uint32_t block_size_fin = __min(sequential_count, 256);
				node_params.blockDim = dim3(block_size_fin);
				node_params.gridDim = dim3((sequential_count + block_size_fin - 1) / block_size_fin);
				node_params.func = &finish_sim_step;

				param_pointers[0] = (void*)&circuit->components_dev;
				param_pointers[1] = (void*)&seq_indices;
				param_pointers[2] = (void*)&sequential_count;

				cu(cudaGraphAddKernelNode(prev_nodes + 1, cuda_graph, &temp_node, 1, &node_params));

				//mark sequential node as present
				prev_nodes_existence |= 2;
			}

			if (prev_nodes_existence == 0) throw std::runtime_error("compile: Something went wrong, schedule bucket has no components");
		}

		cudaGraphNode_t error_node;
		char error_log[1024];
		cu(cudaGraphInstantiate(&cuda_graph_exec, cuda_graph, &error_node, error_log, sizeof(error_log)));

		//cudaGraphDebugDotPrint(cuda_graph, "graph.dot", cudaGraphDebugDotFlagsVerbose | cudaGraphDebugDotFlagsKernelNodeParams | cudaGraphDebugDotFlagsKernelNodeAttributes);
	}

	bool AsapScheduler::is_compiled() const {
		return host_scheduler->is_compiled() && cuda_graph_exec != nullptr;
	}

	bool AsapScheduler::execute(bool host) {
		if (!is_compiled()) return false;

		if (host) return host_scheduler->execute(true);

		cu(cudaGraphLaunch(cuda_graph_exec, 0));
		cu_kernel_errcheck();

		return true;
	}

}
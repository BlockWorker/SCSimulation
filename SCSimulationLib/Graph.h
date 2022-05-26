#pragma once

//#include "library_export.h"
#include <stdint.h>
#include <vector>
#include <unordered_set>
//#include "library_export.h"

namespace scsim {

	class /*SCSIMAPI*/ Graph
	{
	public:
		Graph(uint32_t initial_vertices = 0);

		void clear_graph();
		void clear_edges();

		uint32_t add_vertex();
		void add_edge(uint32_t from, uint32_t to);

		uint32_t get_num_vertices() const;
		uint32_t get_num_edges() const;

		const std::unordered_set<uint32_t>& get_children(uint32_t parent) const;
		const std::unordered_set<uint32_t>& get_parents(uint32_t child) const;

		/// <returns>Set of all vertices with indegree zero ("roots" of the graph)</returns>
		const std::unordered_set<uint32_t>& get_indegree_zero() const;

		/// <returns>Set of all vertices with outdegree zero ("leaves" of the graph)</returns>
		const std::unordered_set<uint32_t>& get_outdegree_zero() const;

		/// <summary>
		/// Topologically sort the graph, if it is acyclic.
		/// </summary>
		/// <param name="order_out">Vector to store the topological order</param>
		/// <returns>true if successful (acyclic), false otherwise</returns>
		bool topological_sort(std::vector<uint32_t>& order_out) const;

		/// <summary>
		/// Find single-source shortest paths using a topological order as input.
		/// </summary>
		/// <param name="topological_order">Graph's topological order, required</param>
		/// <param name="source">Source vertex. May also be -1 to designate a phantom root vertex (parent of all vertices with indegree 0).</param>
		/// <param name="distances">Vector to store the shortest distances from the source.</param>
		/// <param name="predecessors">Optional vector to store the predececcors of vertices.</param>
		void topological_sssp(const std::vector<uint32_t>& topological_order, int64_t source, std::vector<int64_t>& distances, std::vector<int64_t>* predecessors = nullptr) const;

		/// <summary>
		/// Find single-source longest paths using a topological order as input.
		/// </summary>
		/// <param name="topological_order">Graph's topological order, required</param>
		/// <param name="source">Source vertex. May also be -1 to designate a phantom root vertex (parent of all vertices with indegree 0).</param>
		/// <param name="distances">Vector to store the longest distances from the source.</param>
		/// <param name="predecessors">Optional vector to store the predececcors of vertices.</param>
		void topological_sslp(const std::vector<uint32_t>& topological_order, int64_t source, std::vector<int64_t>& distances, std::vector<int64_t>* predecessors = nullptr) const;

		/// <summary>
		/// Find single-destination shortest paths using a topological order as input.
		/// </summary>
		/// <param name="topological_order">Graph's topological order, required</param>
		/// <param name="dest">Destination vertex. May also be -1 to designate a phantom sink vertex (child of all vertices with outdegree 0).</param>
		/// <param name="distances">Vector to store the shortest distances from the source.</param>
		/// <param name="successors">Optional vector to store the successors of vertices.</param>
		void topological_sdsp(const std::vector<uint32_t>& topological_order, int64_t dest, std::vector<int64_t>& distances, std::vector<int64_t>* successors = nullptr) const;

		/// <summary>
		/// Find single-destination longest paths using a topological order as input.
		/// </summary>
		/// <param name="topological_order">Graph's topological order, required</param>
		/// <param name="dest">Destination vertex. May also be -1 to designate a phantom sink vertex (child of all vertices with outdegree 0).</param>
		/// <param name="distances">Vector to store the longest distances from the source.</param>
		/// <param name="successors">Optional vector to store the successors of vertices.</param>
		void topological_sdlp(const std::vector<uint32_t>& topological_order, int64_t dest, std::vector<int64_t>& distances, std::vector<int64_t>* successors = nullptr) const;

	protected:
		uint32_t num_vertices;
		uint32_t num_edges;
		std::vector<std::unordered_set<uint32_t>> adj_list; //adjacency list: item i is the set of children of vertex i
		std::vector<std::unordered_set<uint32_t>> rev_adj_list; //reverse adjacency list: item i is the set of parents of vertex i
		std::unordered_set<uint32_t> indegree_zero; //set of vertices with indegree zero ("roots" of the graph)
		std::unordered_set<uint32_t> outdegree_zero; //set of vertices with outdegree zero ("leaves" of the graph)

	};

}

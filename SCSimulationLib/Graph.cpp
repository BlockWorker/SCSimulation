#include "Graph.h"

#include <stdexcept>
#include <forward_list>
#include <stack>
#include <numeric>

namespace scsim {

	Graph::Graph(uint32_t initial_vertices) {
		num_vertices = initial_vertices;
		num_edges = 0;
		adj_list.clear();
		rev_adj_list.clear();
		adj_list.reserve(initial_vertices);
		rev_adj_list.reserve(initial_vertices);
		for (uint32_t i = 0; i < initial_vertices; i++) {
			adj_list.emplace_back();
			rev_adj_list.emplace_back();
			indegree_zero.insert(i);
			outdegree_zero.insert(i);
		}
	}

	void Graph::clear_graph() {
		for each (auto edges in adj_list) edges.clear();
		for each (auto rev_edges in rev_adj_list) rev_edges.clear();
		num_edges = 0;
		adj_list.clear();
		rev_adj_list.clear();
		indegree_zero.clear();
		num_vertices = 0;
	}

	void Graph::clear_edges() {
		for each (auto edges in adj_list) edges.clear();
		for each (auto rev_edges in rev_adj_list) rev_edges.clear();
		for (uint32_t i = 0; i < num_vertices; i++) {
			indegree_zero.insert(i);
			outdegree_zero.insert(i);
		}
		num_edges = 0;
	}

	uint32_t Graph::add_vertex() {
		adj_list.emplace_back();
		rev_adj_list.emplace_back();
		indegree_zero.insert(num_vertices);
		outdegree_zero.insert(num_vertices);
		return num_vertices++;
	}

	void Graph::add_edge(uint32_t from, uint32_t to) {
		if (from >= num_vertices || to >= num_vertices) throw std::runtime_error("add_edge: Invalid source or destination index");

		adj_list[from].insert(to);
		rev_adj_list[to].insert(from);
		indegree_zero.erase(to);
		outdegree_zero.erase(from);
		num_edges++;
	}

	uint32_t Graph::get_num_vertices() const {
		return num_vertices;
	}

	uint32_t Graph::get_num_edges() const {
		return num_edges;
	}

	const std::unordered_set<uint32_t>& Graph::get_children(uint32_t parent) const {
		if (parent >= num_vertices) throw std::runtime_error("get_children: Invalid parent index");

		return adj_list[parent];
	}

	const std::unordered_set<uint32_t>& Graph::get_parents(uint32_t child) const {
		if (child >= num_vertices) throw std::runtime_error("get_parents: Invalid child index");

		return rev_adj_list[child];
	}

	const std::unordered_set<uint32_t>& Graph::get_indegree_zero() const {
		return indegree_zero;
	}

	const std::unordered_set<uint32_t>& Graph::get_outdegree_zero() const {
		return outdegree_zero;
	}

	bool Graph::topological_sort(std::vector<uint32_t>& order_out) const {
		order_out.clear();

		if (num_vertices == 0) return true;

		std::forward_list<uint32_t> order; //current order

		std::stack<std::pair<uint32_t, bool>> dfs_stack; //pair: vertex index, children pushed?
		std::vector<bool> visit_marks(num_vertices, false); //visited by current DFS iteration
		std::vector<bool> done_marks(num_vertices, false); //DFS of subtree complete, added to order

		for (uint32_t i = 0; i < num_vertices; i++) { //repeat DFS until entire graph has been searched
			if (!done_marks[i]) dfs_stack.push(std::make_pair(i, false)); //initialize stack with unmarked vertex
			while (!dfs_stack.empty()) { //DFS loop
				auto v = dfs_stack.top();
				dfs_stack.pop();
				auto children = adj_list[v.first];

				if (done_marks[v.first]) { //current vertex already marked as done: should never happen, but just remove and continue
					continue;
				} else if (v.second || children.empty()) { //current vertex's subtree has been processed (or doesn't exist): remove, mark as done, and add to order
					done_marks[v.first] = true;
					order.push_front(v.first);
				} else if (visit_marks[v.first]) { //current vertex has been visited in this iteration, but subtree is not done: cycle found, abort!
					return false;
				} else { //current vertex not visited in this iteration: keep in stack for later completion of the subtree, push children, mark as visited
					dfs_stack.push(std::make_pair(v.first, true));
					for each (auto w in children) dfs_stack.push(std::make_pair(w, false));
					visit_marks[v.first] = true;
				}
			}
		}

		order_out.assign(order.begin(), order.end());
		return true;
	}

	void Graph::topological_sssp(const std::vector<uint32_t>& topological_order, int64_t source, std::vector<int64_t>& distances, std::vector<int64_t>* predecessors) const {
		if (source < -1 || source >= num_vertices) throw std::runtime_error("topological_sssp: Invalid source index");

		//initialize distances with infinity, except for source (or roots)
		distances.clear();
		distances.resize(num_vertices, (int64_t)UINT32_MAX + 1);
		if (source < 0) {
			for each (auto root in indegree_zero) distances[root] = 0;
		} else {
			distances[source] = 0;
		}

		//initialize predecessors if given
		if (predecessors != nullptr) {
			predecessors->clear();
			predecessors->resize(num_vertices, -1);
			if (source >= 0) predecessors->operator[](source) = source;
		}

		bool start_found = source < 0; //whether the source has been found in the topological order yet (path finding starts there), immediately true for phantom root.
		for (uint32_t i = 0; i < num_vertices; i++) {
			auto vertex = topological_order[i];

			if (!start_found && vertex != source) continue; //search for start if necessary
			start_found = true;

			auto newdist = distances[vertex] + 1; //new proposed distance for children

			for each (auto child in adj_list[vertex]) { //update children where necessary (shorter path found)
				if (distances[child] > newdist) {
					distances[child] = newdist;
					if (predecessors != nullptr) predecessors->operator[](child) = vertex;
				}
			}
		}
	}

	void Graph::topological_sslp(const std::vector<uint32_t>& topological_order, int64_t source, std::vector<int64_t>& distances, std::vector<int64_t>* predecessors) const {
		if (source < -1 || source >= num_vertices) throw std::runtime_error("topological_sslp: Invalid source index");

		//initialize distances with infinity, except for source (or roots)
		distances.clear();
		distances.resize(num_vertices, (int64_t)UINT32_MAX + 1);
		if (source < 0) {
			for each (auto root in indegree_zero) distances[root] = 0;
		} else {
			distances[source] = 0;
		}

		//initialize predecessors if given
		if (predecessors != nullptr) {
			predecessors->clear();
			predecessors->resize(num_vertices, -1);
			if (source >= 0) predecessors->operator[](source) = source;
		}

		bool start_found = source < 0; //whether the source has been found in the topological order yet (path finding starts there), immediately true for phantom root.
		for (uint32_t i = 0; i < num_vertices; i++) {
			auto vertex = topological_order[i];

			if (!start_found && vertex != source) continue; //search for start if necessary
			start_found = true;

			auto newdist = distances[vertex] + 1; //new proposed distance for children
			if (newdist > UINT32_MAX) continue; //don't care about path if unreachable

			for each (auto child in adj_list[vertex]) { //update children where necessary (previously unreachable, or longer path found)
				if (distances[child] > UINT32_MAX || distances[child] < newdist) {
					distances[child] = newdist;
					if (predecessors != nullptr) predecessors->operator[](child) = vertex;
				}
			}
		}
	}

	void Graph::topological_sdsp(const std::vector<uint32_t>& topological_order, int64_t dest, std::vector<int64_t>& distances, std::vector<int64_t>* successors) const {
		if (dest < -1 || dest >= num_vertices) throw std::runtime_error("topological_sdsp: Invalid destination index");

		//initialize distances with infinity, except for destination (or leaves)
		distances.clear();
		distances.resize(num_vertices, (int64_t)UINT32_MAX + 1);
		if (dest < 0) {
			for each (auto leaf in outdegree_zero) distances[leaf] = 0;
		} else {
			distances[dest] = 0;
		}

		//initialize successors if given
		if (successors != nullptr) {
			successors->clear();
			successors->resize(num_vertices, -1);
			if (dest >= 0) successors->operator[](dest) = dest;
		}

		bool start_found = dest < 0; //whether the destination has been found in the reverse topological order yet (path finding starts there), immediately true for phantom sink.
		for (uint32_t i = 0; i < num_vertices; i++) {
			auto vertex = topological_order[num_vertices - i - 1];

			if (!start_found && vertex != dest) continue; //search for start if necessary
			start_found = true;

			auto newdist = distances[vertex] + 1; //new proposed distance for parents

			for each (auto parent in rev_adj_list[vertex]) { //update parents where necessary (shorter path found)
				if (distances[parent] > newdist) {
					distances[parent] = newdist;
					if (successors != nullptr) successors->operator[](parent) = vertex;
				}
			}
		}
	}

	void Graph::topological_sdlp(const std::vector<uint32_t>& topological_order, int64_t dest, std::vector<int64_t>& distances, std::vector<int64_t>* successors) const {
		if (dest < -1 || dest >= num_vertices) throw std::runtime_error("topological_sdlp: Invalid destination index");

		//initialize distances with infinity, except for destination (or leaves)
		distances.clear();
		distances.resize(num_vertices, (int64_t)UINT32_MAX + 1);
		if (dest < 0) {
			for each (auto leaf in outdegree_zero) distances[leaf] = 0;
		} else {
			distances[dest] = 0;
		}

		//initialize successors if given
		if (successors != nullptr) {
			successors->clear();
			successors->resize(num_vertices, -1);
			if (dest >= 0) successors->operator[](dest) = dest;
		}

		bool start_found = dest < 0; //whether the destination has been found in the reverse topological order yet (path finding starts there), immediately true for phantom sink.
		for (uint32_t i = 0; i < num_vertices; i++) {
			auto vertex = topological_order[num_vertices - i - 1];

			if (!start_found && vertex != dest) continue; //search for start if necessary
			start_found = true;

			auto newdist = distances[vertex] + 1; //new proposed distance for parents
			if (newdist > UINT32_MAX) continue; //don't care about path if unreachable

			for each (auto parent in rev_adj_list[vertex]) { //update parents where necessary (previously unreachable, or longer path found)
				if (distances[parent] > UINT32_MAX || distances[parent] < newdist) {
					distances[parent] = newdist;
					if (successors != nullptr) successors->operator[](parent) = vertex;
				}
			}
		}
	}

}
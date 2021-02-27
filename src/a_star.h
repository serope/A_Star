/*
 * a_star.h
 */
#pragma once
#include "graph.h"
#include "node.h"

/*
 * Perform A* search on a graph. If verbose, print details during computation.
 */
__global__ void a_star(graph_t* g, node_t* origin, node_t* goal, bool verbose);

/*
 * Find the index of n in g's node array and store it in i_ptr.
 */
__device__ static void __find_index(graph_t* g, node_t* n, int* i_ptr);

/*
 * Move the current node from g's unvisited set to its visited set.
 */
__device__ static void __move_to_unvisited(graph_t* graph, node_t* current_node, node_t* origin);

/*
 * Find the index of the current node's nth neighbor in g's nodes array. Store
 * it in i_ptr if found; -1 otherwise.
 */
__device__ static void __find_neighbor_index(graph_t* g, node_t* current_node, int n, int* i_ptr);

/*
 * Perform the relaxation step of A* search on the current node's nth neighbor.
 */
__device__ static void __relax(graph_t* g, node_t* current_node, int cn_index, int n, int neighbor_index, bool verbose);

/*
 * Check if a graph's unvisited set is empty; store the result in bool_ptr.
 */
__device__ static void __unvisited_set_is_empty(graph_t* g, bool* bool_ptr);

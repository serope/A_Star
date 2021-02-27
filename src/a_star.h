/*
 * a_star.h
 */
#pragma once
#include "graph.h"
#include "node.h"

/*
 * Perform A* search on a graph. If verbose, details are printed during
 * computation.
 */
__global__ void a_star(graph_t* g, node_t* origin, node_t* goal, bool verbose);

/*
 * Move the current node from g's unvisited set to its visited set.
 */
__device__ static void __move_to_unvisited(graph_t* graph, node_t* current_node, node_t* origin);

/*
 * Find the index of n in g's node array and store it in i_ptr.
 */
__device__ static void __find_index(graph_t* g, node_t* n, int* i_ptr);


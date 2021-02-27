/*
 * graph.h
 */
#pragma once
#include "node.h"

/*
 * The graph_t type represents an undirected graph.
 */
typedef struct graph_s {
	int 		node_count;
	node_t**	nodes;
	node_t**	visited_nodes;
	float*		distances;
	node_t**	parents;
} graph_t;

/*
 * Initialize a graph on the device.
 */
__global__ void graph_init(graph_t* g);

/*
 * Append a node to a graph.
 */
__global__ void graph_add(graph_t* g, node_t* n);

/*
 * Append a node to an empty graph.
 */
__device__ static void __graph_add_empty(graph_t* g, node_t* n);

/*
 * Destroy a graph.
 */
__global__ void graph_free(graph_t* g);

/*
 * Print a graph's details.
 */
__global__ void graph_print(graph_t* g, bool verbose);

/*
 * Print g's nodes array.
 */
__device__ static void __graph_print_arr_nodes(graph_t* g);

/*
 * Print g's visited_nodes array.
 */
__device__ static void __graph_print_arr_visited_nodes(graph_t* g);

/*
 * Print g's distances array.
 */
__device__ static void __graph_print_arr_distances(graph_t* g);

/*
 * Print g's parents array.
 */
__device__ static void __graph_print_arr_parents(graph_t* g);

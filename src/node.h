/*
 * node.h
 */
#pragma once
#include "graph.h"

/*
 * The node type represents a node in an undirected graph.
 */
typedef struct node_s {
	char            id;
	float           heuristic;
	int             neighbor_count;
	struct node_s** neighbors;
	int*            weights;
} node_t;

/*
 * Initialize a new node on the device.
 */
__global__ void node_init(node_t* n, char id, float heuristic);

/*
 * Add a neighbor to a node.
 */
__global__ void node_add_neighbor(node_t* n, node_t* neighbor, int weight);

/*
 * Add a neighbor to a node which currently doesn't have any neighbors.
 */
__device__ static void __node_add_1st_neighbor(node_t* n, node_t* neighbor, int weight);

/*
 * Destroy a node.
 */
__global__ void node_free(node_t* device_node);

/*
 * Print a node's details.
 */
__global__ void node_print(node_t* n);

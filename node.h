#pragma once
#include "graph.h"

typedef struct node_s {
	char				id;
	float				heuristic;
	int					neighbor_count;
	struct node_s**		neighbors;
	int*				weights;
} node_t;


__global__ void node_new(node_t* n, char id, float heuristic);
__global__ void node_add_neighbor(node_t* n, node_t* neighbor, int weight);
__global__ void node_free(node_t* device_node);
__global__ void node_print(node_t* n);

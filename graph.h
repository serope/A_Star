#pragma once
#include "node.h"

typedef struct graph_s {
	int 		node_count;
	node_t**	nodes;
	node_t**	visited_nodes;
	float*		distances;
	node_t**	parents;
} graph_t;


__global__ void graph_new(graph_t* g);
__global__ void graph_add(graph_t* g, node_t* n);
__global__ void graph_free(graph_t* g);
__global__ void graph_print(graph_t* g, bool verbose);

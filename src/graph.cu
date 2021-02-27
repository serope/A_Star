/*
 * graph.cu
 * 
 * Functions for the graph_t type are implemented here.
 */
#include <stdio.h>
#include "graph.h"
#include "node.h"

__global__ void graph_init(graph_t* g) {
	g->node_count    = 0;
	g->nodes         = NULL;
	g->visited_nodes = NULL;
	g->distances     = NULL;
	g->parents       = NULL;
}

__global__ void graph_add(graph_t* g, node_t* n) {
	// empty graph
	if (g->node_count == 0) {
		__graph_add_empty(g, n);
		return;
	}
	
	/*
	 * CUDA doesn't have a native equivalent to realloc(), so whenever an
	 * item is added to an array, an entire new array must be created...
	 */
	node_t** new_nodes         = (node_t**) malloc(sizeof(node_t*)*(g->node_count+1));
	node_t** new_visited_nodes = (node_t**) malloc(sizeof(node_t*)*(g->node_count+1));
	float* new_distances       = (float*) malloc(sizeof(int)*(g->node_count+1));
	node_t** new_parents       = (node_t**) malloc(sizeof(node_t*)*(g->node_count+1));

	// copy
	for (int x=0; x<(g->node_count); x++) {
		new_nodes[x]         = g->nodes[x];
		new_visited_nodes[x] = g->visited_nodes[x];
		new_distances[x]     = g->distances[x];
		new_parents[x]       = g->parents[x];
	}
	
	// add n
	new_nodes[g->node_count]         = n;
	new_visited_nodes[g->node_count] = NULL;
	new_distances[g->node_count]     = INFINITY;
	new_parents[g->node_count]       = NULL;
	g->node_count += 1;
	
	// destroy old
	free(g->nodes);
	free(g->visited_nodes);
	free(g->distances);
	free(g->parents);
	
	// reassign
	g->nodes         = new_nodes;
	g->visited_nodes = new_visited_nodes;
	g->distances     = new_distances;
	g->parents       = new_parents;
}

__device__ static void __graph_add_empty(graph_t* g, node_t* n) {
	g->nodes            = (node_t**) malloc(sizeof(node_t*));
	g->visited_nodes    = (node_t**) malloc(sizeof(node_t*));
	g->distances        = (float*) malloc(sizeof(int));
	g->parents          = (node_t**) malloc(sizeof(node_t*));
	g->nodes[0]         = n;
	g->visited_nodes[0] = NULL;
	g->distances[0]     = INFINITY;
	g->parents[0]       = NULL;
	g->node_count += 1;
}

__global__ void graph_free(graph_t* g) {
	free(g->nodes);
	free(g->visited_nodes);
	free(g->distances);
	free(g->parents);
	free(g);
}

__global__ void graph_print(graph_t* g, bool verbose) {
	if (verbose) {
		printf("graph               %p \n", g);
		printf("node_count          %d \n", g->node_count);
		printf("nodes[]             %p \n", g->nodes);
		printf("visited_nodes[]     %p \n", g->visited_nodes);
		printf("distances[]         %p \n", g->distances);
		printf("parents[]           %p \n", g->parents);
	}
	__graph_print_arr_nodes(g);
	__graph_print_arr_visited_nodes(g);
	__graph_print_arr_distances(g);
	__graph_print_arr_parents(g);
	printf("\n");
}

__device__ static void __graph_print_arr_nodes(graph_t* g) {
	if (!g->nodes) {
		printf("nodes               -\n");
		return;
	}
	
	printf("nodes \n");
	for (int x=0; x<g->node_count; x++) {
		if (g->nodes[x] != NULL)
			printf("                    %c (%p) \n", g->nodes[x]->id, g->nodes[x]);
		else
			printf("                    -\n");
	}
}

__device__ static void __graph_print_arr_visited_nodes(graph_t* g) {
	if (!g->visited_nodes) {
		printf("visited_nodes       -\n");
		return;
	}
	printf("visited_nodes \n");
	for (int x=0; x<g->node_count; x++) {
		if (g->visited_nodes[x] != NULL)
			printf("                    %c (%p) \n", g->visited_nodes[x]->id, g->visited_nodes[x]);
		else
			printf("                    -\n");
	}
}

__device__ static void __graph_print_arr_distances(graph_t* g) {
	if (!g->distances) {
		printf("distances           -\n");
		return;
	}
	printf("distances \n");
	for (int x=0; x<g->node_count; x++) {
		if (g->distances[x] != INFINITY)
			printf("                    %.3f \n", g->distances[x]);
		else
			printf("                    -\n");
	}
}

__device__ static void __graph_print_arr_parents(graph_t* g) {
	if (!g->parents) {
		printf("parents           -\n");
		return;
	}
	printf("parents \n");
	for (int x=0; x<g->node_count; x++) {
		if (g->parents[x] != NULL)
			printf("                    %c (%p) \n", g->parents[x]->id, g->parents[x]);
		else
			printf("                    -\n");
	}
}

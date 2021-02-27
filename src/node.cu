/*
 * node.cu
 * 
 * Functions for the node_t type are implemented here.
 */
#include "graph.h"
#include "node.h"
#include <stdio.h>

__global__ void node_init(node_t* n, char id, float heuristic) {
	n->id				= id;
	n->heuristic		= heuristic;
	n->neighbor_count	= 0;
	n->neighbors		= NULL;
	n->weights			= NULL;
}

__global__ void node_add_neighbor(node_t* n, node_t* neighbor, int weight) {
	// if no neighbors so far
	if (n->neighbor_count == 0) {
		__node_add_1st_neighbor(n, neighbor, weight);
		return;
	}
	
	/*
	 * CUDA doesn't have a native equivalent to realloc(), so whenever an
	 * item is added to an array, an entire new array must be created...
	 */
	node_t** new_neighbors = (node_t**) malloc(sizeof(node_t*)*(n->neighbor_count+1));
	int* new_weights       = (int*) malloc(sizeof(int)*(n->neighbor_count+1));

	// copy
	for (int l=0; l<(n->neighbor_count); l++) {
		new_neighbors[l] = n->neighbors[l];
		new_weights[l] = n->weights[l];
	}
	
	// add neighbor
	new_neighbors[n->neighbor_count] = neighbor;
	new_weights[n->neighbor_count] = weight;
	n->neighbor_count += 1;
	
	// delete old
	free(n->neighbors);
	free(n->weights);
	
	// assign new
	n->neighbors = new_neighbors;
	n->weights = new_weights;
	
}

__device__ static void __node_add_1st_neighbor(node_t* n, node_t* neighbor, int weight) {
	n->neighbors    = (node_t**) malloc(sizeof(node_t*));
	n->weights      = (int*) malloc(sizeof(int));
	n->neighbors[0] = neighbor;
	n->weights[0]   = weight;
	n->neighbor_count += 1;
}

__global__ void node_free(node_t* n) {
	free(n->neighbors);
	free(n->weights);
	free(n);
}

__global__ void node_print(node_t* n) {
	printf("node                %c (%p)\n", n->id, n);
	printf("heuristic           %d \n", n->heuristic);
	printf("neighbor_count      %d \n", n->neighbor_count);
	printf("neighbors[]         %p \n", n->neighbors);
	printf("weights[]           %p \n", n->weights);
	printf("neighbors           ");
	if (n->neighbor_count == 0)
		printf("-\nweights             -\n");
	else {
		for (int l=0; l<n->neighbor_count; l++) {
			if (l>0)
				printf("                    ");
			printf("%c (%p) \n", n->neighbors[l]->id, n->neighbors[l]);
		}

		printf("weights             ");
		for (int l=0; l<n->neighbor_count; l++) {
			if (l>0)
				printf("                    ");
			printf("%d \n", n->weights[l]);
		}
	}
	
	printf("\n\n");
}

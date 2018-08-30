#include "proto.h"

__global__ void initialize_node(struct node* n, char id, float heuristic) {
	n->id				= id;
	n->heuristic		= heuristic;
	n->neighbor_count	= 0;
	n->neighbors		= NULL;
	n->weights			= NULL;
}

__global__ void add_neighbor(struct node* n, struct node* neighbor, int weight) {
	/*******************************************************************
	 * If it's the first neighbor, attach it and return
	 ******************************************************************/
	if (n->neighbor_count==0) {
		n->neighbors = (array_of_node_ptrs) malloc(sizeof(struct node*));
		n->weights = (int*) malloc(sizeof(int));
		
		n->neighbors[0] = neighbor;
		n->weights[0] = weight;
		
		n->neighbor_count += 1;
		
		return;
	}
	
	/*******************************************************************
	 * Else...
	 * 
	 * CUDA doesn't have a native equivalent to realloc(), so whatever
	 * an item as added to an array, an entire new array must be
	 * created...
	 * 
	 * 1. Create new arrays
	 ******************************************************************/
	array_of_node_ptrs new_neighbors	= (array_of_node_ptrs) malloc(sizeof(struct node*)*(n->neighbor_count+1));
	int* new_weights					= (int*) malloc(sizeof(int)*(n->neighbor_count+1));

	/*******************************************************************
	 * 2. Populate them with the node's current values
	 ******************************************************************/
	for (int l=0; l<(n->neighbor_count); l++) {
		new_neighbors[l] = n->neighbors[l];
		new_weights[l] = n->weights[l];
	}
	
	new_neighbors[n->neighbor_count] = neighbor;
	new_weights[n->neighbor_count] = weight;
	
	n->neighbor_count += 1;
	
	/*******************************************************************
	 * 3. Delete the node's current arrays
	 ******************************************************************/
	free(n->neighbors);
	free(n->weights);
	
	/*******************************************************************
	 * 4. Assign the new arrays
	 ******************************************************************/
	n->neighbors = new_neighbors;
	n->weights = new_weights;
	
}

__global__ void free_node(struct node* n) {
	free(n->neighbors);
	free(n->weights);
	free(n);
}

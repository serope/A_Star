#include "graph.h"
#include "node.h"
#include <stdio.h>

/*
 * node_new()
 * 
 * Initializes the values of a node
 * Use cudaMalloc() to allocate the node first!
 */
__global__ void node_new(node_t* n, char id, float heuristic) {
	n->id				= id;
	n->heuristic		= heuristic;
	n->neighbor_count	= 0;
	n->neighbors		= NULL;
	n->weights			= NULL;
}

/*
 * node_add_neighbor()
 * 
 * Adds a neighboring node to a node with a directed, weighted edge
 * between them
 */
__global__ void node_add_neighbor(node_t* n, node_t* neighbor, int weight) {
	//If this is the 1st neighbor to be added to n, add it and return
	if (n->neighbor_count==0) {
		n->neighbors 		= (node_t**) malloc(sizeof(node_t*));
		n->weights			= (int*) malloc(sizeof(int));
		n->neighbors[0]		= neighbor;
		n->weights[0]		= weight;
		n->neighbor_count += 1;
		return;
	}
	
	/*
	 * CUDA doesn't have a native equivalent to realloc(), so whatever
	 * an item is added to an array, an entire new array must be
	 * created...
	 */
	node_t** new_neighbors	= (node_t**) malloc(sizeof(node_t*)*(n->neighbor_count+1));
	int* new_weights		= (int*) malloc(sizeof(int)*(n->neighbor_count+1));

	//Copy
	for (int l=0; l<(n->neighbor_count); l++) {
		new_neighbors[l] = n->neighbors[l];
		new_weights[l] = n->weights[l];
	}
	
	//Add
	new_neighbors[n->neighbor_count] = neighbor;
	new_weights[n->neighbor_count] = weight;
	n->neighbor_count += 1;
	
	//Delete old
	free(n->neighbors);
	free(n->weights);
	
	//Assign new
	n->neighbors = new_neighbors;
	n->weights = new_weights;
	
}


/*
 * node_free()
 * 
 * Destroys the given node
 */
__global__ void node_free(node_t* n) {
	free(n->neighbors);
	free(n->weights);
	free(n);
}



/*
 * node_print()
 * 
 * Prints details about the given node
 */
__global__ void node_print(node_t* n) {
	printf("node                %c (%p)\n", n->id, n);
	printf("heuristic           %d \n", n->heuristic);
	printf("neighbor_count      %d \n", n->neighbor_count);
	printf("neighbors[]         %p \n", n->neighbors);
	printf("weights[]           %p \n", n->weights);
	printf("neighbors           ");
	if (n->neighbor_count==0)
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

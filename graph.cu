#include "proto.h"

__global__ void initialize_graph(struct graph* g) {
	g->node_count		= 0;
	g->nodes			= NULL;
	g->visited_nodes	= NULL;
	g->distances		= NULL;
	g->parents			= NULL;
}

__global__ void add_to_graph(struct graph* g, struct node* n) {
	/*******************************************************************
	 * 0. If it's the first node, add it and return
	 ******************************************************************/
	if (g->node_count==0) {
		g->nodes			= (array_of_node_ptrs) malloc(sizeof(struct node*));
		g->visited_nodes	= (array_of_node_ptrs) malloc(sizeof(struct node*));
		g->distances		= (float*) malloc(sizeof(int));
		g->parents			= (array_of_node_ptrs) malloc(sizeof(struct node*));
		
		g->nodes[0]			= n;
		g->visited_nodes[0]	= NULL;
		g->distances[0]		= INFINITY;
		g->parents[0]		= NULL;
		
		g->node_count += 1;
		
		return;
	}
	
	/*******************************************************************
	 * 1. Create new arrays
	 ******************************************************************/
	array_of_node_ptrs new_nodes			= (array_of_node_ptrs) malloc(sizeof(struct node*)*(g->node_count+1));
	array_of_node_ptrs new_visited_nodes	= (array_of_node_ptrs) malloc(sizeof(struct node*)*(g->node_count+1));
	float* new_distances					= (float*) malloc(sizeof(int)*(g->node_count+1));
	array_of_node_ptrs new_parents			= (array_of_node_ptrs) malloc(sizeof(struct node*)*(g->node_count+1));

	/*******************************************************************
	 * 2. Populate them with the graph's current values
	 ******************************************************************/
	for (int x=0; x<(g->node_count); x++) {
		new_nodes[x]			= g->nodes[x];
		new_visited_nodes[x]	= g->visited_nodes[x];
		new_distances[x]		= g->distances[x];
		new_parents[x]			= g->parents[x];
	}
	
	new_nodes[g->node_count]			= n;
	new_visited_nodes[g->node_count]	= NULL;
	new_distances[g->node_count]		= INFINITY;
	new_parents[g->node_count]			= NULL;
	g->node_count += 1;
	
	/*******************************************************************
	 * 3. Delete the graph's current arrays
	 ******************************************************************/
	free(g->nodes);
	free(g->visited_nodes);
	free(g->distances);
	free(g->parents);
	
	/*******************************************************************
	 * 4. Assign the new arrays
	 ******************************************************************/
	g->nodes			= new_nodes;
	g->visited_nodes	= new_visited_nodes;
	g->distances		= new_distances;
	g->parents			= new_parents;
}


__global__ void free_graph(struct graph* g) {
	free(g->nodes);
	free(g->visited_nodes);
	free(g->distances);
	free(g->parents);
	free(g);
}

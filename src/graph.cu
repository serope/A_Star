#include <stdio.h>
#include "graph.h"
#include "node.h"


/*
 * graph_new()
 * 
 * Initializes the values of a graph
 * Use cudaMalloc() to allocate the graph first!
 */
__global__ void graph_new(graph_t* g) {
	g->node_count		= 0;
	g->nodes			= NULL;
	g->visited_nodes	= NULL;
	g->distances		= NULL;
	g->parents			= NULL;
}


/*
 * graph_add()
 * 
 * Adds a node to a graph
 */
__global__ void graph_add(graph_t* g, node_t* n) {
	//If it's the first node, add it and return
	if (g->node_count==0) {
		g->nodes			= (node_t**) malloc(sizeof(node_t*));
		g->visited_nodes	= (node_t**) malloc(sizeof(node_t*));
		g->distances		= (float*) malloc(sizeof(int));
		g->parents			= (node_t**) malloc(sizeof(node_t*));
		g->nodes[0]			= n;
		g->visited_nodes[0]	= NULL;
		g->distances[0]		= INFINITY;
		g->parents[0]		= NULL;
		g->node_count += 1;
		return;
	}
	
	/*
	 * CUDA doesn't have a native equivalent to realloc(), so whatever
	 * an item is added to an array, an entire new array must be
	 * created...
	 */
	node_t** new_nodes			= (node_t**) malloc(sizeof(node_t*)*(g->node_count+1));
	node_t** new_visited_nodes	= (node_t**) malloc(sizeof(node_t*)*(g->node_count+1));
	float* new_distances		= (float*) malloc(sizeof(int)*(g->node_count+1));
	node_t** new_parents		= (node_t**) malloc(sizeof(node_t*)*(g->node_count+1));

	//Copy
	for (int x=0; x<(g->node_count); x++) {
		new_nodes[x]			= g->nodes[x];
		new_visited_nodes[x]	= g->visited_nodes[x];
		new_distances[x]		= g->distances[x];
		new_parents[x]			= g->parents[x];
	}
	
	//Add
	new_nodes[g->node_count]			= n;
	new_visited_nodes[g->node_count]	= NULL;
	new_distances[g->node_count]		= INFINITY;
	new_parents[g->node_count]			= NULL;
	g->node_count += 1;
	
	//Delete old
	free(g->nodes);
	free(g->visited_nodes);
	free(g->distances);
	free(g->parents);
	
	//Assign new
	g->nodes			= new_nodes;
	g->visited_nodes	= new_visited_nodes;
	g->distances		= new_distances;
	g->parents			= new_parents;
}


/*
 * graph_free()
 * 
 * Destroys the given graph
 */
__global__ void graph_free(graph_t* g) {
	free(g->nodes);
	free(g->visited_nodes);
	free(g->distances);
	free(g->parents);
	free(g);
}



/*
 * graph_print()
 * 
 * Prints all details about the given graph
 */
__global__ void graph_print(graph_t* g, bool verbose) {
	//Pointers/addresses of interest
	if (verbose) {
		printf("graph               %p \n", g);
		printf("node_count          %d \n", g->node_count);
		printf("nodes[]             %p \n", g->nodes);
		printf("visited_nodes[]     %p \n", g->visited_nodes);
		printf("distances[]         %p \n", g->distances);
		printf("parents[]           %p \n", g->parents);
	}
	
	//nodes[] array
	if (g->nodes==NULL)
		printf("nodes               -\n");
	else {
		printf("nodes \n");
		for (int x=0; x<g->node_count; x++) {
			if (g->nodes[x]!=NULL)
				printf("                    %c (%p) \n", g->nodes[x]->id, g->nodes[x]);
			else
				printf("                    -\n");
		}
	}
	
	//visited_nodes[] array
	if (g->visited_nodes==NULL)
		printf("visited_nodes       -\n");
	else {
		printf("visited_nodes \n");
		for (int x=0; x<g->node_count; x++) {
			if (g->visited_nodes[x]!=NULL)
				printf("                    %c (%p) \n", g->visited_nodes[x]->id, g->visited_nodes[x]);
			else
				printf("                    -\n");
		}
	}
	
	//distances[] array
	if (g->distances==NULL)
		printf("distances           -\n");
	else {
		printf("distances \n");
		for (int x=0; x<g->node_count; x++) {
			if (g->distances[x]!=INFINITY)
				printf("                    %.3f \n", g->distances[x]);
			else
				printf("                    -\n");
		}
	}
	
	//parents[] array
	if (g->parents==NULL)
		printf("parents           -\n");
	else {
		printf("parents \n");
		for (int x=0; x<g->node_count; x++) {
			if (g->parents[x]!=NULL)
				printf("                    %c (%p) \n", g->parents[x]->id, g->parents[x]);
			else
				printf("                    -\n");
		}
	}
	
	printf("\n");
}

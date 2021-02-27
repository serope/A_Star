/*
 * a_star.cu
 * 
 * The A* search algorithm is implemented here.
 */
#include <stdio.h>
#include "graph.h"
#include "node.h"

__global__ void a_star(graph_t* g, node_t* origin, node_t* goal, bool verbose) {
	node_t* current_node = origin;
	
	// repeat until finished
	start:
	
	//Find index of current node
	int index = 0;
	while (g->nodes[index] != current_node)
		index += 1;
		
	//Transfer current node from unvisited set to visited set
	if (current_node==origin) {
		g->nodes[index]			= NULL;
		g->visited_nodes[index]	= current_node;
		g->distances[index]		= 0;
		g->parents[index]		= NULL;
	}
	
	else {
		g->nodes[index]			= NULL;
		g->visited_nodes[index]	= current_node;
	}
	
	//For each neighbor...
	for (int n=0; n < current_node->neighbor_count; n++) {
		//Find the neighbor's index
		int neighbor_index = 0;
		bool neighbor_already_analyzed = false;
		while (g->nodes[neighbor_index] != current_node->neighbors[n]) {
			neighbor_index += 1;
			if (neighbor_index==g->node_count) {
				neighbor_already_analyzed = true;
				break;
			}
		}
		
		//Relaxation
		if (!neighbor_already_analyzed) {
			float tentative_distance = g->distances[index] + current_node->weights[n] + current_node->heuristic;
			if (verbose) {
				printf("Current node:           \t%c (%p) \n", current_node->id, current_node);
				printf("Current neighbor:       \t%c (%p) \n", current_node->neighbors[n]->id, current_node->neighbors[n]);
				printf("Replace %.2f with %.2f? \t", g->distances[neighbor_index], tentative_distance);
			}
			
			bool replace = false;
			if (tentative_distance < g->distances[neighbor_index]) {
				replace = true;
				g->distances[neighbor_index] = tentative_distance;
				g->parents[neighbor_index] = current_node;
			}
			
			if (verbose) {
				if (replace)
					printf("Y \n\n");
				else
					printf("N \n\n");
			}
		}
	}
	
	//Stop if the unvisited set is empty
	bool empty = true;
	for (int n=0; n < g->node_count; n++) {
		if (g->nodes[n] != NULL) {
			empty = false;
			break;
		}
	}
	if (empty)
		return;
	
	//Repeat using the next unvisited node with the shortest distance
	float shortest = INFINITY;
	index = 0;
	for (int n=0; n < g->node_count; n++) {
		if (g->distances[n] <= shortest && g->nodes[n] != NULL) {
			shortest = g->distances[n];
			index = n;
		}
	}
	current_node = g->nodes[index];
	goto start;
}

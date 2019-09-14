/***********************************************************************
 * A* Search
 **********************************************************************/
#include <stdio.h>
#include "graph.h"
#include "node.h"

__global__ void a_star(graph_t* g, node_t* origin, node_t* goal, bool verbose);

int main() {
	//Declare nodes
	node_t* a;
	node_t* b;
	node_t* c;
	node_t* d;
	node_t* e;
	node_t* f;
	node_t* g;
	node_t* z;
	
	//Allocate nodes
	cudaMalloc((void**) &a, sizeof(node_t));
	cudaMalloc((void**) &b, sizeof(node_t));
	cudaMalloc((void**) &c, sizeof(node_t));
	cudaMalloc((void**) &d, sizeof(node_t));
	cudaMalloc((void**) &e, sizeof(node_t));
	cudaMalloc((void**) &f, sizeof(node_t));
	cudaMalloc((void**) &g, sizeof(node_t));
	cudaMalloc((void**) &z, sizeof(node_t));
	
	//A
	node_new<<<1,1>>>(a, 'A', 6.47);
	node_add_neighbor<<<1,1>>>(a, b, 4);
	node_add_neighbor<<<1,1>>>(a, c, 6);
	node_add_neighbor<<<1,1>>>(a, d, 1);
	
	//B
	node_new<<<1,1>>>(b, 'B', 5.12);
	node_add_neighbor<<<1,1>>>(b, a, 4);
	node_add_neighbor<<<1,1>>>(b, c, 8);
	node_add_neighbor<<<1,1>>>(b, e, 10);
	
	//C
	node_new<<<1,1>>>(c, 'C', 4.00);
	node_add_neighbor<<<1,1>>>(c, a, 6);
	node_add_neighbor<<<1,1>>>(c, b, 8);
	node_add_neighbor<<<1,1>>>(c, e, 1);
	node_add_neighbor<<<1,1>>>(c, f, 2); 
	
	//D
	node_new<<<1,1>>>(d, 'D', 4.75);
	node_add_neighbor<<<1,1>>>(d, a, 1);
	node_add_neighbor<<<1,1>>>(d, f, 2);
	node_add_neighbor<<<1,1>>>(d, g, 7);
	
	//E
	node_new<<<1,1>>>(e, 'E', 3.30);
	node_add_neighbor<<<1,1>>>(e, b, 10);
	node_add_neighbor<<<1,1>>>(e, c, 1);
	node_add_neighbor<<<1,1>>>(e, f, 5);
	node_add_neighbor<<<1,1>>>(e, z, 2);
	
	//F
	node_new<<<1,1>>>(f, 'F', 2.44);
	node_add_neighbor<<<1,1>>>(f, c, 2);
	node_add_neighbor<<<1,1>>>(f, d, 2);
	node_add_neighbor<<<1,1>>>(f, e, 5);
	node_add_neighbor<<<1,1>>>(f, g, 5);
	node_add_neighbor<<<1,1>>>(f, z, 12);
	
	//G
	node_new<<<1,1>>>(g, 'G', 4.13);
	node_add_neighbor<<<1,1>>>(g, d, 7);
	node_add_neighbor<<<1,1>>>(g, f, 5);
	node_add_neighbor<<<1,1>>>(g, z, 5);
	
	//Z
	node_new<<<1,1>>>(z, 'Z', 0.00);
	node_add_neighbor<<<1,1>>>(z, e, 2);
	node_add_neighbor<<<1,1>>>(z, f, 12);
	node_add_neighbor<<<1,1>>>(z, g, 5);
	
	//Print some node details
	/*
	node_print<<<1,1>>>(a);
	node_print<<<1,1>>>(b);
	node_print<<<1,1>>>(c);
	node_print<<<1,1>>>(d);
	*/
	
	//Create graph
	graph_t* gr;
	cudaMalloc((void**) &gr, sizeof(graph_t));
	graph_new<<<1,1>>>(gr);
	graph_add<<<1,1>>>(gr, a);
	graph_add<<<1,1>>>(gr, b);
	graph_add<<<1,1>>>(gr, c);
	graph_add<<<1,1>>>(gr, d);
	graph_add<<<1,1>>>(gr, e);
	graph_add<<<1,1>>>(gr, f);
	graph_add<<<1,1>>>(gr, g);
	graph_add<<<1,1>>>(gr, z);
	
	//A* search
	a_star<<<1,1>>>(gr, a, z, false);
	graph_print<<<1,1>>>(gr, false);
	cudaDeviceSynchronize();
	
	//End
	node_free<<<1,1>>>(a);
	node_free<<<1,1>>>(b);
	node_free<<<1,1>>>(c);
	node_free<<<1,1>>>(d);
	node_free<<<1,1>>>(e);
	node_free<<<1,1>>>(f);
	node_free<<<1,1>>>(g);
	node_free<<<1,1>>>(z);
	graph_free<<<1,1>>>(gr);
	return 0;
}






/*
 * a_star()
 * 
 * Performs the A* search algorithm.
 * 
 * g		A graph containing the start and end points
 * origin	The start point
 * goal		The end point
 * verbose	Pass 'true' to print details
 * 
 * Note: CUDA kernel functions can't be recursive, so goto is used
 *       instead.
 */
__global__ void a_star(graph_t* g, node_t* origin, node_t* goal, bool verbose) {
	//The current node is initially the origin
	node_t* current_node = origin;
	
	//Repeat until finished
	start:
	
	//Find index of current node
	int index = 0;
	while (g->nodes[index]!=current_node)
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
		if (g->nodes[n]!=NULL) {
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

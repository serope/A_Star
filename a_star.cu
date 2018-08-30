/***********************************************************************
 * 
 *   A* Search (Nvidia CUDA)
 * 
 *   https://serope.com/ai/a-star.html
 * 
 **********************************************************************/
#include <stdio.h>
#include "proto.h"
__global__ void a_star(struct graph* g, struct node* origin, struct node* goal);

int main() {
	//Declare nodes
	struct node* a;
	struct node* b;
	struct node* c;
	struct node* d;
	struct node* e;
	struct node* f;
	struct node* g;
	struct node* z;
	
	//Allocate nodes
	cudaMalloc((void**) &a, sizeof(struct node));
	cudaMalloc((void**) &b, sizeof(struct node));
	cudaMalloc((void**) &c, sizeof(struct node));
	cudaMalloc((void**) &d, sizeof(struct node));
	cudaMalloc((void**) &e, sizeof(struct node));
	cudaMalloc((void**) &f, sizeof(struct node));
	cudaMalloc((void**) &g, sizeof(struct node));
	cudaMalloc((void**) &z, sizeof(struct node));
	
	//A
	initialize_node<<<1,1>>>(a, 'A', 6.47);
	add_neighbor<<<1,1>>>(a, b, 4);
	add_neighbor<<<1,1>>>(a, c, 6);
	add_neighbor<<<1,1>>>(a, d, 1);
	
	//B
	initialize_node<<<1,1>>>(b, 'B', 5.12);
	add_neighbor<<<1,1>>>(b, a, 4);
	add_neighbor<<<1,1>>>(b, c, 8);
	add_neighbor<<<1,1>>>(b, e, 10);
	
	//C
	initialize_node<<<1,1>>>(c, 'C', 4.00);
	add_neighbor<<<1,1>>>(c, a, 6);
	add_neighbor<<<1,1>>>(c, b, 8);
	add_neighbor<<<1,1>>>(c, e, 1);
	add_neighbor<<<1,1>>>(c, f, 2); 
	
	//D
	initialize_node<<<1,1>>>(d, 'D', 4.75);
	add_neighbor<<<1,1>>>(d, a, 1);
	add_neighbor<<<1,1>>>(d, f, 2);
	add_neighbor<<<1,1>>>(d, g, 7);
	
	//E
	initialize_node<<<1,1>>>(e, 'E', 3.30);
	add_neighbor<<<1,1>>>(e, b, 10);
	add_neighbor<<<1,1>>>(e, c, 1);
	add_neighbor<<<1,1>>>(e, f, 5);
	add_neighbor<<<1,1>>>(e, z, 2);
	
	//F
	initialize_node<<<1,1>>>(f, 'F', 2.44);
	add_neighbor<<<1,1>>>(f, c, 2);
	add_neighbor<<<1,1>>>(f, d, 2);
	add_neighbor<<<1,1>>>(f, e, 5);
	add_neighbor<<<1,1>>>(f, g, 5);
	add_neighbor<<<1,1>>>(f, z, 12);
	
	//G
	initialize_node<<<1,1>>>(g, 'G', 4.13);
	add_neighbor<<<1,1>>>(g, d, 7);
	add_neighbor<<<1,1>>>(g, f, 5);
	add_neighbor<<<1,1>>>(g, z, 5);
	
	//Z
	initialize_node<<<1,1>>>(z, 'Z', 0.00);
	add_neighbor<<<1,1>>>(z, e, 2);
	add_neighbor<<<1,1>>>(z, f, 12);
	add_neighbor<<<1,1>>>(z, g, 5);
	
	//Print some node details
	print_node<<<1,1>>>(a);
	print_node<<<1,1>>>(b);
	print_node<<<1,1>>>(c);
	print_node<<<1,1>>>(d);
	
	//Draw the graph
	struct graph* gr;
	cudaMalloc((void**) &gr, sizeof(struct graph));
	initialize_graph<<<1,1>>>(gr);
	
	add_to_graph<<<1,1>>>(gr, a);
	add_to_graph<<<1,1>>>(gr, b);
	add_to_graph<<<1,1>>>(gr, c);
	add_to_graph<<<1,1>>>(gr, d);
	add_to_graph<<<1,1>>>(gr, e);
	add_to_graph<<<1,1>>>(gr, f);
	add_to_graph<<<1,1>>>(gr, g);
	add_to_graph<<<1,1>>>(gr, z);
	
	//A* search
	a_star<<<1,1>>>(gr, a, z);
	print_graph<<<1,1>>>(gr);
	cudaDeviceSynchronize();
	
	//Free everything
	free_node<<<1,1>>>(a);
	free_node<<<1,1>>>(b);
	free_node<<<1,1>>>(c);
	free_node<<<1,1>>>(d);
	free_node<<<1,1>>>(e);
	free_node<<<1,1>>>(f);
	free_node<<<1,1>>>(g);
	free_node<<<1,1>>>(z);
	free_graph<<<1,1>>>(gr);
	
	exit(EXIT_SUCCESS);
}



__global__ void a_star(struct graph* g, struct node* origin, struct node* goal) {
	/*******************************************************************
	 * 0. Repeat the function if the end condition wasn't met
	 ******************************************************************/
	struct node* current_node = origin;
	start:
	
	/*******************************************************************
	 * 1. Find the index that represents the current node in the
	 * graph's parallel arrays
	 ******************************************************************/
	int index = 0;
	while (g->nodes[index]!=current_node)
		index += 1;
		
	/*******************************************************************
	 * 2. Use the index to move the current node from the unvisited
	 * array to the visited array
	 ******************************************************************/
	//If the current node is the origin
	if (current_node==origin) {
		g->nodes[index]			= NULL;
		g->visited_nodes[index]	= current_node;
		g->distances[index]		= 0;
		g->parents[index]		= NULL;
	}
	
	//If the current node is NOT the origin
	else {
		g->nodes[index]			= NULL;
		g->visited_nodes[index]	= current_node;
	}
	
	/*******************************************************************
	 * 3. Perform the comparison step for each neighbor
	 ******************************************************************/
	for (int n=0; n < current_node->neighbor_count; n++) {
		/***************************************************************
		 * 3A. Find the index of this neighbor
		 **************************************************************/
		int neighbor_index = 0;
		bool neighbor_already_analyzed = false;
		while (g->nodes[neighbor_index] != current_node->neighbors[n]) {
			neighbor_index += 1;
			if (neighbor_index==g->node_count) {
				neighbor_already_analyzed = true;
				break;
			}
		}
		
		/***************************************************************
		 * 3B. Compare this neighbor's distance from the current node to
		 * the distance that has already been established. If the
		 * distance is shorter than the established one, then replace
		 * the established one.
		 **************************************************************/
		if (!neighbor_already_analyzed) {
			float tentative_distance = g->distances[index] + current_node->weights[n] + current_node->heuristic;
			printf("Current node:           \t%c (%p) \n", current_node->id, current_node);
			printf("Current neighbor:       \t%c (%p) \n", current_node->neighbors[n]->id, current_node->neighbors[n]);
			printf("Replace %.2f with %.2f? \t", g->distances[neighbor_index], tentative_distance);
			
			if (tentative_distance < g->distances[neighbor_index]) {
				g->distances[neighbor_index] = tentative_distance;
				g->parents[neighbor_index] = current_node;
				printf("Y \n\n");
			}
			else
				printf("N \n\n");
		}
	}
	
	/*******************************************************************
	 * 4. Set the exit condition (if the unvisited set is empty, then
	 * the algorithm is complete and the function may terminate).
	 ******************************************************************/
	bool unvisited_set_is_empty = true;
	for (int n=0; n < g->node_count; n++)
		if (g->nodes[n]!=NULL) {
			unvisited_set_is_empty = false;
			break;
		}
	if (unvisited_set_is_empty)
		return;
	
	/*******************************************************************
	 * 5. Otherwise, repeat the algorithm with the node that has the
	 * shortest distance.
	 ******************************************************************/
	float shortest_distance = INFINITY;
	index = 0;
	
	for (int n=0; n < g->node_count; n++)
		if (g->distances[n] <= shortest_distance && g->nodes[n] != NULL) {
			shortest_distance = g->distances[n];
			index = n;
		}
	
	current_node = g->nodes[index];
	goto start;
}

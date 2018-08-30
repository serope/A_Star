#pragma once
typedef struct node** array_of_node_ptrs;

/***********************************************************************
 * Structure types
 **********************************************************************/
struct node {
	char id;
	float heuristic;
	int neighbor_count;
	array_of_node_ptrs neighbors;
	int* weights;
};

struct graph {
	int node_count;
	
	array_of_node_ptrs nodes;
	array_of_node_ptrs visited_nodes;
	float* distances;
	array_of_node_ptrs parents;
};



/***********************************************************************
 * Debugging functions
 **********************************************************************/
__global__ void print_node(struct node* n);
__global__ void print_graph(struct graph* g);

/***********************************************************************
 * Node functions
 **********************************************************************/
__global__ void initialize_node(struct node* n, char id, float heuristic);
__global__ void add_neighbor(struct node* n, struct node* neighbor, int weight);
__global__ void free_node(struct node* device_node);

/***********************************************************************
 * Graph functions
 **********************************************************************/
__global__ void initialize_graph(struct graph* g);
__global__ void add_to_graph(struct graph* g, struct node* n);
__global__ void free_graph(struct graph* g);

/*
 * a_star
 * 
 * An implementation of the A* search algorithm to run on a Nvidia GPU.
 * This is just for practice/fun. It doesn't utilize parallelism at all.
 * 
 * https://en.wikipedia.org/wiki/A*_search_algorithm
 */
#include <stdio.h>
#include "graph.h"
#include "node.h"
#include "a_star.h"

int main() {
	// nodes
	node_t* a;
	node_t* b;
	node_t* c;
	node_t* d;
	node_t* e;
	node_t* f;
	node_t* g;
	node_t* z;
	
	cudaMalloc((void**) &a, sizeof(node_t));
	cudaMalloc((void**) &b, sizeof(node_t));
	cudaMalloc((void**) &c, sizeof(node_t));
	cudaMalloc((void**) &d, sizeof(node_t));
	cudaMalloc((void**) &e, sizeof(node_t));
	cudaMalloc((void**) &f, sizeof(node_t));
	cudaMalloc((void**) &g, sizeof(node_t));
	cudaMalloc((void**) &z, sizeof(node_t));
	
	// A
	node_init<<<1,1>>>(a, 'A', 6.47);
	node_add_neighbor<<<1,1>>>(a, b, 4);
	node_add_neighbor<<<1,1>>>(a, c, 6);
	node_add_neighbor<<<1,1>>>(a, d, 1);
	
	// B
	node_init<<<1,1>>>(b, 'B', 5.12);
	node_add_neighbor<<<1,1>>>(b, a, 4);
	node_add_neighbor<<<1,1>>>(b, c, 8);
	node_add_neighbor<<<1,1>>>(b, e, 10);
	
	// C
	node_init<<<1,1>>>(c, 'C', 4.00);
	node_add_neighbor<<<1,1>>>(c, a, 6);
	node_add_neighbor<<<1,1>>>(c, b, 8);
	node_add_neighbor<<<1,1>>>(c, e, 1);
	node_add_neighbor<<<1,1>>>(c, f, 2); 
	
	// D
	node_init<<<1,1>>>(d, 'D', 4.75);
	node_add_neighbor<<<1,1>>>(d, a, 1);
	node_add_neighbor<<<1,1>>>(d, f, 2);
	node_add_neighbor<<<1,1>>>(d, g, 7);
	
	// E
	node_init<<<1,1>>>(e, 'E', 3.30);
	node_add_neighbor<<<1,1>>>(e, b, 10);
	node_add_neighbor<<<1,1>>>(e, c, 1);
	node_add_neighbor<<<1,1>>>(e, f, 5);
	node_add_neighbor<<<1,1>>>(e, z, 2);
	
	// F
	node_init<<<1,1>>>(f, 'F', 2.44);
	node_add_neighbor<<<1,1>>>(f, c, 2);
	node_add_neighbor<<<1,1>>>(f, d, 2);
	node_add_neighbor<<<1,1>>>(f, e, 5);
	node_add_neighbor<<<1,1>>>(f, g, 5);
	node_add_neighbor<<<1,1>>>(f, z, 12);
	
	// G
	node_init<<<1,1>>>(g, 'G', 4.13);
	node_add_neighbor<<<1,1>>>(g, d, 7);
	node_add_neighbor<<<1,1>>>(g, f, 5);
	node_add_neighbor<<<1,1>>>(g, z, 5);
	
	// Z
	node_init<<<1,1>>>(z, 'Z', 0.00);
	node_add_neighbor<<<1,1>>>(z, e, 2);
	node_add_neighbor<<<1,1>>>(z, f, 12);
	node_add_neighbor<<<1,1>>>(z, g, 5);
	
	// graph
	graph_t* gr;
	cudaMalloc((void**) &gr, sizeof(graph_t));
	graph_init<<<1,1>>>(gr);
	graph_add<<<1,1>>>(gr, a);
	graph_add<<<1,1>>>(gr, b);
	graph_add<<<1,1>>>(gr, c);
	graph_add<<<1,1>>>(gr, d);
	graph_add<<<1,1>>>(gr, e);
	graph_add<<<1,1>>>(gr, f);
	graph_add<<<1,1>>>(gr, g);
	graph_add<<<1,1>>>(gr, z);
	
	// A* search
	a_star<<<1,1>>>(gr, a, z, false);
	graph_print<<<1,1>>>(gr, false);
	cudaDeviceSynchronize();
	
	// end
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

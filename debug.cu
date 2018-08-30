#include <stdio.h>
#include "proto.h"

__global__ void print_node(struct node* n) {
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

__global__ void print_graph(struct graph* g) {
	//Print the graph's addresses
	printf("graph               %p \n", g);
	printf("node_count          %d \n", g->node_count);
	printf("nodes[]             %p \n", g->nodes);
	printf("visited_nodes[]     %p \n", g->visited_nodes);
	printf("distances[]         %p \n", g->distances);
	printf("parents[]           %p \n", g->parents);
	
	//Print the nodes[] array
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
	
	//Print the visited_nodes[] array
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
	
	//Print the distances[] array
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
	
	//Print the parents[] array
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

/*
 * a_star.h
 */
#pragma once
#include "graph.h"
#include "node.h"

/*
 * Perform A* search on a graph. If verbose, details are printed during
 * computation.
 */
__global__ void a_star(graph_t* g, node_t* origin, node_t* goal, bool verbose);

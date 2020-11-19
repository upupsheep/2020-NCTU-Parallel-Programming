#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
#define BOTTOMUP_NOT_VISITED_MARKER 0
#define THRESHOLD 10000000

void vertex_set_clear(vertex_set *list) {
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count) {
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances) {
    int local_count = 0;
#pragma omp parallel
#pragma omp for reduction(+ \
                          : local_count)
    for (int i = 0; i < frontier->count; i++) {
        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER) {
                distances[outgoing] = distances[node] + 1;
                // int index = new_frontier->count++;
                // new_frontier->vertices[index] = outgoing;
                local_count++;
                new_frontier->vertices[local_count] = outgoing;
            }
        }
    }
    new_frontier->count = local_count;
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol) {
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bottom_up_step(
    Graph g,
    vertex_set *frontier,
    int *distances,
    int iteration) {
    int local_count = 0;
// #pragma omp parallel num_threads(NUM_THREADS) private(local_count)
// #pragma omp parallel private(local_count)
#pragma omp parallel
    {
// local_count = 0;
#pragma omp for reduction(+ \
                          : local_count)
        for (int i = 0; i < g->num_nodes; i++) {
            if (frontier->vertices[i] == BOTTOMUP_NOT_VISITED_MARKER) {
                int start_edge = g->incoming_starts[i];
                int end_edge = (i == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[i + 1];
                for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                    int incoming = g->incoming_edges[neighbor];
                    // if(__sync_bool_compare_and_swap(&frontier->vertices[incoming], iteration, distances[node] + 1))
                    if (frontier->vertices[incoming] == iteration) {
                        distances[i] = distances[incoming] + 1;
                        // frontier->count++;
                        local_count++;
                        frontier->vertices[i] = iteration + 1;
                        break;
                    }
                }
            }
        }
    }
    frontier->count = local_count;
}

void bfs_bottom_up(Graph graph, solution *sol) {
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    vertex_set list1;

    vertex_set_init(&list1, graph->num_nodes);

    int iteration = 1;

    vertex_set *frontier = &list1;

    // setup frontier with the root node
    // just like put the root into queue
    frontier->vertices[frontier->count++] = 1;

    // set the root distance with 0
    sol->distances[ROOT_NODE_ID] = 0;

    // just like pop the queue
    while (frontier->count != 0) {
        frontier->count = 0;
        double start_time = CycleTimer::currentSeconds();

        bottom_up_step(graph, frontier, sol->distances, iteration);

        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);

        iteration++;
    }
}

void bfs_hybrid(Graph graph, solution *sol) {
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    vertex_set list1;

    vertex_set_init(&list1, graph->num_nodes);

    int iteration = 1;

    vertex_set *frontier = &list1;

    // setup frontier with the root node
    // just like put the root into queue
    memset(frontier->vertices, 0, sizeof(int) * graph->num_nodes);

    frontier->vertices[frontier->count++] = 1;

    // set the root distance with 0
    sol->distances[ROOT_NODE_ID] = 0;

    // just like pop the queue
    while (frontier->count != 0) {
        if (frontier->count >= THRESHOLD) {
            frontier->count = 0;
            bottom_up_step(graph, frontier, sol->distances, iteration);
        } else {
            frontier->count = 0;
            top_down_step(graph, frontier, sol->distances, iteration);
        }

        // double end_time = CycleTimer::currentSeconds();
        // printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);

        iteration++;
    }
}

#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <vector>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

using namespace std;

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence) {
    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    int numNodes = num_nodes(g);
    double equal_prob = 1.0 / numNodes;
    /*
    for (int i = 0; i < numNodes; ++i) {
        solution[i] = equal_prob;
    }
    */

    /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */

    double *score_old = (double *)malloc(sizeof(double) * numNodes);
    bool converged = false;
    double score_new, global_diff;
    int i, v;

#pragma omp parallel for
    // initialization: see example code above
    for (int i = 0; i < numNodes; ++i) {
        score_old[i] = equal_prob;
    }

    // graph index with no outgoing edges
    vector<int> no_outgoing;
#pragma omp for nowait
    for (i = 0; i < numNodes; i++) {
        if (outgoing_size(g, i) == 0) {
            no_outgoing.push_back(i);
        }
    }

    while (!converged) {
        global_diff = 0.0;

        // compute score_new[vi] for all nodes vi:
        for (i = 0; i < numNodes; i++) {
            score_new = 0.0;
            const Vertex *in_begin = incoming_begin(g, i);
            const Vertex *in_end = incoming_end(g, i);

#pragma omp parallel for reduction(+ \
                                   : score_new)
            // score_new[vi] = sum over all nodes vj reachable from incoming edges
            // { score_old[vj] / number of edges leaving vj  }
            for (const Vertex *vj = in_begin; vj != in_end; vj++) {
                score_new += score_old[*vj] / outgoing_size(g, *vj);
            }

            // score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;
            score_new = (damping * score_new) + (1.0 - damping) / numNodes;

#pragma omp parallel for reduction(+ \
                                   : score_new)
            // score_new[vi] += sum over all nodes v in graph with no outgoing edges
            // { damping * score_old[v] / numNodes }
            for (v = 0; v < no_outgoing.size(); v++) {
                score_new += damping * score_old[v] / numNodes;
            }

            solution[i] = score_new;

            // global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
            global_diff += abs(score_new - score_old[i]);
        }

        // converged = (global_diff < convergence)
        converged = (global_diff < convergence);
        memcpy(score_old, solution, sizeof(double) * numNodes);
    }

    free(score_old);
    no_outgoing.clear();
}

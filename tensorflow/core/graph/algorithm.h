<<<<<<< HEAD
=======
/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

>>>>>>> tensorflow/master
#ifndef TENSORFLOW_GRAPH_ALGORITHM_H_
#define TENSORFLOW_GRAPH_ALGORITHM_H_

#include <functional>
#include <unordered_set>
<<<<<<< HEAD
=======
#include <vector>
>>>>>>> tensorflow/master

#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

// Perform a depth-first-search on g starting at the source node.
// If enter is not empty, calls enter(n) before visiting any children of n.
// If leave is not empty, calls leave(n) after visiting all children of n.
extern void DFS(const Graph& g, std::function<void(Node*)> enter,
                std::function<void(Node*)> leave);

<<<<<<< HEAD
// Stores in *order the post-order numbering of all nodes
// in graph found via a depth first search starting at the source node.
//
// Note that this is equivalent to topological sorting when the
=======
// Perform a reverse depth-first-search on g starting at the sink node.
// If enter is not empty, calls enter(n) before visiting any parents of n.
// If leave is not empty, calls leave(n) after visiting all parents of n.
extern void ReverseDFS(const Graph& g, std::function<void(Node*)> enter,
                       std::function<void(Node*)> leave);

// Stores in *order the post-order numbering of all nodes
// in graph found via a depth first search starting at the source node.
//
// Note that this is equivalent to reverse topological sorting when the
>>>>>>> tensorflow/master
// graph does not have cycles.
//
// REQUIRES: order is not NULL.
void GetPostOrder(const Graph& g, std::vector<Node*>* order);

// Stores in *order the reverse post-order numbering of all nodes
void GetReversePostOrder(const Graph& g, std::vector<Node*>* order);

// Prune nodes in "g" that are not in some path from the source node
// to any node in 'nodes'.
void PruneForReverseReachability(Graph* g,
                                 const std::unordered_set<const Node*>& nodes);

// Connect all nodes with no incoming edges to source.
// Connect all nodes with no outgoing edges to sink.
<<<<<<< HEAD
void FixupSourceAndSinkEdges(Graph* g);
=======
//
// Returns true if and only if 'g' is mutated.
bool FixupSourceAndSinkEdges(Graph* g);
>>>>>>> tensorflow/master

}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_ALGORITHM_H_

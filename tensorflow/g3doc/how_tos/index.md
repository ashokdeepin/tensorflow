<<<<<<< HEAD
# Overview
=======
# How-Tos
>>>>>>> tensorflow/master


## Variables: Creation, Initializing, Saving, and Restoring

TensorFlow Variables are in-memory buffers containing tensors.  Learn how to
use them to hold and update model parameters during training.

<<<<<<< HEAD
[View Tutorial](../how_tos/variables/index.md)
=======
[View Tutorial](variables/index.md)
>>>>>>> tensorflow/master


## TensorFlow Mechanics 101

A step-by-step walk through of the details of using TensorFlow infrastructure
to train models at scale, using MNIST handwritten digit recognition as a toy
example.

[View Tutorial](../tutorials/mnist/tf/index.md)


## TensorBoard: Visualizing Learning

TensorBoard is a useful tool for visualizing the training and evaluation of
your model(s).  This tutorial describes how to build and run TensorBoard as well
as how to add Summary ops to automatically output data to the Events files that
TensorBoard uses for display.

<<<<<<< HEAD
[View Tutorial](../how_tos/summaries_and_tensorboard/index.md)
=======
[View Tutorial](summaries_and_tensorboard/index.md)
>>>>>>> tensorflow/master


## TensorBoard: Graph Visualization

This tutorial describes how to use the graph visualizer in TensorBoard to help
you understand the dataflow graph and debug it.

<<<<<<< HEAD
[View Tutorial](../how_tos/graph_viz/index.md)
=======
[View Tutorial](graph_viz/index.md)
>>>>>>> tensorflow/master


## Reading Data

This tutorial describes the three main methods of getting data into your
TensorFlow program: Feeding, Reading and Preloading.

<<<<<<< HEAD
[View Tutorial](../how_tos/reading_data/index.md)
=======
[View Tutorial](reading_data/index.md)
>>>>>>> tensorflow/master


## Threading and Queues

This tutorial describes the various constructs implemented by TensorFlow
to facilitate asynchronous and concurrent training.

<<<<<<< HEAD
[View Tutorial](../how_tos/threading_and_queues/index.md)
=======
[View Tutorial](threading_and_queues/index.md)
>>>>>>> tensorflow/master


## Adding a New Op

TensorFlow already has a large suite of node operations from which you can
compose in your graph, but here are the details of how to add you own custom Op.

<<<<<<< HEAD
[View Tutorial](../how_tos/adding_an_op/index.md)
=======
[View Tutorial](adding_an_op/index.md)


## Writing Documentation

TensorFlow's documentation is largely generated from its source code. Here is an
introduction to the formats we use, a style guide, and instructions on how to
build updated documentation from the source.

[View Tutorial](documentation/index.md)
>>>>>>> tensorflow/master


## Custom Data Readers

If you have a sizable custom data set, you may want to consider extending
TensorFlow to read your data directly in it's native format.  Here's how.

<<<<<<< HEAD
[View Tutorial](../how_tos/new_data_formats/index.md)
=======
[View Tutorial](new_data_formats/index.md)
>>>>>>> tensorflow/master


## Using GPUs

This tutorial describes how to construct and execute models on GPU(s).

<<<<<<< HEAD
[View Tutorial](../how_tos/using_gpu/index.md)
=======
[View Tutorial](using_gpu/index.md)
>>>>>>> tensorflow/master


## Sharing Variables

When deploying large models on multiple GPUs, or when unrolling complex LSTMs
or RNNs, it is often necessary to access the same Variable objects from
different locations in the model construction code.

The "Variable Scope" mechanism is designed to facilitate that.

<<<<<<< HEAD
[View Tutorial](../how_tos/variable_scope/index.md)

<div class='sections-order' style="display: none;">
<!--
<!-- variables/index.md -->
<!-- ../tutorials/mnist/tf/index.md -->
<!-- summaries_and_tensorboard/index.md -->
<!-- graph_viz/index.md -->
<!-- reading_data/index.md -->
<!-- threading_and_queues/index.md -->
<!-- adding_an_op/index.md -->
<!-- new_data_formats/index.md -->
<!-- using_gpu/index.md -->
<!-- variable_scope/index.md -->
-->
</div>

=======
[View Tutorial](variable_scope/index.md)

## A Tool Developer's Guide to TensorFlow Model Files

If you're developing a tool to load, analyze, or manipulate TensorFlow model
files, it's useful to understand a bit about the format in which they're stored.
This guide covers the details of the saved model format.

[View Tutorial](../how_tos/tool_developers/index.md)

## How to Retrain Inception using Transfer Learning

Training a full object recognition model like Inception takes a long time and a
lot of images. This example shows how to use the technique of transfer learning
to retrain just the final layer of a fully-trained model to recognize new
categories of objects, which is a lot faster and easier than completely
retraining a new model.

[View Tutorial](../how_tos/image_retraining/index.md)
>>>>>>> tensorflow/master

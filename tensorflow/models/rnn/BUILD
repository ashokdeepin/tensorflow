# Description:
# Example RNN models, including language models and sequence-to-sequence models.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

<<<<<<< HEAD
load("/tensorflow/tensorflow", "cuda_py_tests")

=======
>>>>>>> tensorflow/master
py_library(
    name = "linear",
    srcs = [
        "linear.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

<<<<<<< HEAD
py_test(
    name = "linear_test",
    size = "small",
    srcs = ["linear_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":linear",
        "//tensorflow:tensorflow_py",
    ],
)

=======
>>>>>>> tensorflow/master
py_library(
    name = "rnn_cell",
    srcs = [
        "rnn_cell.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":linear",
        "//tensorflow:tensorflow_py",
    ],
)

<<<<<<< HEAD
py_test(
    name = "rnn_cell_test",
    size = "small",
    srcs = ["rnn_cell_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":rnn_cell",
        "//tensorflow:tensorflow_py",
    ],
)

=======
>>>>>>> tensorflow/master
py_library(
    name = "package",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":rnn",
        ":rnn_cell",
        ":seq2seq",
    ],
)

py_library(
    name = "rnn",
    srcs = [
        "rnn.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":rnn_cell",
        "//tensorflow:tensorflow_py",
    ],
)

<<<<<<< HEAD
cuda_py_tests(
    name = "rnn_tests",
    srcs = [
        "rnn_test.py",
    ],
    additional_deps = [
        ":rnn",
    ],
)

=======
>>>>>>> tensorflow/master
py_library(
    name = "seq2seq",
    srcs = [
        "seq2seq.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":rnn",
        "//tensorflow:tensorflow_py",
    ],
)

<<<<<<< HEAD
py_test(
    name = "seq2seq_test",
    srcs = [
        "seq2seq_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":seq2seq",
        "//tensorflow:tensorflow_py",
    ],
)

=======
>>>>>>> tensorflow/master
filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

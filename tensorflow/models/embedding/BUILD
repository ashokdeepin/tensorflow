# Description:
# TensorFlow model for word2vec

<<<<<<< HEAD
=======
package(default_visibility = ["//tensorflow:internal"])

>>>>>>> tensorflow/master
licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

<<<<<<< HEAD
load("/tensorflow/tensorflow", "tf_gen_op_wrapper_py")
=======
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

py_library(
    name = "package",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":gen_word2vec",
        ":word2vec",
        ":word2vec_optimized",
    ],
)
>>>>>>> tensorflow/master

py_binary(
    name = "word2vec",
    srcs = [
        "word2vec.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_word2vec",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
    ],
)

py_binary(
    name = "word2vec_optimized",
    srcs = [
        "word2vec_optimized.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_word2vec",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
    ],
)

py_test(
    name = "word2vec_test",
    size = "small",
    srcs = ["word2vec_test.py"],
    srcs_version = "PY2AND3",
<<<<<<< HEAD
=======
    tags = [
        "notsan",  # b/25864127
    ],
>>>>>>> tensorflow/master
    deps = [
        ":word2vec",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "word2vec_optimized_test",
    size = "small",
    srcs = ["word2vec_optimized_test.py"],
    srcs_version = "PY2AND3",
<<<<<<< HEAD
=======
    tags = [
        "notsan",
    ],
>>>>>>> tensorflow/master
    deps = [
        ":word2vec_optimized",
        "//tensorflow:tensorflow_py",
    ],
)

cc_library(
    name = "word2vec_ops",
    srcs = [
        "word2vec_ops.cc",
    ],
<<<<<<< HEAD
=======
    linkstatic = 1,
>>>>>>> tensorflow/master
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "word2vec_kernels",
    srcs = [
        "word2vec_kernels.cc",
    ],
<<<<<<< HEAD
    visibility = ["//tensorflow:internal"],
    deps = [
=======
    linkstatic = 1,
    visibility = ["//tensorflow:internal"],
    deps = [
        ":word2vec_ops",
>>>>>>> tensorflow/master
        "//tensorflow/core",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "gen_word2vec",
    out = "gen_word2vec.py",
    deps = [":word2vec_ops"],
)

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

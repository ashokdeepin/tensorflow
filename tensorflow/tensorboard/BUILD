# Description:
# TensorBoard, a dashboard for investigating TensorFlow

package(default_visibility = ["//tensorflow:internal"])

<<<<<<< HEAD
filegroup(
    name = "tensorboard_frontend",
=======
licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

filegroup(
    name = "frontend",
>>>>>>> tensorflow/master
    srcs = [
        "dist/index.html",
        "dist/tf-tensorboard.html",
        "//tensorflow/tensorboard/bower:bower",
<<<<<<< HEAD
    ] + glob(["lib/**/*"]),
)

py_library(
    name = "tensorboard_handler",
    srcs = ["tensorboard_handler.py"],
    deps = [
        ":float_wrapper",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "float_wrapper",
    srcs = ["float_wrapper.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "float_wrapper_test",
    size = "small",
    srcs = ["float_wrapper_test.py"],
    deps = [
        ":float_wrapper",
        "//tensorflow/python:platform_test",
    ],
    srcs_version = "PY2AND3",
)

py_binary(
    name = "tensorboard",
    srcs = ["tensorboard.py"],
    data = [":tensorboard_frontend"],
    deps = [
        ":tensorboard_handler",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
    ],
    srcs_version = "PY2AND3",
=======
        "TAG",
    ] + glob(["lib/**/*"]),
)

py_binary(
    name = "tensorboard",
    srcs = ["tensorboard.py"],
    data = [":frontend"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:platform",
        "//tensorflow/tensorboard/backend:server",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "**/node_modules/**",
            "**/typings/**",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
>>>>>>> tensorflow/master
)

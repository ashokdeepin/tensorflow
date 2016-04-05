licenses(["restricted"])

<<<<<<< HEAD
load("/tensorflow/tensorflow", "if_cuda")
=======
load("//tensorflow:tensorflow.bzl", "if_cuda")
>>>>>>> tensorflow/master

cc_library(
    name = "stream_executor",
    srcs = glob(
        [
            "*.cc",
            "lib/*.cc",
<<<<<<< HEAD
=======
            "platform/default/*.cc",
>>>>>>> tensorflow/master
        ],
        exclude = [
            "**/*_test.cc",
        ],
    ) + if_cuda(
        glob([
            "cuda/*.cc",
        ]),
    ),
    hdrs = glob([
        "*.h",
<<<<<<< HEAD
        "lib/*.h",
=======
        "cuda/*.h",
        "lib/*.h",
        "lib/gtl/*.h",
>>>>>>> tensorflow/master
        "platform/**/*.h",
    ]),
    data = [
        "//tensorflow/core:cuda",
        "//third_party/gpus/cuda:cublas",
        "//third_party/gpus/cuda:cudnn",
<<<<<<< HEAD
=======
        "//third_party/gpus/cuda:cufft",
>>>>>>> tensorflow/master
    ],
    linkopts = [
        "-ldl",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//third_party/gpus/cuda:cuda_headers",
    ],
    alwayslink = 1,
)

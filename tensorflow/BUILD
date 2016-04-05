# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.

package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
    "ACKNOWLEDGMENTS",
])

# open source marker; do not remove

<<<<<<< HEAD
=======
# Config setting for determining if we are building for Android.
config_setting(
    name = "android",
    values = {
        "crosstool_top": "//external:android/crosstool",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

>>>>>>> tensorflow/master
package_group(
    name = "internal",
    packages = ["//tensorflow/..."],
)

sh_binary(
    name = "swig",
    srcs = ["tools/swig/swig.sh"],
    data = glob(["tools/swig/**"]),
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "g3doc/sitemap.md",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

py_library(
    name = "tensorflow_py",
    srcs = ["__init__.py"],
<<<<<<< HEAD
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/python"],
)
=======
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/python"],
)

# -------------------------------------------
# New rules should be added above this target.
# -------------------------------------------
cc_binary(
    name = "libtensorflow.so",
    linkshared = 1,
    deps = [
        "//tensorflow/core:tensorflow",
    ],
)

cc_binary(
    name = "libtensorflow_cc.so",
    linkshared = 1,
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:tensorflow",
    ],
)

>>>>>>> tensorflow/master

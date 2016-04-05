# Description:
#   Tensorflow camera demo app for Android.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

<<<<<<< HEAD
exports_files(["LICENSE"])

cc_library(
    name = "tensorflow_native_libs",
    srcs = glob(["jni/**/*.cc"]),
    hdrs = glob(["jni/**/*.h"]),
    copts = [
        "-std=c++11",
        "-mfpu=neon",
    ],
    linkopts = ["-llog -landroid -lm -ljnigraphics"],
=======
load("//tensorflow:tensorflow.bzl", "tf_copts")

exports_files(["LICENSE"])

cc_binary(
    name = "libtensorflow_demo.so",
    srcs = glob([
        "jni/**/*.cc",
        "jni/**/*.h",
    ]) + [":libpthread.so"],
    copts = tf_copts(),
    linkopts = [
        "-landroid",
        "-ljnigraphics",
        "-llog",
        "-lm",
        "-z defs",
        "-s",
        "-Wl,--icf=all",  # Identical Code Folding
        "-Wl,--exclude-libs,ALL",  # Exclude syms in all libs from auto export
    ],
    linkshared = 1,
    linkstatic = 1,
>>>>>>> tensorflow/master
    tags = [
        "manual",
        "notap",
    ],
<<<<<<< HEAD
    deps = [
        ":dummy_pthread",
        "//tensorflow/core:android_tensorflow_lib",
    ],
=======
    deps = ["//tensorflow/core:android_tensorflow_lib"],
>>>>>>> tensorflow/master
)

# This library only exists as a workaround to satisfy dependencies
# that declare -lpthread in their linkopts. Although Android supports
# pthreads, it does not provide it as a separate library.
<<<<<<< HEAD
cc_library(
    name = "dummy_pthread",
    srcs = ["jni/libpthread.so"],
=======
cc_binary(
    name = "libpthread.so",
    srcs = [],
    linkopts = ["-shared"],
    tags = [
        "manual",
        "notap",
    ],
)

cc_library(
    name = "tensorflow_native_libs",
    srcs = [
        ":libpthread.so",
        ":libtensorflow_demo.so",
    ],
    tags = [
        "manual",
        "notap",
    ],
>>>>>>> tensorflow/master
)

android_binary(
    name = "tensorflow_demo",
    srcs = glob([
        "src/**/*.java",
    ]),
    assets = glob(["assets/**"]),
    assets_dir = "assets",
    custom_package = "org.tensorflow.demo",
    inline_constants = 1,
<<<<<<< HEAD
    legacy_native_support = 0,
=======
>>>>>>> tensorflow/master
    manifest = "AndroidManifest.xml",
    resource_files = glob(["res/**"]),
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":tensorflow_native_libs",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "bin/**",
            "gen/**",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
<<<<<<< HEAD
=======

filegroup(
    name = "java_files",
    srcs = glob(["src/**/*.java"]),
)

filegroup(
    name = "jni_files",
    srcs = glob([
        "jni/**/*.cc",
        "jni/**/*.h",
    ]),
)

filegroup(
    name = "resource_files",
    srcs = glob(["res/**"]),
)

exports_files(["AndroidManifest.xml"])
>>>>>>> tensorflow/master

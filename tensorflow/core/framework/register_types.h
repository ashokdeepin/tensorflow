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
#ifndef TENSORFLOW_FRAMEWORK_REGISTER_TYPES_H_
#define TENSORFLOW_FRAMEWORK_REGISTER_TYPES_H_
// This file is used by cuda code and must remain compilable by nvcc.

<<<<<<< HEAD
#include "tensorflow/core/platform/port.h"
=======
#include "tensorflow/core/platform/types.h"
>>>>>>> tensorflow/master

// Macros to apply another macro to lists of supported types.  If you change
// the lists of types, please also update the list in types.cc.
//
// See example uses of these macros in core/ops.
//
//
// Each of these TF_CALL_XXX_TYPES(m) macros invokes the macro "m" multiple
// times by passing each invocation a data type supported by TensorFlow.
//
// The different variations pass different subsets of the types.
// TF_CALL_ALL_TYPES(m) applied "m" to all types supported by TensorFlow.
// The set of types depends on the compilation platform.
//.
// This can be used to register a different template instantiation of
// an OpKernel for different signatures, e.g.:
/*
   #define REGISTER_PARTITION(type)                                  \
     REGISTER_TF_OP_KERNEL("partition", DEVICE_CPU, #type ", int32", \
                           PartitionOp<type>);
   TF_CALL_ALL_TYPES(REGISTER_PARTITION)
   #undef REGISTER_PARTITION
*/

#if !defined(__ANDROID__)

// Call "m" for all number types that support the comparison operations "<" and
// ">".
#define TF_CALL_REAL_NUMBER_TYPES(m) \
  m(float);                          \
  m(double);                         \
  m(int64);                          \
  m(int32);                          \
  m(uint8);                          \
  m(int16);                          \
  m(int8)

#define TF_CALL_REAL_NUMBER_TYPES_NO_INT32(m) \
  m(float);                                   \
  m(double);                                  \
  m(int64);                                   \
  m(uint8);                                   \
  m(int16);                                   \
  m(int8)

<<<<<<< HEAD
// Call "m" for all number types, including complex64.
#define TF_CALL_NUMBER_TYPES(m) \
  TF_CALL_REAL_NUMBER_TYPES(m); \
  m(complex64)

#define TF_CALL_NUMBER_TYPES_NO_INT32(m) \
  TF_CALL_REAL_NUMBER_TYPES_NO_INT32(m); \
  m(complex64)

// Call "m" on all types.
#define TF_CALL_ALL_TYPES(m) \
  TF_CALL_NUMBER_TYPES(m);   \
  m(bool);                   \
=======
// Call "m" for all number types, including complex64 and complex128.
#define TF_CALL_NUMBER_TYPES(m) \
  TF_CALL_REAL_NUMBER_TYPES(m); \
  m(complex64);                 \
  m(complex128)

#define TF_CALL_NUMBER_TYPES_NO_INT32(m) \
  TF_CALL_REAL_NUMBER_TYPES_NO_INT32(m); \
  m(complex64);                          \
  m(complex128)

#define TF_CALL_POD_TYPES(m) \
  TF_CALL_NUMBER_TYPES(m);   \
  m(bool)

// Call "m" on all types.
#define TF_CALL_ALL_TYPES(m) \
  TF_CALL_POD_TYPES(m);      \
>>>>>>> tensorflow/master
  m(string)

// Call "m" on all types supported on GPU.
#define TF_CALL_GPU_NUMBER_TYPES(m) \
  m(float);                         \
  m(double)

<<<<<<< HEAD
#else  // defined(__ANDROID__)
=======
// Call "m" on all quantized types.
#define TF_CALL_QUANTIZED_TYPES(m) \
  m(qint8);                        \
  m(quint8);                       \
  m(qint32)

#elif defined(__ANDROID_TYPES_FULL__)

#define TF_CALL_REAL_NUMBER_TYPES(m) \
  m(float);                          \
  m(int32);                          \
  m(int64)

#define TF_CALL_NUMBER_TYPES(m) TF_CALL_REAL_NUMBER_TYPES(m)

#define TF_CALL_REAL_NUMBER_TYPES_NO_INT32(m) \
  m(float);                                   \
  m(int64)

#define TF_CALL_NUMBER_TYPES_NO_INT32(m) TF_CALL_REAL_NUMBER_TYPES_NO_INT32(m)

#define TF_CALL_POD_TYPES(m) TF_CALL_REAL_NUMBER_TYPES(m)

#define TF_CALL_ALL_TYPES(m) TF_CALL_REAL_NUMBER_TYPES(m)

// Maybe we could put an empty macro here for Android?
#define TF_CALL_GPU_NUMBER_TYPES(m) m(float)

// Call "m" on all quantized types.
#define TF_CALL_QUANTIZED_TYPES(m) \
  m(qint8);                        \
  m(quint8);                       \
  m(qint32)

#else  // defined(__ANDROID__) && !defined(__ANDROID_TYPES_FULL__)
>>>>>>> tensorflow/master

#define TF_CALL_REAL_NUMBER_TYPES(m) \
  m(float);                          \
  m(int32)

#define TF_CALL_NUMBER_TYPES(m) TF_CALL_REAL_NUMBER_TYPES(m)

#define TF_CALL_REAL_NUMBER_TYPES_NO_INT32(m) m(float)

#define TF_CALL_NUMBER_TYPES_NO_INT32(m) TF_CALL_REAL_NUMBER_TYPES_NO_INT32(m)

<<<<<<< HEAD
=======
#define TF_CALL_POD_TYPES(m) TF_CALL_REAL_NUMBER_TYPES(m)

>>>>>>> tensorflow/master
#define TF_CALL_ALL_TYPES(m) TF_CALL_REAL_NUMBER_TYPES(m)

// Maybe we could put an empty macro here for Android?
#define TF_CALL_GPU_NUMBER_TYPES(m) m(float)

<<<<<<< HEAD
=======
#define TF_CALL_QUANTIZED_TYPES(m)

>>>>>>> tensorflow/master
#endif  // defined(__ANDROID__)

#endif  // TENSORFLOW_FRAMEWORK_REGISTER_TYPES_H_

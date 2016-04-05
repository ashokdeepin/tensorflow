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
#include "tensorflow/stream_executor/stream_executor_internal.h"

#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"

namespace perftools {
namespace gputools {
namespace internal {

// -- CUDA

StreamExecutorFactory* MakeCUDAExecutorImplementation() {
  static StreamExecutorFactory instance;
  return &instance;
}
<<<<<<< HEAD
EventFactory* MakeCUDAEventImplementation() {
  static EventFactory instance;
  return &instance;
}
StreamFactory* MakeCUDAStreamImplementation() {
  static StreamFactory instance;
  return &instance;
}
TimerFactory* MakeCUDATimerImplementation() {
  static TimerFactory instance;
  return &instance;
}
KernelFactory* MakeCUDAKernelImplementation() {
  static KernelFactory instance;
  return &instance;
}
=======
>>>>>>> tensorflow/master

// -- OpenCL

StreamExecutorFactory* MakeOpenCLExecutorImplementation() {
  static StreamExecutorFactory instance;
  return &instance;
}
<<<<<<< HEAD
StreamExecutorFactory* MakeOpenCLAlteraExecutorImplementation() {
  static StreamExecutorFactory instance;
  return &instance;
}
StreamFactory* MakeOpenCLStreamImplementation() {
  static StreamFactory instance;
  return &instance;
}
TimerFactory* MakeOpenCLTimerImplementation() {
  static TimerFactory instance;
  return &instance;
}
KernelFactory* MakeOpenCLKernelImplementation() {
  static KernelFactory instance;
  return &instance;
}
=======
>>>>>>> tensorflow/master

// -- Host

StreamExecutorFactory MakeHostExecutorImplementation;
<<<<<<< HEAD
StreamFactory MakeHostStreamImplementation;
TimerFactory MakeHostTimerImplementation;
=======
>>>>>>> tensorflow/master


}  // namespace internal
}  // namespace gputools
}  // namespace perftools

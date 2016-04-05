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
#ifndef TENSORFLOW_FRAMEWORK_NUMERIC_TYPES_H_
#define TENSORFLOW_FRAMEWORK_NUMERIC_TYPES_H_

#include <complex>

<<<<<<< HEAD
#include "tensorflow/core/platform/port.h"
=======
#include "tensorflow/core/platform/types.h"
>>>>>>> tensorflow/master

namespace tensorflow {

// Single precision complex.
typedef std::complex<float> complex64;
<<<<<<< HEAD
=======
// Double precision complex.
typedef std::complex<double> complex128;
>>>>>>> tensorflow/master

}  // end namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_NUMERIC_TYPES_H_

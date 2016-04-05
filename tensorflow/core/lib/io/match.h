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
#ifndef TENSORFLOW_LIB_IO_MATCH_H_
#define TENSORFLOW_LIB_IO_MATCH_H_

#include <vector>
<<<<<<< HEAD
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/env.h"
=======
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/env.h"
>>>>>>> tensorflow/master

namespace tensorflow {
class Env;
namespace io {

// Given a pattern, return the set of files that match the pattern.
// Note that this routine only supports wildcard characters in the
// basename portion of the pattern, not in the directory portion.  If
// successful, return Status::OK and store the matching files in
// "*results".  Otherwise, return a non-OK status.
Status GetMatchingFiles(Env* env, const string& pattern,
<<<<<<< HEAD
                             std::vector<string>* results);
=======
                        std::vector<string>* results);
>>>>>>> tensorflow/master

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_IO_MATCH_H_

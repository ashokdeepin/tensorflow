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
#ifndef TENSORFLOW_COMMON_RUNTIME_SESSION_FACTORY_H_
#define TENSORFLOW_COMMON_RUNTIME_SESSION_FACTORY_H_

#include <string>

<<<<<<< HEAD
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/status.h"
=======
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"
>>>>>>> tensorflow/master

namespace tensorflow {

class Session;
<<<<<<< HEAD
class SessionOptions;
=======
struct SessionOptions;
>>>>>>> tensorflow/master

class SessionFactory {
 public:
  virtual Session* NewSession(const SessionOptions& options) = 0;
<<<<<<< HEAD
  virtual ~SessionFactory() {}
  static void Register(const string& runtime_type, SessionFactory* factory);
  static SessionFactory* GetFactory(const string& runtime_type);
=======
  virtual bool AcceptsOptions(const SessionOptions& options) = 0;
  virtual ~SessionFactory() {}
  static void Register(const string& runtime_type, SessionFactory* factory);
  static Status GetFactory(const SessionOptions& options,
                           SessionFactory** out_factory);
>>>>>>> tensorflow/master
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_SESSION_FACTORY_H_

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
// The utility to write checkpoints for google brain tensor ops and v3
// checkpoints for dist_belief.
//

#ifndef TENSORFLOW_UTIL_TENSOR_SLICE_WRITER_H_
#define TENSORFLOW_UTIL_TENSOR_SLICE_WRITER_H_

#include <unordered_map>

<<<<<<< HEAD
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/tensor_shape.h"
#include "tensorflow/core/public/status.h"
=======
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
>>>>>>> tensorflow/master
#include "tensorflow/core/util/saved_tensor_slice.pb.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

namespace tensorflow {

namespace checkpoint {

class TensorSliceWriter {
 public:
  // Abstract interface that TensorSliceWriter uses for building
  class Builder {
   public:
    virtual ~Builder() {}
    virtual void Add(StringPiece key, StringPiece value) = 0;
    virtual Status Finish(int64* file_size) = 0;
  };
<<<<<<< HEAD
  typedef std::function<Status(const string&, Builder**)>
      CreateBuilderFunction;
=======
  typedef std::function<Status(const string&, Builder**)> CreateBuilderFunction;
>>>>>>> tensorflow/master

  TensorSliceWriter(const string& filename,
                    CreateBuilderFunction create_builder);
  virtual ~TensorSliceWriter() {}
  // Adds a slice. We support float and int32 for now.
  // TODO(yangke): add more supports
  template <typename T>
  Status Add(const string& name, const TensorShape& shape,
<<<<<<< HEAD
                  const TensorSlice& slice, const T* data);
=======
             const TensorSlice& slice, const T* data);
>>>>>>> tensorflow/master
  Status Finish();

 private:
  // Allocate "num_elements" elements in "ss" and save the data in "data"
  // there.
  template <typename T>
  static void SaveData(const T* data, int num_elements, SavedSlice* ss);

  const string filename_;
  const CreateBuilderFunction create_builder_;
  const string tmpname_;

  // A mapping from the tensor names to their index in meta_.saved_slice_meta()
  std::unordered_map<string, int> name_to_index_;
  // The metadata that holds all the saved tensor slices.
  SavedTensorSlices sts_;
  // The data to be written to the builder
  std::map<string, string> data_;
  // Total number of slices written
  int slices_;
  TF_DISALLOW_COPY_AND_ASSIGN(TensorSliceWriter);
};

template <typename T>
Status TensorSliceWriter::Add(const string& name, const TensorShape& shape,
<<<<<<< HEAD
                                   const TensorSlice& slice, const T* data) {
=======
                              const TensorSlice& slice, const T* data) {
>>>>>>> tensorflow/master
  // The tensor and the slice have to be compatible
  if (shape.dims() != slice.dims()) {
    return errors::Internal("Incompatible tensor shape and slice: ", "shape = ",
                            shape.DebugString(), ", slice = ",
                            slice.DebugString());
  }
  DataType dt = DataTypeToEnum<T>::value;
  // We need to add an entry for "name" if there isn't an entry already.
  int index = gtl::FindWithDefault(name_to_index_, name, -1);
  if (index >= 0) {
    // The same tensor has been registered -- we verify that the shapes and the
    // type agree.
    const SavedSliceMeta& ssm = sts_.meta().tensor(index);
    CHECK_EQ(name, ssm.name()) << ssm.ShortDebugString();
    TensorShape ssm_shape(ssm.shape());
    if (!shape.IsSameSize(ssm_shape)) {
      return errors::Internal("Mismatching shapes: existing tensor = ",
                              ssm_shape.DebugString(), ", trying to add name ",
                              name, ", shape = ", shape.DebugString());
    }
    if (dt != ssm.type()) {
      return errors::Internal(
          "Mismatching types: existing type = ", DataTypeString(ssm.type()),
          ", trying to add name ", name, ", type = ", DataTypeString(dt));
    }
  } else {
    // Insert the new tensor name with the shape information
    index = sts_.meta().tensor_size();
    name_to_index_.insert(std::make_pair(name, index));
    SavedSliceMeta* ssm = sts_.mutable_meta()->add_tensor();
    ssm->set_name(name);
    shape.AsProto(ssm->mutable_shape());
    ssm->set_type(dt);
  }
  // Now we need to add the slice info the list of slices.
  SavedSliceMeta* ssm = sts_.mutable_meta()->mutable_tensor(index);
  slice.AsProto(ssm->add_slice());

  // Now we need to add the real data.
  {
    SavedTensorSlices sts;
    SavedSlice* ss = sts.mutable_data();
    ss->set_name(name);
    slice.AsProto(ss->mutable_slice());
    TensorShape saved_shape(ssm->shape());
    TensorShape sliced_shape;
    TF_RETURN_IF_ERROR(slice.SliceTensorShape(saved_shape, &sliced_shape));
    SaveData(data, sliced_shape.num_elements(), ss);
    string key = EncodeTensorNameSlice(name, slice);
    // TODO(yangke): consider doing a two-pass thing where the first pass just
    // list the tensor slices we want to save and then another pass to actually
    // set the data. Need to figure out if the interface works well.
    std::pair<string, string> key_value(key, "");
<<<<<<< HEAD
    sts.AppendToString(&key_value.second);
=======
    if (!sts.AppendToString(&key_value.second)) {
      return errors::Internal("Error writing Tensor. Possible size overflow.");
    }
>>>>>>> tensorflow/master
    data_.insert(key_value);
  }
  ++slices_;
  return Status::OK();
}

template <typename T>
void TensorSliceWriter::SaveData(const T* data, int num_elements,
                                 SavedSlice* ss) {
  Fill(data, num_elements, ss->mutable_data());
}

// Create a table builder that will write to "filename" in
// tensorflow::io::Table format.  If successful, return OK
// and set "*builder" to the allocated builder.  Otherwise, return a
// non-OK status.
Status CreateTableTensorSliceBuilder(const string& filename,
<<<<<<< HEAD
                                          TensorSliceWriter::Builder** builder);
=======
                                     TensorSliceWriter::Builder** builder);
>>>>>>> tensorflow/master

}  // namespace checkpoint

}  // namespace tensorflow

#endif  // TENSORFLOW_UTIL_TENSOR_SLICE_WRITER_H_

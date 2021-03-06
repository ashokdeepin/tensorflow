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
#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_GPU_DEVICE_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_GPU_DEVICE_H_

<<<<<<< HEAD
#include "tensorflow/core/common_runtime/device_factory.h"
=======
#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
>>>>>>> tensorflow/master
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
<<<<<<< HEAD
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/stream_executor/stream.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

class EigenAllocator;

=======
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

>>>>>>> tensorflow/master
class BaseGPUDevice : public LocalDevice {
 public:
  BaseGPUDevice(const SessionOptions& options, const string& name,
                Bytes memory_limit, BusAdjacency bus_adjacency, int gpu_id,
                const string& physical_device_desc, Allocator* gpu_allocator,
<<<<<<< HEAD
                Allocator* cpu_allocator);
=======
                Allocator* cpu_allocator, bool sync_every_op,
                int32 max_streams);
>>>>>>> tensorflow/master

  ~BaseGPUDevice() override;

  // GPU devices require the Op Compute method to save a reference to
  // any temporary tensors that are allocated until the Op execution
  // completes.
<<<<<<< HEAD
  bool SaveTemporaryTensors() const override { return true; }
=======
  bool RequiresRecordingAccessedTensors() const override;

  void ConsumeListOfAccessedTensors(
      DeviceContext* device_context,
      const TensorReferenceVector& tensor_refs) override;
>>>>>>> tensorflow/master

  Status FillContextMap(const Graph* graph,
                        DeviceContextMap* device_context_map);

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;

  Status Sync() override;

  void ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                    AsyncOpKernel::DoneCallback done) override;

  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override;

  // The caller owns the returned device.
<<<<<<< HEAD
  const PerOpGpuDevice* MakeGpuDevice(DeviceContext* dc,
                                      Allocator* allocator) override;
=======
  PerOpGpuDevice* MakeGpuDevice() override;

  void ReinitializeGpuDevice(OpKernelContext* context, PerOpGpuDevice* device,
                             DeviceContext* dc, Allocator* allocator) override;
>>>>>>> tensorflow/master

 protected:
  Allocator* gpu_allocator_;  // not owned
  Allocator* cpu_allocator_;  // not owned

 private:
<<<<<<< HEAD
  std::vector<gpu::Stream*> streams_;
=======
  struct StreamGroup {
    gpu::Stream* compute;
    gpu::Stream* host_to_device;
    gpu::Stream* device_to_host;
    gpu::Stream* device_to_device;
  };
  gtl::InlinedVector<StreamGroup, 4> streams_;
>>>>>>> tensorflow/master
  std::vector<GPUDeviceContext*> device_contexts_;
  GpuDeviceInfo* gpu_device_info_ = nullptr;
  mutex trace_mu_;
  int gpu_id_ = -1;
<<<<<<< HEAD
  std::unique_ptr<EventMgr> em_;

  const PerOpGpuDevice* NewDevice(int stream_id, Allocator* allocator);
=======
  const bool sync_every_op_ = false;
  std::unique_ptr<EventMgr> em_;

  void ReinitializeDevice(OpKernelContext* context, PerOpGpuDevice* device,
                          int stream_id, Allocator* allocator);
>>>>>>> tensorflow/master
};

class BaseGPUDeviceFactory : public DeviceFactory {
 public:
  void CreateDevices(const SessionOptions& options, const string& name_prefix,
                     std::vector<Device*>* devices) override;

 private:
  LocalDevice* CreateGPUDevice(const SessionOptions& options,
                               const string& name, int gpu_id);

  virtual LocalDevice* CreateGPUDevice(const SessionOptions& options,
                                       const string& name, Bytes memory_limit,
                                       BusAdjacency bus_adjacency, int gpu_id,
                                       const string& physical_device_desc,
                                       Allocator* gpu_allocator,
                                       Allocator* cpu_allocator) = 0;

  void GetValidDeviceIds(std::vector<int>* ids);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_GPU_DEVICE_H_

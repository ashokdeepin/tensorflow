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
#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_GPU_EVENT_MGR_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_GPU_EVENT_MGR_H_

#include <deque>
#include <vector>
<<<<<<< HEAD
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/tensor.h"
=======
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
>>>>>>> tensorflow/master

namespace perftools {
namespace gputools {
class Event;
class Stream;
class StreamExecutor;
}  // namespace gputools
}  // namespace perftools

namespace tensorflow {

<<<<<<< HEAD
=======
class GPUOptions;

>>>>>>> tensorflow/master
// An object to keep track of pending Events in the StreamExecutor streams
// and associated Tensors that cannot safely be deleted until the associated
// Events are recorded.
class EventMgr {
 public:
<<<<<<< HEAD
  explicit EventMgr(perftools::gputools::StreamExecutor* se);

  ~EventMgr();

  // Takes ownership of *tensors and deletes it as soon as all events
  // currently enqueued on *stream have completed.
  inline void ThenDeleteTensors(perftools::gputools::Stream* stream,
                                std::vector<Tensor>* tensors) {
    mutex_lock l(mu_);
    QueueTensors(stream, tensors);
    PollEvents(false);
  }
=======
  EventMgr(perftools::gputools::StreamExecutor* se,
           const GPUOptions& gpu_options);

  ~EventMgr();

  // Releases the references on the elements of "tensors" as soon as
  // all events currently enqueued on "stream" have completed.
  void ThenDeleteTensors(perftools::gputools::Stream* stream,
                         const TensorReferenceVector& tensors);
>>>>>>> tensorflow/master

  struct BufRec {
    Allocator* alloc;
    void* buf;
<<<<<<< HEAD
=======
    // operation and step_id are only populated when
    // LogMemory::IsEnabled() is true.
    string operation;
    int64 step_id;
>>>>>>> tensorflow/master
  };

  // Takes ownership of *bufrec.buf and calls bufrec.alloc->DeallocateRaw()
  // on it as soon as all events currently enqueued on *stream have completed.
  inline void ThenDeleteBuffer(perftools::gputools::Stream* stream,
                               BufRec bufrec) {
<<<<<<< HEAD
    mutex_lock l(mu_);
    QueueBuffer(stream, bufrec);
    PollEvents(false);
=======
    ToFreeVector to_free;
    {
      mutex_lock l(mu_);
      QueueBuffer(stream, bufrec);
      PollEvents(false, &to_free);
    }
    FreeMemory(to_free);
>>>>>>> tensorflow/master
  }

  inline void ThenExecute(perftools::gputools::Stream* stream,
                          std::function<void()> func) {
<<<<<<< HEAD
    mutex_lock l(mu_);
    QueueFunc(stream, func);
    PollEvents(false);
=======
    ToFreeVector to_free;
    {
      mutex_lock l(mu_);
      QueueFunc(stream, func);
      PollEvents(false, &to_free);
    }
    FreeMemory(to_free);
>>>>>>> tensorflow/master
  }

 private:
  friend class TEST_EventMgrHelper;
<<<<<<< HEAD
  mutex mu_;
  perftools::gputools::StreamExecutor* exec_;

  struct InUse {
    perftools::gputools::Event* event;
    std::vector<Tensor>* mem;
=======
  perftools::gputools::StreamExecutor* const exec_;
  const int64 deferred_bytes_threshold_;
  mutex mu_;
  condition_variable events_pending_ GUARDED_BY(mu_);

  void FlushAccumulatedTensors() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  struct InUse {
    perftools::gputools::Event* event;
    TensorReferenceVector* mem;
>>>>>>> tensorflow/master
    BufRec bufrec;
    std::function<void()> func;
  };

<<<<<<< HEAD
=======
  typedef gtl::InlinedVector<InUse, 4> ToFreeVector;

  void FreeMemory(const ToFreeVector& to_free) {
    for (const auto& iu : to_free) {
      if (iu.mem != nullptr) {
        for (auto& t : *(iu.mem)) {
          t.Unref();
        }
        delete iu.mem;
      }
      if (iu.bufrec.buf) {
        if (LogMemory::IsEnabled()) {
          LogMemory::RecordRawDeallocation(iu.bufrec.operation,
                                           iu.bufrec.step_id, iu.bufrec.buf,
                                           iu.bufrec.alloc, false);
        }
        iu.bufrec.alloc->DeallocateRaw(iu.bufrec.buf);
      }
      // The function must be called in another thread.
      if (iu.func != nullptr) threadpool_.Schedule(iu.func);
    }
  }

>>>>>>> tensorflow/master
  // Stream-enqueue an unused Event and save with it a collection of
  // Tensors and/or a BufRec to be deleted only after the Event
  // records.
  void QueueInUse(perftools::gputools::Stream* stream, InUse in_use)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void QueueTensors(perftools::gputools::Stream* stream,
<<<<<<< HEAD
                    std::vector<Tensor>* tensors)
=======
                    TensorReferenceVector* tensors)
>>>>>>> tensorflow/master
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    QueueInUse(stream, {nullptr, tensors, BufRec(), nullptr});
  }

  void QueueBuffer(perftools::gputools::Stream* stream, BufRec bufrec)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    QueueInUse(stream, {nullptr, nullptr, bufrec, nullptr});
  }

  void QueueFunc(perftools::gputools::Stream* stream,
                 std::function<void()> func) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    QueueInUse(stream, {nullptr, nullptr, BufRec(), func});
  }

  // This function should be called at roughly the same tempo as
  // QueueTensors() to check whether pending events have recorded,
<<<<<<< HEAD
  // and then retire them.
  void PollEvents(bool is_dedicated_poller) EXCLUSIVE_LOCKS_REQUIRED(mu_);
=======
  // and then retire them.  It appends InUse elements that need cleanup
  // to "*to_free".  The caller should call FreeMemory(to_free)
  // when this returns.
  void PollEvents(bool is_dedicated_poller, ToFreeVector* to_free)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);
>>>>>>> tensorflow/master

  // An internal polling loop that runs at a low frequency to clear
  // straggler Events.
  void PollLoop();

<<<<<<< HEAD
  // A stack of unused events
  std::vector<perftools::gputools::Event*> free_events_ GUARDED_BY(mu_);

  // A FIFO queue of InUse events and associated tensors.
  std::deque<InUse> used_events_ GUARDED_BY(mu_);

  Notification stop_polling_;
  Notification polling_stopped_;
=======
  // Setup/Teardown functions for the polling loop.
  void StartPollingLoop();
  void StopPollingLoop();

  // A stack of unused events
  std::vector<perftools::gputools::Event*> free_events_ GUARDED_BY(mu_);

  // Buffered list of tensors waiting to have an event queued for deletion
  perftools::gputools::Stream* accumulated_stream_ GUARDED_BY(mu_);
  TensorReferenceVector* accumulated_tensors_ GUARDED_BY(mu_);
  // Sum of the TotalBytes() of the tensors in "accumulated_tensors_"
  int64 accumulated_tensor_bytes_ GUARDED_BY(mu_);

  // A FIFO queue of InUse events and associated tensors.
  std::deque<InUse> used_events_ GUARDED_BY(mu_);

  std::unique_ptr<Notification> stop_polling_;
  std::unique_ptr<Notification> polling_stopped_;
>>>>>>> tensorflow/master

  // The main PollLoop for the event manager runs in this threadpool.
  thread::ThreadPool threadpool_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_GPU_EVENT_MGR_H_

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/training/coordinator.h"

#include "tensorflow/cc/training/queue_runner.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace {

using error::Code;

void WaitForStopThread(Coordinator* coord, bool* stopped, Notification* done) {
  coord->WaitForStop();
  *stopped = true;
  done->Notify();
}

TEST(CoordinatorTest, TestStopAndWaitOnStop) {
  Coordinator coord;
  EXPECT_EQ(coord.ShouldStop(), false);

  bool stopped = false;
  Notification done;
  Env::Default()->SchedClosure(
      std::bind(&WaitForStopThread, &coord, &stopped, &done));
  Env::Default()->SleepForMicroseconds(10000000);
  EXPECT_EQ(stopped, false);

  coord.RequestStop();
  done.WaitForNotification();
  EXPECT_EQ(stopped, true);
  EXPECT_EQ(coord.ShouldStop(), true);
}

class MockQueueRunner : public RunnerInterface {
 public:
  MockQueueRunner(Coordinator* coord) {
    coord_ = coord;
    join_counter_ = nullptr;
    thread_pool_.reset(new thread::ThreadPool(Env::Default(), "test-pool", 10));
  }

  MockQueueRunner(Coordinator* coord, int* join_counter)
      : MockQueueRunner(coord) {
    join_counter_ = join_counter;
  }

  void StartCounting(std::atomic<int>* counter, int until) {
    thread_pool_->Schedule(
        std::bind(&MockQueueRunner::CountThread, this, counter, until));
  }

  void StartSettingStatus(const Status& status, BlockingCounter* counter) {
    thread_pool_->Schedule(
        std::bind(&MockQueueRunner::SetStatusThread, this, status, counter));
  }

  Status Join() {
    if (join_counter_ != nullptr) {
      (*join_counter_)++;
    }
    thread_pool_.reset();
    return status_;
  }

  Status GetStatus() { return status_; }

  void SetStatus(const Status& status) { status_ = status; }

 private:
  void CountThread(std::atomic<int>* counter, int until) {
    while (!coord_->ShouldStop() && counter->load() < until) {
      (*counter)++;
      Env::Default()->SleepForMicroseconds(100000);
    }
    coord_->RequestStop();
  }
  void SetStatusThread(const Status& status, BlockingCounter* counter) {
    Env::Default()->SleepForMicroseconds(100000);
    SetStatus(status);
    counter->DecrementCount();
  }
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  Status status_;
  Coordinator* coord_;
  int* join_counter_;
};

TEST(CoordinatorTest, TestRealStop) {
  std::atomic<int> counter(0);
  Coordinator coord;

  std::unique_ptr<MockQueueRunner> qr1(new MockQueueRunner(&coord));
  qr1->StartCounting(&counter, 100);
  coord.RegisterRunner(std::move(qr1));

  std::unique_ptr<MockQueueRunner> qr2(new MockQueueRunner(&coord));
  qr2->StartCounting(&counter, 100);
  coord.RegisterRunner(std::move(qr2));

  // Wait until the counting has started
  while (counter.load() == 0)
    ;
  coord.RequestStop();

  int temp_counter = counter.load();
  Env::Default()->SleepForMicroseconds(10000000);
  EXPECT_EQ(temp_counter, counter.load());
  TF_EXPECT_OK(coord.Join());
}

TEST(CoordinatorTest, TestRequestStop) {
  Coordinator coord;
  std::atomic<int> counter(0);
  std::unique_ptr<MockQueueRunner> qr;
  for (int i = 0; i < 10; i++) {
    qr.reset(new MockQueueRunner(&coord));
    qr->StartCounting(&counter, 10);
    coord.RegisterRunner(std::move(qr));
  }

  coord.WaitForStop();
  EXPECT_EQ(coord.ShouldStop(), true);
  EXPECT_EQ(counter.load(), 10);
  TF_EXPECT_OK(coord.Join());
}

TEST(CoordinatorTest, TestJoin) {
  Coordinator coord;
  int join_counter = 0;
  std::unique_ptr<MockQueueRunner> qr1(
      new MockQueueRunner(&coord, &join_counter));
  coord.RegisterRunner(std::move(qr1));
  std::unique_ptr<MockQueueRunner> qr2(
      new MockQueueRunner(&coord, &join_counter));
  coord.RegisterRunner(std::move(qr2));

  TF_EXPECT_OK(coord.Join());
  EXPECT_EQ(join_counter, 2);
}

TEST(CoordinatorTest, StatusReporting) {
  Coordinator coord({Code::CANCELLED, Code::OUT_OF_RANGE});
  BlockingCounter counter(3);

  std::unique_ptr<MockQueueRunner> qr1(new MockQueueRunner(&coord));
  qr1->StartSettingStatus(Status(Code::CANCELLED, ""), &counter);
  coord.RegisterRunner(std::move(qr1));

  std::unique_ptr<MockQueueRunner> qr2(new MockQueueRunner(&coord));
  qr2->StartSettingStatus(Status(Code::INVALID_ARGUMENT, ""), &counter);
  coord.RegisterRunner(std::move(qr2));

  std::unique_ptr<MockQueueRunner> qr3(new MockQueueRunner(&coord));
  qr3->StartSettingStatus(Status(Code::OUT_OF_RANGE, ""), &counter);
  coord.RegisterRunner(std::move(qr3));

  counter.Wait();
  EXPECT_EQ(coord.Join().code(), Code::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace tensorflow

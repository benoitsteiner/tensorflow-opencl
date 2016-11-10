/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <deque>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

class FileSystemRegistryImpl : public FileSystemRegistry {
 public:
  Status Register(const string& scheme, Factory factory) override;
  FileSystem* Lookup(const string& scheme) override;
  Status GetRegisteredFileSystemSchemes(std::vector<string>* schemes) override;

 private:
  mutable mutex mu_;
  mutable std::unordered_map<string, std::unique_ptr<FileSystem>> registry_
      GUARDED_BY(mu_);
};

Status FileSystemRegistryImpl::Register(const string& scheme,
                                        FileSystemRegistry::Factory factory) {
  mutex_lock lock(mu_);
  if (!registry_.emplace(string(scheme), std::unique_ptr<FileSystem>(factory()))
           .second) {
    return errors::AlreadyExists("File factory for ", scheme,
                                 " already registered");
  }
  return Status::OK();
}

FileSystem* FileSystemRegistryImpl::Lookup(const string& scheme) {
  mutex_lock lock(mu_);
  const auto found = registry_.find(scheme);
  if (found == registry_.end()) {
    return nullptr;
  }
  return found->second.get();
}

Status FileSystemRegistryImpl::GetRegisteredFileSystemSchemes(
    std::vector<string>* schemes) {
  mutex_lock lock(mu_);
  for (const auto& e : registry_) {
    schemes->push_back(e.first);
  }
  return Status::OK();
}

Env::Env() : file_system_registry_(new FileSystemRegistryImpl) {}

Status Env::GetFileSystemForFile(const string& fname, FileSystem** result) {
  StringPiece scheme, host, path;
  io::ParseURI(fname, &scheme, &host, &path);
  FileSystem* file_system = file_system_registry_->Lookup(scheme.ToString());
  if (!file_system) {
    return errors::Unimplemented("File system scheme ", scheme,
                                 " not implemented");
  }
  *result = file_system;
  return Status::OK();
}

Status Env::GetRegisteredFileSystemSchemes(std::vector<string>* schemes) {
  return file_system_registry_->GetRegisteredFileSystemSchemes(schemes);
}

Status Env::RegisterFileSystem(const string& scheme,
                               FileSystemRegistry::Factory factory) {
  return file_system_registry_->Register(scheme, factory);
}

Status Env::NewRandomAccessFile(const string& fname,
                                std::unique_ptr<RandomAccessFile>* result) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->NewRandomAccessFile(fname, result);
}

Status Env::NewReadOnlyMemoryRegionFromFile(
    const string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->NewReadOnlyMemoryRegionFromFile(fname, result);
}

Status Env::NewWritableFile(const string& fname,
                            std::unique_ptr<WritableFile>* result) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->NewWritableFile(fname, result);
}

Status Env::NewAppendableFile(const string& fname,
                              std::unique_ptr<WritableFile>* result) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->NewAppendableFile(fname, result);
}

Status Env::FileExists(const string& fname) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->FileExists(fname);
}

Status Env::GetChildren(const string& dir, std::vector<string>* result) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dir, &fs));
  return fs->GetChildren(dir, result);
}

Status Env::GetMatchingPaths(const string& pattern,
                             std::vector<string>* results) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(pattern, &fs));
  return fs->GetMatchingPaths(pattern, results);
}

Status Env::DeleteFile(const string& fname) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->DeleteFile(fname);
}

Status Env::RecursivelyCreateDir(const string& dirname) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));
  return fs->RecursivelyCreateDir(dirname);
}

Status Env::CreateDir(const string& dirname) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));
  return fs->CreateDir(dirname);
}

Status Env::DeleteDir(const string& dirname) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));
  return fs->DeleteDir(dirname);
}

Status Env::Stat(const string& fname, FileStatistics* stat) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->Stat(fname, stat);
}

Status Env::IsDirectory(const string& fname) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->IsDirectory(fname);
}

Status Env::DeleteRecursively(const string& dirname, int64* undeleted_files,
                              int64* undeleted_dirs) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(dirname, &fs));
  return fs->DeleteRecursively(dirname, undeleted_files, undeleted_dirs);
}

Status Env::GetFileSize(const string& fname, uint64* file_size) {
  FileSystem* fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(fname, &fs));
  return fs->GetFileSize(fname, file_size);
}

Status Env::RenameFile(const string& src, const string& target) {
  FileSystem* src_fs;
  FileSystem* target_fs;
  TF_RETURN_IF_ERROR(GetFileSystemForFile(src, &src_fs));
  TF_RETURN_IF_ERROR(GetFileSystemForFile(target, &target_fs));
  if (src_fs != target_fs) {
    return errors::Unimplemented("Renaming ", src, " to ", target,
                                 " not implemented");
  }
  return src_fs->RenameFile(src, target);
}

Thread::~Thread() {}

EnvWrapper::~EnvWrapper() {}

Status ReadFileToString(Env* env, const string& fname, string* data) {
  uint64 file_size;
  Status s = env->GetFileSize(fname, &file_size);
  if (!s.ok()) {
    return s;
  }
  std::unique_ptr<RandomAccessFile> file;
  s = env->NewRandomAccessFile(fname, &file);
  if (!s.ok()) {
    return s;
  }
  gtl::STLStringResizeUninitialized(data, file_size);
  char* p = gtl::string_as_array(data);
  StringPiece result;
  s = file->Read(0, file_size, &result, p);
  if (!s.ok()) {
    data->clear();
  } else if (result.size() != file_size) {
    s = errors::Aborted("File ", fname, " changed while reading: ", file_size,
                        " vs. ", result.size());
    data->clear();
  } else if (result.data() == p) {
    // Data is already in the correct location
  } else {
    memmove(p, result.data(), result.size());
  }
  return s;
}

Status WriteStringToFile(Env* env, const string& fname,
                         const StringPiece& data) {
  std::unique_ptr<WritableFile> file;
  Status s = env->NewWritableFile(fname, &file);
  if (!s.ok()) {
    return s;
  }
  s = file->Append(data);
  if (s.ok()) {
    s = file->Close();
  }
  return s;
}

// A ZeroCopyInputStream on a RandomAccessFile.
namespace {
class FileStream : public ::tensorflow::protobuf::io::ZeroCopyInputStream {
 public:
  explicit FileStream(RandomAccessFile* file) : file_(file), pos_(0) {}

  void BackUp(int count) override { pos_ -= count; }
  bool Skip(int count) override {
    pos_ += count;
    return true;
  }
  protobuf_int64 ByteCount() const override { return pos_; }
  Status status() const { return status_; }

  bool Next(const void** data, int* size) override {
    StringPiece result;
    Status s = file_->Read(pos_, kBufSize, &result, scratch_);
    if (result.empty()) {
      status_ = s;
      return false;
    }
    pos_ += result.size();
    *data = result.data();
    *size = result.size();
    return true;
  }

 private:
  static const int kBufSize = 512 << 10;

  RandomAccessFile* file_;
  int64 pos_;
  Status status_;
  char scratch_[kBufSize];
};

}  // namespace

Status WriteBinaryProto(Env* env, const string& fname,
                        const ::tensorflow::protobuf::MessageLite& proto) {
  string serialized;
  proto.AppendToString(&serialized);
  return WriteStringToFile(env, fname, serialized);
}

Status ReadBinaryProto(Env* env, const string& fname,
                       ::tensorflow::protobuf::MessageLite* proto) {
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(fname, &file));
  std::unique_ptr<FileStream> stream(new FileStream(file.get()));

  // TODO(jiayq): the following coded stream is for debugging purposes to allow
  // one to parse arbitrarily large messages for MessageLite. One most likely
  // doesn't want to put protobufs larger than 64MB on Android, so we should
  // eventually remove this and quit loud when a large protobuf is passed in.
  ::tensorflow::protobuf::io::CodedInputStream coded_stream(stream.get());
  // Total bytes hard limit / warning limit are set to 1GB and 512MB
  // respectively.
  coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);

  if (!proto->ParseFromCodedStream(&coded_stream)) {
    TF_RETURN_IF_ERROR(stream->status());
    return errors::DataLoss("Can't parse ", fname, " as binary proto");
  }
  return Status::OK();
}

Status WriteTextProto(Env* env, const string& fname,
                      const ::tensorflow::protobuf::Message& proto) {
#if !defined(TENSORFLOW_LITE_PROTOS)
  string serialized;
  if (!::tensorflow::protobuf::TextFormat::PrintToString(proto, &serialized)) {
    return errors::FailedPrecondition("Unable to convert proto to text.");
  }
  return WriteStringToFile(env, fname, serialized);
#else
  return errors::Unimplemented("Can't write text protos with protolite.");
#endif
}

Status ReadTextProto(Env* env, const string& fname,
                     ::tensorflow::protobuf::Message* proto) {
#if !defined(TENSORFLOW_LITE_PROTOS)
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(fname, &file));
  std::unique_ptr<FileStream> stream(new FileStream(file.get()));

  if (!::tensorflow::protobuf::TextFormat::Parse(stream.get(), proto)) {
    TF_RETURN_IF_ERROR(stream->status());
    return errors::DataLoss("Can't parse ", fname, " as text proto");
  }
  return Status::OK();
#else
  return errors::Unimplemented("Can't parse text protos with protolite.");
#endif
}

}  // namespace tensorflow

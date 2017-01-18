/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_DISASSEMBLER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_DISASSEMBLER_H_

#include <memory>
#include <string>

#include "external/llvm/include/llvm/MC/MCContext.h"
#include "external/llvm/include/llvm/MC/MCDisassembler/MCDisassembler.h"
#include "external/llvm/include/llvm/MC/MCInstPrinter.h"
#include "external/llvm/include/llvm/MC/MCInstrAnalysis.h"
#include "external/llvm/include/llvm/MC/MCObjectFileInfo.h"
#include "external/llvm/include/llvm/MC/MCSubtargetInfo.h"
#include "external/llvm/include/llvm/Object/ObjectFile.h"
#include "external/llvm/include/llvm/Target/TargetMachine.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace cpu {

// Class for disassembling object files (and potentially other constructs) into
// X86 assembly. Builds all the LLVM disassembly and instruction printing
// constructs from a given TargetMachine.
class Disassembler {
 public:
  explicit Disassembler(const llvm::TargetMachine& target_machine);

  // Returns a string containing the disassembled text sections of the given
  // object file.
  //
  // If we couldnt' retrieve a disassembler for this platform, an error status
  // is returned.
  StatusOr<string> DisassembleObjectFile(
      const llvm::object::ObjectFile& object_file) const;

 private:
  const llvm::MCSubtargetInfo& subtarget_info_;
  std::unique_ptr<llvm::MCObjectFileInfo> objfile_info_;
  std::unique_ptr<llvm::MCContext> mc_context_;
  std::unique_ptr<llvm::MCDisassembler> disassembler_;
  std::unique_ptr<llvm::MCInstPrinter> inst_printer_;
  std::unique_ptr<llvm::MCInstrAnalysis> inst_analysis_;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_DISASSEMBLER_H_

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Functionality to save information for the silicon debugger.
// Saves information about kernels and some variables to a file that is used by the silicon debugger (gdb server).

#pragma once

namespace tt::tt_metal {

// Initilize the file
void InitSiliconDebuggerInterfaceFile();

// Log kernel build information to the file
void SiliconDebuggerInterfaceLogKernel(std::shared_ptr<Kernel> kernel, const tt::tt_metal::JitBuildOptions &build_options);

}
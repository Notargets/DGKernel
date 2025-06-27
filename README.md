# DGKernel

DGKernel is a high-level abstraction for writing partition-parallel kernels for scientific computing in Go. It provides automatic memory management, code generation, and backend portability across CPU, GPU, and other accelerators through OCCA.

## Features

- **Automatic memory management** with alignment support
- **Partition-parallel execution** for domain decomposition
- **Code generation** for efficient data access patterns
- **Static data embedding** for optimal matrix operations
- **Backend portability** across CPU, GPU, and other accelerators

## Installation

```bash
go get github.com/notargets/DGKernel

License
This project is licensed under the MIT License - see the LICENSE file for details.

# Chapter 0 - Introduction

## The Gap Between Mathematics and Hardware

Modern computational physics exists at the intersection of three domains: mathematical formulations, numerical algorithms, and hardware architectures. Each domain has its own abstractions, constraints, and optimal patterns. The challenge lies not in any single domain, but in effectively bridging between them.

Consider element-based methods for solving partial differential equations. Mathematically, they involve:

- Basis functions defined on reference elements
- Operators for differentiation and integration
- Coupling between elements through interfaces

Computationally, achieving performance requires:

- Coalesced memory access on GPUs
- Cache-blocking on CPUs
- Minimized communication in distributed systems
- Vectorization of innermost loops

Most existing frameworks force a choice: either work at the mathematical level and accept performance limitations, or work at the hardware level and obscure the mathematics.

## What DGKernel Provides

DGKernel is a kernel generation system that bridges this gap through a focused computational model. It is built on a key principle: **problems can be decomposed into partitions where the vast majority of computation occurs locally, with consolidated communication between partitions**. This partition-centric model enables a clean mapping between mathematical elements and parallel hardware.

The system leverages OCCA’s kernel abstraction to target both SMP and GPU architectures through a structured parallel model that maps naturally to modern hardware architectures.

### OCCA Loop Model and Restrictions

OCCA provides a portable parallel programming model with specific constraints that shape how DGKernel structures computations:

**OCCA loop requirements:**

- Every kernel must have exactly one @outer loop level
- Every kernel must have at least one @inner loop nested within @outer
- No code or loops allowed between @outer and @inner declarations
- Loop ranges must be compile-time constants or kernel arguments
- @outer iterations execute independently (no dependencies between iterations)
- @inner iterations within an @outer iteration can share data via @shared memory
- Only @inner loops can be synchronized with barriers

**Hardware mapping:**

- @outer loops → GPU thread blocks or CPU parallel regions
- @inner loops → GPU threads or CPU vectorized/serial iterations

DGKernel adopts a consistent pattern within these constraints:

- @outer loops iterate over partitions
- @inner loops iterate over elements within partitions

While users can write kernels with different mappings, DGKernel’s automated patterns and optimizations assume this partition/element structure.

### The Technical Approach

DGKernel operates as a two-stage system:

**Stage 1: Specification**

- Define elements with their mathematical properties
- Compose operators that transform field data
- Describe computational stages and data dependencies

**Stage 2: Code Generation**

- Analyze data access patterns
- Generate architecture-specific kernels
- Optimize memory layouts and execution strategies

This separation allows algorithm developers to work with familiar mathematical concepts while the system handles hardware complexity.

## Core Components

### Elements as Mathematical Objects

In DGKernel, an element encapsulates the complete mathematical machinery needed for numerical computation:

**Core attributes common to all elements:**

- Polynomial order and basis type
- Number of nodes, faces, and dimensions
- Topological connectivity information

**Basis representation:**

- Interpolation matrices mapping between nodal and modal spaces
- Quadrature rules for accurate integration
- Face-to-volume lifting operators

**Geometric mappings:**

- Metric tensor components (rx, ry, rz, sx, sy, sz, tx, ty, tz)
- Jacobian transformations between reference and physical space
- Surface normal vectors and area scalings

**Mathematical operators:**

- Differentiation matrices in reference coordinates
- Mass and stiffness matrices
- Specialized operators for the element’s basis functions

These components form a complete mathematical description that the DGKernel system uses to generate efficient computational kernels. The element designer provides these mathematical building blocks once, and they become available to all algorithm developers through the operator contract system.

### Separation of Concerns

A critical design principle in DGKernel is the separation between element design and algorithm development:

**Element Designer’s Role:**

- Define mathematical properties (basis functions, quadrature, etc.)
- Map these to granular operator contracts (Gradient, Divergence, Mass, LIFT)
- Each operator has clear input/output specifications
- This mapping is done once per element type

**Algorithm Developer’s Role:**

- Compose operators within OCCA kernel code to implement physics
- Write C-like code that calls operators as building blocks
- Focus on the mathematical algorithm, not operator implementation
- Work at the level of physics equations

For example, an algorithm developer might write:

```c
@kernel void computeRHS(...) {
    for (int p = 0; p < Npartitions; ++p; @outer) {
        // Operators work on partition-level data
        PhysicalGradient(u, ux, uy, uz, K[part]);
        ComputeFaceFlux(faceValues, faceFlux, K[part]);
        MATMUL_LIFT(faceFlux, lifted, K[part]);
        
        // Element-level updates
        for (int elem = 0; elem < KpartMax; ++elem; @inner) {
            if (elem < K[part]) {
                for (int n = 0; n < NP_TET; ++n) {
                    int id = elem * NP_TET + n;
                    // Physics: Burgers equation div(u²/2)
                    real_t rhs = -u[id] * (ux[id] + uy[id] + uz[id]);
                    u[id] = resu[id] + b_dt * (rhs + lifted[id]);
                }
            }
        }
    }
}
```

DGKernel ensures the operators have access to properly sized and strided arrays based on the element type in each partition.

### Historical Context

The idea of separating numerical methods from implementation details isn’t new. Starting in the 1990s, several C++ projects attempted this goal:

- Template-based libraries from MIT and other institutions
- Object-oriented finite element frameworks
- Expression template systems

However, these systems struggled to gain widespread adoption due to their inherent complexity. Template metaprogramming, multiple inheritance, and complex build systems created barriers to entry that limited their audience to C++ experts.

Modern languages like Go provide a different path. With:

- Simple, readable syntax
- Fast compilation
- Built-in concurrency primitives
- Excellent tooling

Go allows DGKernel to achieve the same flexibility as earlier C++ systems but with dramatically reduced complexity. The code remains accessible to domain scientists, not just systems programmers.

### The Partition Buffer System

The Partition Buffer manages communication between partitions for various types of data:

**Design characteristics:**

- Indexes any partition boundary data with element-based addressing
- User-defined strides for different data types (face points, vertices, etc.)
- Consolidates all inter-partition communication into efficient message passing
- Contiguous send/receive buffers optimized for MPI and GPU transfers

**Flexible data communication:**

- Face data: stride over face points for flux computation
- Vertex data: stride over vertex values for continuous fields
- Edge data: stride over edge quadrature points
- Custom data: any element-based data at partition boundaries

**Usage pattern:**

```c
// Pack various boundary data types
PackBoundaryData(faceValues, vertexValues, customData, sendBuffer);
// Single consolidated communication step
ExchangePartitionBuffers();
// Unpack for local use
UnpackBoundaryData(receiveBuffer, remoteFaceData, remoteVertexData);
```

This design supports complex coupling between partitions while maintaining efficient communication patterns for distributed computing.

## Practical Considerations

### Memory Management

DGKernel provides explicit control over memory allocation and layout:

- **Persistent arrays**: Solution fields that persist across time steps
- **Workspace arrays**: Temporary storage reused within kernels
- **Communication buffers**: Explicitly managed for MPI/GPU transfers

The system computes optimal layouts based on:

- Access patterns in operators
- Architecture memory hierarchy
- Communication requirements

### Parallelization Strategy

The framework uses OCCA’s two-level parallelization model:

```c
// OCCA kernel structure
for (int p = 0; p < Npartitions; ++p; @outer) {
    // Partition-level parallelism
    for (int e = 0; e < ElementsPerPartition; ++e; @inner) {
        // Element-level parallelism
        // Computation here...
    }
}
```

This maps directly to hardware:

- **GPU**: @outer → thread blocks, @inner → threads
- **CPU**: @outer → parallel regions, @inner → vectorized loops

The beauty of this model is its simplicity: the same two-level structure works efficiently across architectures. OCCA handles the translation to architecture-specific code.

### Hardware Abstraction Through OCCA

OCCA provides the hardware abstraction layer, translating the @outer/@inner parallel model to each architecture:

```c
// You write OCCA kernels
for (int p = 0; p < Npartitions; ++p; @outer) {
    for (int e = 0; e < ElementsPerPartition; ++e; @inner) {
        // Your computation
    }
}

// OCCA generates architecture-specific code
// - CUDA: Thread blocks and threads
// - OpenMP: Parallel regions and vectorization
// - OpenCL: Work groups and work items
// - HIP: Thread blocks and threads
```

DGKernel focuses on structuring computations to work well with OCCA’s model, while OCCA handles all hardware-specific translation.

## Design Philosophy

### Element Abstraction and Algorithm Simplification

A significant benefit of the operator contract system is how it enables element-agnostic algorithm development:

**Unified algorithm expression:**

```c
@kernel void computeFlowRHS(...) {
    // This kernel works for ANY element type
    for (int p = 0; p < Npartitions; ++p; @outer) {
        // Operators adapt to partition's element type
        PhysicalGradient(velocity, vel_x, vel_y, vel_z, K[part]);
        ComputeFlux(vel_x, vel_y, vel_z, flux, K[part]);
        MATMUL_LIFT(flux, lifted, K[part]);
        // Physics remains the same regardless of tets, hexes, prisms...
    }
}
```

**Automatic mortar coupling:**

- Mixed element meshes use different partitions for different element types
- Element developers implement mortar methods for face coupling
- DGKernel automatically applies mortar projections at partition interfaces
- Algorithm developers write physics without considering element transitions

For example, a mesh with tetrahedra in the boundary layer and hexahedra in the far field appears uniform to the algorithm developer - the mortar coupling happens transparently through the face buffer system.

### Transparency, Not Opacity

While DGKernel automates common patterns, it maintains a philosophy of transparency:

**Nothing is hidden:**

- Face mappings remain accessible for custom flux functions
- Operator implementations can be extended or replaced
- Memory layouts are documented and accessible
- Mortar coupling can be customized when needed

**Progressive complexity:**

- Default behavior handles most cases automatically
- Advanced users can access lower-level components
- Custom operators can be added alongside standard ones
- The system provides convenience without preventing control

This approach means researchers can start with simple, element-agnostic algorithms and progressively add sophistication only where their specific physics requires it.

### OCCA as the Foundation

DGKernel builds on OCCA’s kernel language, which provides:

- Portable kernel syntax with @outer/@inner annotations
- Automatic translation to CUDA, OpenMP, OpenCL, etc.
- Just-in-time compilation with architecture-specific optimizations

This means DGKernel users write kernels once using OCCA syntax, and the system handles all architecture-specific details.

### Explicit Over Implicit

DGKernel makes key aspects explicit:

- Data dependencies between operations
- Memory allocation and lifetime
- Communication patterns

This explicitness enables optimization while maintaining clarity about the numerical method being implemented, whether it’s:

- Strong form collocation (pointwise enforcement of PDEs)
- Weak form Galerkin methods (integral formulations)
- Spectral methods (modal basis functions)
- Flux reconstruction approaches
- Hybridizable methods

The same infrastructure supports all these approaches because they share common computational patterns.

### Composition Over Monolithic Solutions

Rather than providing complete solvers, DGKernel provides granular building blocks:

- **Element-provided operators**: Gradient, Divergence, Mass, LIFT, etc.
- **Physics-specific functions**: Flux computation, equation of state, etc.
- **OCCA kernel code**: Combines operators to implement algorithms
- **Reusable patterns**: The same operators work across different physics

This compositional approach means new physics solvers are developed by writing OCCA kernels that combine existing operators in new ways, without reimplementing low-level numerics.

### Performance Through Structure

High performance comes from exploiting structure:

- Regular memory access patterns
- Predictable communication
- Amortized setup costs

The framework identifies and exploits these patterns automatically.

## Limitations and Scope

DGKernel’s focused model works within OCCA’s constraints:

**Computational model:**

- Partitions perform local computations independently
- Communication occurs in consolidated steps between partitions
- Various data types (faces, vertices, edges) can be exchanged
- The partition buffer system manages all boundary data transfers

**Within Scope:**

- Methods where most computation is partition-local
- Algorithms that fit the @outer/@inner nested loop structure
- Coupling through partition boundaries (faces, vertices, edges)
- Multigrid methods (transfers within partition structure)
- Mixed element meshes (each partition has a fixed element type)

**Outside Scope:**

- Algorithms requiring different parallel decompositions
- Methods that can’t use OCCA’s loop restrictions
- Dynamic parallelism or recursive algorithms
- Kernels needing variable loop bounds within iterations

These constraints enable DGKernel to automate common patterns while working within OCCA’s portable parallel model.

## Book Organization

This book presents DGKernel from the ground up:

**Part I: Foundations**

- Chapter 1: Element Definition - Mathematical building blocks
- Chapter 2: Operators - Computational patterns
- Chapter 3: Meshes - Connecting elements to domains

**Part II: Parallel Execution**

- Chapter 4: Partitioning - Domain decomposition strategies
- Chapter 5: Fields - Data management and I/O
- Chapter 6: Kernel Building - Code generation process

**Part III: Applications**

- Chapter 7: Methods - Complete numerical schemes
- Appendices A-D: Detailed examples from various physics domains

Each chapter builds on previous concepts while maintaining independence where possible.

## A Working Example

To illustrate the complete pipeline:

```go
// 1. Element designer provides granular operators
element := Tetrahedron{Order: 3}
// Element implements operators like:
// - Gradient: (u) → (ur, us, ut) in reference space
// - PhysicalGradient: (u, metrics) → (ux, uy, uz) in physical space
// - Divergence: (Fx, Fy, Fz) → (divF)
// - Mass: (u) → (Mu)
// - LIFT: (faceData) → (volumeData)

// 2. Algorithm developer writes natural code
@kernel void computeNavierStokes(...) {
    for (int p = 0; p < Npartitions; ++p; @outer) {
        for (int e = 0; e < ElementsPerPartition; ++e; @inner) {
            // Natural mathematical notation - no manual indexing
            PhysicalGradient(velocity, velocity_x, velocity_y, velocity_z);
            
            // DGKernel manages:
            // - Field allocation based on operator contracts
            // - Correct strides for this partition's element type
            // - Memory layout optimization
            
            ComputeStress(velocity_x, velocity_y, velocity_z, stress);
            Divergence(stress, stress_divergence);
            LIFT(face_flux, volume_contribution);
            
            // Combine terms...
            Add(stress_divergence, volume_contribution, rhs);
        }
    }
}

// 3. Mixed element example
// Partition 0: Tetrahedra (boundary layer)
// Partition 1: Hexahedra (far field)
// Same operator names, different implementations per partition
```

The crucial point: algorithm developers use natural field names and mathematical operations. DGKernel handles all the complexity of memory management, dimensionality, and ensuring operator contracts are satisfied.

-----

*Continue to Chapter 1: Element Definition →*

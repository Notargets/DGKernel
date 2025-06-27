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

DGKernel is a kernel generation system that bridges this gap through a focused computational model. It is built on a key simplifying assumption: **problems can be decomposed into partitions that communicate only through element faces**. This discontinuous Galerkin-inspired model (hence "DG" in DGKernel) enables a clean mapping between mathematical elements and parallel hardware.

The system leverages OCCA's kernel abstraction to target both SMP and GPU architectures through a restricted but powerful parallel model:
- **@outer loops** map to partitions (thread blocks on GPU, parallel regions on CPU)
- **@inner loops** map to elements within partitions (threads on GPU, vectorized loops on CPU)

This focused approach trades some generality for significant simplification: element-based algorithms map directly to excellent machine performance without complex transformations.

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

In DGKernel, an element is a complete mathematical entity:

```go
type Element interface {
    // Basis functions and quadrature
    Np() int                          // Number of points
    N() int                           // Polynomial order
    InterpolationMatrix() []float64   // Vandermonde matrix
    
    // Differentiation
    DifferentiationMatrices() (Dr, Ds, Dt []float64)
    
    // Integration  
    Mass() []float64                  // Mass matrix
    Weights() []float64               // Quadrature weights
    
    // Geometry
    ReferenceNodes() (r, s, t []float64)
    FaceNodes() [][]int               // Node indices per face
}
```

This interface captures the mathematical structure independent of implementation details.

### Separation of Concerns

A critical design principle in DGKernel is the separation between element design and algorithm development:

**Element Designer's Role:**
- Define mathematical properties (basis functions, quadrature, etc.)
- Map these to granular operator contracts (Gradient, Divergence, Mass, LIFT)
- Each operator has clear input/output specifications
- This mapping is done once per element type

**Algorithm Developer's Role:**
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

The idea of separating numerical methods from implementation details isn't new. Starting in the 1990s, several C++ projects attempted this goal:
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

### The Face Buffer System

The Face Buffer manages communication between elements through their faces:

**Design characteristics:**
- Elements within a partition communicate through face interfaces
- Face data organized for flux computation
- Contiguous send/receive buffers for MPI communication
- Enables efficient distributed computing across multiple nodes

**Memory layout:**
- Face values stored in arrays sized by partition element count
- Send buffers packed with face data for remote partitions
- Receive buffers hold incoming face data from other partitions
- Buffer layout optimized for network communication

This design supports both local face operations within a partition and remote communication between partitions, providing the foundation for scalable parallel execution.

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

The framework uses OCCA's two-level parallelization model:

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

DGKernel focuses on structuring computations to work well with OCCA's model, while OCCA handles all hardware-specific translation.

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

DGKernel builds on OCCA's kernel language, which provides:
- Portable kernel syntax with @outer/@inner annotations
- Automatic translation to CUDA, OpenMP, OpenCL, etc.
- Just-in-time compilation with architecture-specific optimizations

This means DGKernel users write kernels once using OCCA syntax, and the system handles all architecture-specific details.

### Explicit Over Implicit

DGKernel makes key aspects explicit:
- Data dependencies between operations
- Memory allocation and lifetime
- Communication patterns

This explicitness enables optimization while maintaining clarity about the numerical method being implemented, whether it's:
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

DGKernel's focused model has clear boundaries:

**Within Scope:**
- Methods where elements communicate only through faces
- Algorithms that fit the @outer (partition) / @inner (element) parallel model
- Local operations within elements
- Face-based coupling between elements
- Multigrid methods (injection/prolongation within partitions)
- Mixed element meshes (each partition has a fixed element type)

**Outside Scope:**
- Methods requiring arbitrary element-to-element communication beyond faces
- Algorithms that don't map to the partition/element parallel structure
- Global assembly operations that can't be decomposed

These constraints enable DGKernel's key capability: automatic management of operator inputs and outputs based on adherence to operator contracts.

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

// 2. Algorithm developer writes OCCA kernel
@kernel void computeBurgers(...) {
    for (int p = 0; p < Npartitions; ++p; @outer) {
        // Partition-level operators
        PhysicalGradient(u, ux, uy, uz, K[part]);
        ComputeBurgersFlux(faceValues, faceFlux, K[part]);
        MATMUL_LIFT(faceFlux, lifted, K[part]);
        
        // Element-level physics
        for (int elem = 0; elem < KpartMax; ++elem; @inner) {
            if (elem < K[part]) {
                // Update residual (RK stage)
                for (int n = 0; n < NP_TET; ++n) {
                    int id = elem * NP_TET + n;
                    resu[id] = a * u[id] + resu[id];
                }
                
                // Apply physics: div(u²/2)
                for (int n = 0; n < NP_TET; ++n) {
                    int id = elem * NP_TET + n;
                    real_t divFlux = -u[id] * (ux[id] + uy[id] + uz[id]);
                    u[id] = resu[id] + b_dt * (divFlux + lifted[id]);
                }
            }
        }
    }
}

// 3. Key points:
// - Operators like PhysicalGradient work on entire partitions
// - Element loops handle local physics
// - DGKernel manages array dimensions based on element type
// - Mixed meshes: each partition can have different NP values
```

The structure shows how operators handle the heavy lifting at partition level, while element loops implement the specific physics.

---

*Continue to Chapter 1: Element Definition →*

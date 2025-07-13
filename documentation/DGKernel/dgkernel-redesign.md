# DGKernel V2 Implementation Design Document

## Executive Summary

This document describes the refactoring of DGKernel from its current single-kernel-focused API to a multi-phase API that supports persistent device state across multiple kernel executions. The refactor preserves all existing core functionality while providing cleaner separation of concerns and better support for iterative simulations.

## Architecture Overview

### Current State
- **Single kernel focus**: Memory allocation, parameter binding, and execution are tightly coupled
- **Implicit memory management**: Device memory is allocated during kernel definition
- **Limited reusability**: Parameters must be redefined for each kernel

### Target State
- **Multi-phase lifecycle**: Clear separation between device setup, binding definition, and kernel execution
- **Explicit memory management**: Device memory persists across kernel executions
- **Flexible parameter usage**: Bindings are defined once, used multiple times with different actions

## Implementation Phases

### Phase 0: Device Setup
```go
device := gocca.CreateDevice(...)
runner := runner.NewRunner(device, builder.Config{K: []int{...}})
```

### Phase 1: Prep Phase (One-time Setup)
```go
// Define bindings - establishes host↔device relationships
runner.DefineBindings(
    builder.Input("U").Bind(hostU),
    builder.Input("V").Bind(hostV),
    builder.Output("RHS").Bind(hostRHS),
    builder.Temp("work").Type(builder.Float64).Size(totalNodes),
    builder.Input("Dr").Bind(drMatrix).ToMatrix(),
    builder.Input("Mass").Bind(massMatrix).ToMatrix().Static(),
)

// Allocate all device memory and prepare kernel infrastructure
runner.AllocateDevice()
```

### Phase 2: Execute Kernel Phase (Repeatable)
```go
// Configure parameter actions for this specific kernel
params := runner.ConfigureKernel("volumeKernel",
    runner.Param("U").CopyTo(),
    runner.Param("V"),           // No action - use existing device data
    runner.Param("RHS").CopyBack(),
    runner.Param("work"),        // Temp arrays available
)

// Build kernel with configured parameters
signature := params.GetSignature()
kernelSource := fmt.Sprintf(`...kernel code...`, signature)
runner.BuildKernel(kernelSource, "volumeKernel")

// Execute with automatic memory operations
runner.ExecuteKernel("volumeKernel")

// Manual memory operations also available
runner.CopyToDevice("U")
runner.CopyFromDevice("RHS")
```

## Core Components

### 1. DeviceBinding Structure
```go
type DeviceBinding struct {
    Name         string
    HostBinding  interface{}      // Original host data reference ([]T or mat.Matrix)
    HostType     builder.DataType // Element type in host data
    DeviceType   builder.DataType // Element type on device (may differ from HostType)
    IsMatrix     bool            // Host data is mat.Matrix
    IsStatic     bool
    IsPartitioned bool
    // ... other metadata
}
```

**Implicit Type Conversions**: When `HostType` differs from `DeviceType`, conversions happen automatically during `CopyTo()` and `CopyBack()` operations. For example:
- Host: `[]float64` or `mat.Matrix` (float64 elements)
- Device: float32 array
- `CopyTo()`: float64 → float32 conversion
- `CopyBack()`: float32 → float64 conversion

### 2. KernelConfig Structure
```go
type KernelConfig struct {
    Name       string
    Parameters []ParameterUsage
}

type CopyConfig struct {
    Parameters []ParameterUsage
}

type ParameterUsage struct {
    Binding    *DeviceBinding
    Actions    ActionFlags  // CopyTo, CopyBack, etc.
}
```

### 3. Runner Structure Updates
```go
type Runner struct {
    *builder.Builder
    Device            *gocca.OCCADevice
    
    // Phase 1 state
    Bindings          map[string]*DeviceBinding
    PooledMemory      map[string]*gocca.OCCAMemory
    IsAllocated       bool
    
    // Phase 2 state
    KernelConfigs     map[string]*KernelConfig
    Kernels           map[string]*gocca.OCCAKernel
}
```

## Key Methods to Implement

### Phase 1: Binding Definition

#### DefineBindings
- **Purpose**: Define all host↔device bindings
- **Uses existing**: `builder.ParamBuilder` for parameter specification
- **New behavior**: Stores bindings without allocation
- **Implicit mappings**:
  - `[]float64` → device array of float64 (or float32 if `.Convert()` was used)
  - `mat.Matrix` → device array with automatic row→column major transformation
  - Host and device types are determined at binding time, not execution time

#### AllocateDevice
- **Purpose**: Allocate all device memory and compute offsets
- **Uses existing**: 
  - `allocateSingleArray()` for array allocation
  - `AllocateDeviceMatrices()` for matrix allocation
  - `CalculateAlignedOffsetsAndSize()` for offset computation
- **New behavior**: Operates on stored bindings, sets `IsAllocated` flag

### Phase 2: Kernel Configuration

#### ConfigureKernel
- **Purpose**: Create kernel-specific parameter configuration
- **New method**: Returns a `KernelConfig` with action overlays

#### Param
- **Purpose**: Reference a binding for kernel-specific configuration
- **New method**: Returns a lightweight action configurator

#### BuildKernel
- **Uses existing**: Current `BuildKernel` implementation
- **Modified**: Accepts `KernelConfig` instead of relying on `kernelDefinitions`

#### ExecuteKernel
- **Purpose**: Execute kernel with automatic memory operations
- **Uses existing**:
  - `performPreKernelCopies()` for host→device transfers
  - `buildKernelArguments()` for argument preparation
  - `performPostKernelCopies()` for device→host transfers
- **Modified**: Operates on `KernelConfig` instead of `KernelDefinition`

### Memory Operations

#### ConfigureCopy
- **Purpose**: Configure memory operations without kernel execution
- **New method**: Similar to `ConfigureKernel` but for standalone copies
- **Returns**: A `CopyConfig` object with configured actions

#### ExecuteCopy
- **Purpose**: Execute configured memory operations
- **Uses existing**:
  - `copyToDeviceWithConversion()`
  - `copyFromDeviceWithConversion()`
- **New behavior**: Executes all configured copies, then clears action state

#### Example:
```go
// Configure multiple copy operations
copyOp := runner.ConfigureCopy(
    runner.Param("U").CopyTo(),
    runner.Param("V").CopyTo(),
    runner.Param("RHS").CopyBack(),
)

// Execute all configured copies
runner.ExecuteCopy(copyOp)

// Alternative: Individual operations
runner.CopyToDevice("U")     // Simple single-parameter copy
runner.CopyFromDevice("RHS")  // Simple single-parameter copy
```

## Migration Strategy

### What Stays
- `builder/` package remains unchanged
- Core allocation logic (`CalculateAlignedOffsetsAndSize`, etc.)
- Memory copy infrastructure
- Kernel compilation and execution
- Type conversion system

### What Changes
- `DefineKernel` → `DefineBindings` + `ConfigureKernel`
- `KernelDefinition` → `DeviceBinding` + `KernelConfig`
- Parameter binding happens once, not per kernel

### What Gets Deleted
- `KernelDefinition` struct (replaced by split structures)
- Per-kernel memory allocation logic
- Tight coupling between definition and execution

## Directory Structure

```
runner/
├── runner.go              # Core Runner struct and lifecycle methods
├── binding.go             # NEW: DeviceBinding and DefineBindings
├── kernel_config.go       # NEW: KernelConfig and ConfigureKernel
├── kernel_execution.go    # Modified from kernel_definition.go
├── memory_operations.go   # NEW: Manual memory operation methods
├── kernel_copy.go         # Unchanged: Copy infrastructure
├── builder/              # Unchanged: Parameter builders
│   ├── builder.go
│   ├── builder_param.go
│   └── ...
└── tests/                # Updated tests for new API
```

## Refactoring Opportunities

### 1. Simplify Parameter Tracking
Current system tracks parameters in multiple places. New design centralizes in `Bindings` map.

### 2. Reduce Allocation Checks
Current `DefineKernel` checks if arrays are allocated on every call. New design allocates once.

### 3. Cleaner Kernel Arguments
Current `buildKernelArguments` is complex due to mixed concerns. New design separates binding from usage.

### 4. Explicit State Management
Current system has implicit state transitions. New design has explicit phases with clear invariants.

## Type System and Implicit Conversions

### Host to Device Mappings

The system supports flexible host data types that map to simpler device representations:

| Host Type | Device Type | Implicit Operations |
|-----------|-------------|-------------------|
| `[]float64` | float64/float32 array | Direct copy or type conversion |
| `[]float32` | float32/float64 array | Direct copy or type conversion |
| `mat.Matrix` | float64/float32 array | Row→column major transform + optional type conversion |
| `[][]float64` | Partitioned array | Per-partition copy with offsets |
| `[]mat.Matrix` | Partitioned array | Per-partition transform + copy |

### Binding Attributes and Actions

| Binding Attribute | Set By | Host→Device Action | Device→Host Action |
|-------------------|--------|-------------------|-------------------|
| **Data Layout** | | | |
| Row-major (mat.Matrix) | Implicit from type | Transpose to column-major | Transpose to row-major |
| Flat array ([]T) | Implicit from type | Direct copy | Direct copy |
| Partitioned ([][]T) | Implicit from type | Copy to computed offsets | Copy from computed offsets |
| **Type Conversion** | | | |
| Host type | Implicit from binding | - | Convert to host type |
| Device type | `.Convert()` or default | Convert to device type | - |
| **Memory Location** | | | |
| Static matrix | `.Static()` | Embed in kernel (no copy) | N/A |
| Device matrix | `.ToMatrix()` | One-time copy at allocation | No automatic copy |
| Regular array | Default | As configured | As configured |
| **Copy Behavior** | | | |
| CopyTo | `.CopyTo()` | Execute before kernel | - |
| CopyBack | `.CopyBack()` | - | Execute after kernel |
| Copy | `.Copy()` | Execute before kernel | Execute after kernel |
| NoCopy | `.NoCopy()` or default | No action | No action |

### Matrix Storage Transform Example

```go
// Host: mat.Matrix (row-major)
// [[1, 2, 3],
//  [4, 5, 6],
//  [7, 8, 9]]
// Stored as: [1, 2, 3, 4, 5, 6, 7, 8, 9]

// Device: column-major array
// Stored as: [1, 4, 7, 2, 5, 8, 3, 6, 9]
// This allows efficient column access in kernels
```

### Conversion Flow

1. **At Binding Time**: The relationship between host and device types is established
   ```go
   builder.Input("U").Bind(hostMatrix)           // mat.Matrix → float64 array
   builder.Input("V").Bind(hostArray).Convert(builder.Float32) // []float64 → float32 array
   ```

2. **At Copy Time**: Conversions happen automatically
   - `CopyTo()`: Performs any necessary type conversion and layout transformation
   - `CopyBack()`: Reverses the transformation and conversion

3. **No Explicit `.Convert()` in Phase 2**: Since types are determined at binding time, kernel configuration only specifies actions:
   ```go
   runner.Param("U").CopyTo()    // Implicitly uses binding's conversion rules
   ```

## Example Usage Pattern

```go
// Setup phase (once)
runner := runner.NewRunner(device, config)
defer runner.Free()

runner.DefineBindings(
    builder.Input("Q").Bind(Q),
    builder.Input("Qold").Bind(Qold),
    builder.Output("RHS").Bind(RHS),
    builder.Input("Dr").Bind(Dr).ToMatrix(),
    builder.Temp("flux").Type(builder.Float64).Size(totalFluxNodes),
)
runner.AllocateDevice()

// Simulation loop
for step := 0; step < nSteps; step++ {
    // Volume kernel
    vol := runner.ConfigureKernel("volume",
        runner.Param("Q").CopyTo(),
        runner.Param("RHS").CopyBack(),
    )
    runner.BuildKernel(volKernelSrc, "volume")
    runner.ExecuteKernel("volume")
    
    // Surface kernel - reuses device data
    surf := runner.ConfigureKernel("surface", 
        runner.Param("Q"),          // Already on device
        runner.Param("flux"),       // Temp array
        runner.Param("RHS"),        // Update in place
    )
    runner.BuildKernel(surfKernelSrc, "surface")
    runner.ExecuteKernel("surface")
    
    // Manual copy operation - no kernel
    if step % 10 == 0 {
        // Save checkpoint
        checkpoint := runner.ConfigureCopy(
            runner.Param("Q").CopyBack(),
            runner.Param("Qold").CopyBack(),
        )
        runner.ExecuteCopy(checkpoint)
        saveCheckpoint(Q, Qold)
    }
    
    // Update kernel
    update := runner.ConfigureKernel("update",
        runner.Param("Q").CopyBack(),    // Get final result
        runner.Param("Qold").CopyTo(),   // Save previous state
        runner.Param("RHS"),
    )
    runner.BuildKernel(updateKernelSrc, "update")
    runner.ExecuteKernel("update")
}
```

## Testing Strategy

1. **Compatibility Tests**: Ensure existing single-kernel patterns still work
2. **Multi-Kernel Tests**: Verify state persistence across kernel executions
3. **Memory State Tests**: Verify CopyTo/CopyBack behavior with state tracking
4. **Performance Tests**: Ensure refactoring doesn't introduce overhead

## Implementation Order

1. Implement `DeviceBinding` and `DefineBindings`
2. Refactor allocation to work with bindings
3. Implement `KernelConfig` and `ConfigureKernel`
4. Update execution path to use configurations
5. Add manual memory operations
6. Update tests
7. Add migration guide for users

## Code Reuse and Simplification Strategy

### Unified Copy Infrastructure

All copy operations should flow through a single, shared implementation:

```go
// Core copy engine - used by ALL copy operations
func (kr *Runner) executeCopyActions(actions []ParameterUsage) error {
    for _, param := range actions {
        if param.Actions&CopyTo != 0 {
            if err := kr.copyToDeviceWithConversion(param.Binding); err != nil {
                return err
            }
        }
        if param.Actions&CopyBack != 0 {
            if err := kr.copyFromDeviceWithConversion(param.Binding); err != nil {
                return err
            }
        }
    }
    return nil
}

// Thin wrappers around the core engine
func (kr *Runner) ExecuteKernel(name string) error {
    config := kr.KernelConfigs[name]
    
    // Pre-kernel copies
    if err := kr.executeCopyActions(config.Parameters); err != nil {
        return err
    }
    
    // Run kernel...
    
    // Post-kernel copies (same infrastructure)
    return kr.executeCopyActions(config.Parameters)
}

func (kr *Runner) ExecuteCopy(config CopyConfig) error {
    return kr.executeCopyActions(config.Parameters)
}

func (kr *Runner) CopyToDevice(name string) error {
    return kr.executeCopyActions([]ParameterUsage{
        {Binding: kr.Bindings[name], Actions: CopyTo},
    })
}
```

### Simplification Opportunities

1. **Single Copy Path**: Unlike current system with `performPreKernelCopies` and `performPostKernelCopies`, use one unified method
2. **No Duplicate Logic**: `ExecuteKernel`, `ExecuteCopy`, and manual operations all use the same underlying infrastructure
3. **Cleaner Parameter Resolution**: Bindings are resolved once, not repeatedly
4. **Simpler State Management**: Action flags are clearer than current implicit state

This approach ensures maximum code reuse while keeping the implementation clean and maintainable.

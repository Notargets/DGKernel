// File: runner/binding.go
// Phase 1: Foundation - Define DeviceBinding structure and ActionFlags

package runner

import (
	"github.com/notargets/DGKernel/runner/builder"
)

// ActionFlags represents the memory operations to perform for a parameter
type ActionFlags int

const (
	// No action
	NoAction ActionFlags = 0
	// Copy from host to device before kernel execution
	CopyTo ActionFlags = 1 << iota
	// Copy from device to host after kernel execution
	CopyBack
	// Bidirectional copy (CopyTo | CopyBack)
	Copy = CopyTo | CopyBack
)

// DeviceBinding represents a host↔device data binding
// This structure captures all metadata about a parameter's memory relationship
type DeviceBinding struct {
	// Parameter identification
	Name string

	// Host data reference - original binding from user
	HostBinding interface{} // []T, [][]T, mat.Matrix, []mat.Matrix, or scalar

	// Type information
	HostType   builder.DataType // Element type in host data
	DeviceType builder.DataType // Element type on device (may differ for conversions)

	// Size information
	Size        int64 // Total number of elements
	ElementSize int   // Size of each element in bytes on device

	// Data layout flags
	IsMatrix      bool // Host data is mat.Matrix or []mat.Matrix
	IsStatic      bool // Matrix embedded as static const in kernel
	IsPartitioned bool // Data is partitioned ([][]T or []mat.Matrix)
	IsScalar      bool // Single value parameter
	IsTemp        bool // Device-only temporary array

	// Matrix-specific metadata
	MatrixRows int
	MatrixCols int
	Stride     int // For flat array to matrix promotion

	// Partitioned data metadata
	PartitionCount int
	PartitionSizes []int // Size of each partition

	// Memory attributes
	Alignment builder.AlignmentType
	IsOutput  bool // Whether parameter can be written to in kernel

	// Original parameter specification (for compatibility during migration)
	ParamSpec *builder.ParamSpec
}

// ParameterUsage represents how a binding is used in a specific kernel or copy operation
type ParameterUsage struct {
	Binding *DeviceBinding
	Actions ActionFlags
}

// HasAction checks if a specific action is set
func (pu *ParameterUsage) HasAction(action ActionFlags) bool {
	return pu.Actions&action != 0
}

// NeedsCopyTo returns true if this usage requires host→device copy
func (pu *ParameterUsage) NeedsCopyTo() bool {
	return pu.HasAction(CopyTo)
}

// NeedsCopyBack returns true if this usage requires device→host copy
func (pu *ParameterUsage) NeedsCopyBack() bool {
	return pu.HasAction(CopyBack)
}

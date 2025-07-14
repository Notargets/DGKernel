// File: runner/binding.go
// Phase 1: Foundation - Define DeviceBinding structure and ActionFlags
// Phase 2: Binding System - Implement DefineBindings and helper methods

package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"gonum.org/v1/gonum/mat"
	"reflect"
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

// DefineBindings establishes host↔device data relationships
// This is Phase 1 of the new API - define all bindings once
func (kr *Runner) DefineBindings(params ...*builder.ParamBuilder) error {
	if kr.IsAllocated {
		return fmt.Errorf("bindings cannot be defined after AllocateDevice has been called")
	}

	// Extract and validate parameter specifications
	for i, p := range params {
		spec := p.Spec
		if err := spec.Validate(); err != nil {
			return fmt.Errorf("parameter %d: %w", i, err)
		}

		// Create DeviceBinding from ParamSpec
		binding, err := kr.createBindingFromParam(&spec)
		if err != nil {
			return fmt.Errorf("failed to create binding for %s: %w", spec.Name, err)
		}

		// Store the binding
		kr.Bindings[spec.Name] = binding
	}

	return nil
}

// createBindingFromParam converts a ParamSpec into a DeviceBinding
// This extracts and refactors logic from DefineKernel
func (kr *Runner) createBindingFromParam(spec *builder.ParamSpec) (*DeviceBinding, error) {
	binding := &DeviceBinding{
		Name:        spec.Name,
		HostBinding: spec.HostBinding,
		ParamSpec:   spec,
		Alignment:   spec.Alignment,
		IsOutput:    !spec.IsConst(),
	}

	// Set direction-based flags
	switch spec.Direction {
	case builder.DirectionScalar:
		binding.IsScalar = true
		binding.HostType = spec.DataType
		binding.DeviceType = spec.DataType // Scalars don't convert
		binding.Size = 1
		binding.ElementSize = int(SizeOfType(binding.DeviceType))
		return binding, nil

	case builder.DirectionTemp:
		binding.IsTemp = true
		binding.DeviceType = spec.GetEffectiveType()
		binding.Size = spec.Size
		binding.ElementSize = int(SizeOfType(binding.DeviceType))
		return binding, nil
	}

	// Handle matrix flags
	binding.IsMatrix = spec.IsMatrix
	binding.IsStatic = spec.IsStatic
	binding.MatrixRows = spec.MatrixRows
	binding.MatrixCols = spec.MatrixCols
	binding.Stride = spec.Stride

	// Handle partitioned data
	binding.IsPartitioned = spec.IsPartitioned
	binding.PartitionCount = spec.PartitionCount

	// Infer types from host binding
	if spec.HostBinding != nil {
		if err := kr.inferBindingTypes(binding, spec); err != nil {
			return nil, err
		}
	} else {
		// No host binding - use spec types directly
		binding.HostType = spec.DataType
		binding.DeviceType = spec.GetEffectiveType()
		binding.Size = spec.Size
	}

	// Set element size based on device type
	binding.ElementSize = int(SizeOfType(binding.DeviceType))

	// Validate partitioned data constraints
	if binding.IsPartitioned {
		if !kr.IsPartitioned {
			return nil, fmt.Errorf("partitioned data %s provided to non-partitioned kernel", spec.Name)
		}
		if binding.PartitionCount != kr.NumPartitions {
			return nil, fmt.Errorf("partition count mismatch for %s: expected %d, got %d",
				spec.Name, kr.NumPartitions, binding.PartitionCount)
		}
	} else if kr.IsPartitioned && !binding.IsMatrix && !binding.IsScalar && !binding.IsTemp {
		return nil, fmt.Errorf("non-partitioned array %s provided to partitioned kernel", spec.Name)
	}

	return binding, nil
}

// inferBindingTypes determines host and device types from the host binding
func (kr *Runner) inferBindingTypes(binding *DeviceBinding, spec *builder.ParamSpec) error {
	v := reflect.ValueOf(binding.HostBinding)
	t := v.Type()

	// Handle partitioned arrays ([][]T)
	if t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Slice {
		binding.IsPartitioned = true
		binding.PartitionCount = v.Len()

		// Calculate total size and get element type
		totalSize := int64(0)
		binding.PartitionSizes = make([]int, binding.PartitionCount)

		for i := 0; i < v.Len(); i++ {
			partition := v.Index(i)
			partSize := partition.Len()
			binding.PartitionSizes[i] = partSize
			totalSize += int64(partSize)
		}

		binding.Size = totalSize

		// Infer element type from first partition
		if v.Len() > 0 {
			elemType := t.Elem().Elem()
			binding.HostType = getDataTypeFromReflectKind(elemType.Kind())
		}

		// Device type may differ if conversion specified
		binding.DeviceType = spec.GetEffectiveType()
		if binding.DeviceType == 0 {
			binding.DeviceType = binding.HostType
		}

		return nil
	}

	// Handle partitioned matrices ([]mat.Matrix)
	if t.Kind() == reflect.Slice && v.Len() > 0 {
		if _, ok := v.Index(0).Interface().(mat.Matrix); ok {
			binding.IsPartitioned = true
			binding.IsMatrix = true
			binding.PartitionCount = v.Len()

			// Calculate total size
			totalSize := int64(0)
			for i := 0; i < v.Len(); i++ {
				if m, ok := v.Index(i).Interface().(mat.Matrix); ok {
					rows, cols := m.Dims()
					if i == 0 {
						binding.MatrixRows = rows
						binding.MatrixCols = cols
					}
					totalSize += int64(rows * cols)
				}
			}

			binding.Size = totalSize
			binding.HostType = builder.Float64 // gonum matrices are float64
			binding.DeviceType = spec.GetEffectiveType()
			if binding.DeviceType == 0 {
				binding.DeviceType = binding.HostType
			}

			return nil
		}
	}

	// Handle single matrix
	if m, ok := binding.HostBinding.(mat.Matrix); ok {
		rows, cols := m.Dims()
		binding.Size = int64(rows * cols)
		binding.HostType = builder.Float64 // gonum matrices are float64
		binding.MatrixRows = rows
		binding.MatrixCols = cols

		// IMPORTANT: Only set IsMatrix if the spec says so
		// This respects whether .ToMatrix() was called
		binding.IsMatrix = spec.IsMatrix
		binding.IsStatic = spec.IsStatic

		binding.DeviceType = spec.GetEffectiveType()
		if binding.DeviceType == 0 {
			binding.DeviceType = binding.HostType
		}

		return nil
	}

	// Handle regular slices
	if t.Kind() == reflect.Slice {
		binding.Size = int64(v.Len())

		// Infer element type
		elemType := t.Elem()
		binding.HostType = getDataTypeFromReflectKind(elemType.Kind())

		binding.DeviceType = spec.GetEffectiveType()
		if binding.DeviceType == 0 {
			binding.DeviceType = binding.HostType
		}

		return nil
	}

	// Handle scalars
	binding.HostType = getDataTypeFromReflectKind(t.Kind())
	binding.DeviceType = binding.HostType // Scalars don't convert
	binding.Size = 1

	return nil
}

// getDataTypeFromReflectKind converts reflect.Kind to builder.DataType
func getDataTypeFromReflectKind(kind reflect.Kind) builder.DataType {
	switch kind {
	case reflect.Float32:
		return builder.Float32
	case reflect.Float64:
		return builder.Float64
	case reflect.Int32:
		return builder.INT32
	case reflect.Int, reflect.Int64:
		return builder.INT64
	default:
		return builder.Float64 // Default
	}
}

// GetBinding returns a binding by name
func (kr *Runner) GetBinding(name string) *DeviceBinding {
	return kr.Bindings[name]
}

// HasBinding checks if a binding exists
func (kr *Runner) HasBinding(name string) bool {
	_, exists := kr.Bindings[name]
	return exists
}

// AllocateDevice allocates device memory for all defined bindings
// This is Phase 2 of the new API - allocate memory once
func (kr *Runner) AllocateDevice() error {
	if kr.IsAllocated {
		return fmt.Errorf("device memory already allocated")
	}

	if len(kr.Bindings) == 0 {
		return fmt.Errorf("no bindings defined - call DefineBindings first")
	}

	// Process each binding
	for name, binding := range kr.Bindings {
		// Skip scalars - they don't need device allocation
		if binding.IsScalar {
			continue
		}

		// Skip already allocated arrays (for compatibility during migration)
		if _, exists := kr.PooledMemory[name+"_global"]; exists {
			continue
		}

		// Allocate based on binding type
		if binding.IsMatrix && binding.IsStatic {
			if err := kr.addStaticMatrixFromBinding(binding); err != nil {
				return fmt.Errorf("failed to add static matrix %s: %w", name, err)
			}
		} else if binding.IsMatrix && !binding.IsStatic {
			if err := kr.addDeviceMatrixFromBinding(binding); err != nil {
				return fmt.Errorf("failed to add device matrix %s: %w", name, err)
			}
		} else {
			// Regular array or temp array
			if err := kr.allocateArrayFromBinding(binding); err != nil {
				return fmt.Errorf("failed to allocate array %s: %w", name, err)
			}
		}
	}

	// Allocate all device matrices that were added
	if err := kr.AllocateDeviceMatrices(); err != nil {
		return fmt.Errorf("failed to allocate device matrices: %w", err)
	}

	kr.IsAllocated = true
	return nil
}

// allocateArrayFromBinding allocates an array from DeviceBinding
func (kr *Runner) allocateArrayFromBinding(binding *DeviceBinding) error {
	// Create array spec from binding
	arraySpec := builder.ArraySpec{
		Name:      binding.Name,
		Size:      binding.Size * int64(binding.ElementSize),
		DataType:  binding.DeviceType,
		Alignment: binding.Alignment,
		IsOutput:  binding.IsOutput,
	}

	// Store binding in host bindings map for compatibility
	if binding.HostBinding != nil {
		if kr.hostBindings == nil {
			kr.hostBindings = make(map[string]interface{})
		}
		kr.hostBindings[binding.Name] = binding.HostBinding
	}

	// Allocate using existing infrastructure
	if err := kr.allocateSingleArray(arraySpec, binding.ParamSpec); err != nil {
		return err
	}

	// Store metadata
	kr.arrayMetadata[binding.Name] = ArrayMetadata{
		spec:      arraySpec,
		dataType:  binding.DeviceType,
		isOutput:  binding.IsOutput,
		paramSpec: binding.ParamSpec,
	}

	return nil
}

// addStaticMatrixFromBinding adds a static matrix from DeviceBinding
func (kr *Runner) addStaticMatrixFromBinding(binding *DeviceBinding) error {
	var matrix mat.Matrix

	switch data := binding.HostBinding.(type) {
	case mat.Matrix:
		matrix = data
	case []float64:
		if binding.Stride > 0 {
			matrix = mat.NewDense(len(data)/binding.Stride, binding.Stride, data)
		} else {
			return fmt.Errorf("flat array requires stride for matrix conversion")
		}
	default:
		return fmt.Errorf("invalid matrix binding type: %T", binding.HostBinding)
	}

	kr.Builder.AddStaticMatrix(binding.Name, matrix)

	// Store metadata
	rows, cols := matrix.Dims()
	meta := ArrayMetadata{
		spec: builder.ArraySpec{
			Name:     binding.Name,
			DataType: binding.DeviceType,
			Size:     int64(rows * cols * binding.ElementSize),
		},
		dataType:  binding.DeviceType,
		paramSpec: binding.ParamSpec,
	}
	kr.arrayMetadata[binding.Name] = meta

	return nil
}

// addDeviceMatrixFromBinding adds a device matrix from DeviceBinding
func (kr *Runner) addDeviceMatrixFromBinding(binding *DeviceBinding) error {
	var matrix mat.Matrix

	switch data := binding.HostBinding.(type) {
	case mat.Matrix:
		matrix = data
	case []float64:
		if binding.Stride > 0 {
			matrix = mat.NewDense(len(data)/binding.Stride, binding.Stride, data)
		} else {
			return fmt.Errorf("flat array requires stride for matrix conversion")
		}
	default:
		return fmt.Errorf("invalid matrix binding type: %T", binding.HostBinding)
	}

	kr.Builder.AddDeviceMatrix(binding.Name, matrix)

	// Store metadata
	rows, cols := matrix.Dims()
	meta := ArrayMetadata{
		spec: builder.ArraySpec{
			Name:     binding.Name,
			DataType: binding.DeviceType,
			Size:     int64(rows * cols * binding.ElementSize),
		},
		dataType:  binding.DeviceType,
		paramSpec: binding.ParamSpec,
	}
	kr.arrayMetadata[binding.Name] = meta

	return nil
}

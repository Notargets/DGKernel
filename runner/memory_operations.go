// File: runner/memory_operations.go
// Phase 4: Unified Copy Infrastructure - Single copy execution path

package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/gocca"
	"gonum.org/v1/gonum/mat"
	"unsafe"
)

// executeCopyActions is the core copy engine used by ALL copy operations
// This provides a single, unified path for all memory operations
func (kr *Runner) executeCopyActions(actions []ParameterUsage) error {
	for _, param := range actions {
		// Skip if no actions needed
		if param.Actions == NoAction {
			continue
		}

		// Perform host→device copy if requested
		if param.HasAction(CopyTo) {
			if err := kr.copyToDeviceFromBinding(param.Binding); err != nil {
				return fmt.Errorf("failed to copy %s to device: %w", param.Binding.Name, err)
			}
		}

		// Perform device→host copy if requested
		if param.HasAction(CopyBack) {
			if err := kr.copyFromDeviceFromBinding(param.Binding); err != nil {
				return fmt.Errorf("failed to copy %s from device: %w", param.Binding.Name, err)
			}
		}
	}
	return nil
}

// copyToDeviceFromBinding performs host→device copy using DeviceBinding
func (kr *Runner) copyToDeviceFromBinding(binding *DeviceBinding) error {
	// Skip if no host binding
	if binding.HostBinding == nil {
		return nil
	}

	// Skip scalars - they're passed by value
	if binding.IsScalar {
		return nil
	}

	// Get device memory
	var mem *gocca.OCCAMemory
	if binding.IsMatrix && !binding.IsStatic {
		// Device matrices are stored without "_global" suffix
		mem = kr.PooledMemory[binding.Name]
	} else {
		// Regular arrays use GetMemory which adds "_global"
		mem = kr.GetMemory(binding.Name)
	}

	if mem == nil {
		return fmt.Errorf("no device memory allocated for %s", binding.Name)
	}

	// Handle partitioned data
	if binding.IsPartitioned {
		return kr.copyPartitionedDataToDevice(binding, mem)
	}

	// Handle matrices
	if binding.IsMatrix {
		matrix, ok := binding.HostBinding.(mat.Matrix)
		if !ok {
			return fmt.Errorf("invalid matrix binding for %s", binding.Name)
		}
		return kr.copyMatrixToDevice(matrix, mem, binding.DeviceType)
	}

	// Handle regular arrays
	return kr.copyArrayToDevice(binding, mem)
}

// copyFromDeviceFromBinding performs device→host copy using DeviceBinding
func (kr *Runner) copyFromDeviceFromBinding(binding *DeviceBinding) error {
	// Skip if no host binding
	if binding.HostBinding == nil {
		return nil
	}

	// Skip scalars
	if binding.IsScalar {
		return nil
	}

	// Get device memory
	var mem *gocca.OCCAMemory
	if binding.IsMatrix && !binding.IsStatic {
		// Device matrices are stored without "_global" suffix
		mem = kr.PooledMemory[binding.Name]
	} else {
		// Regular arrays use GetMemory which adds "_global"
		mem = kr.GetMemory(binding.Name)
	}

	if mem == nil {
		return fmt.Errorf("no device memory allocated for %s", binding.Name)
	}

	// Handle partitioned data
	if binding.IsPartitioned {
		return kr.copyPartitionedDataFromDeviceBinding(binding, mem)
	}

	// Handle matrices
	if binding.IsMatrix {
		matrix, ok := binding.HostBinding.(mat.Matrix)
		if !ok {
			return fmt.Errorf("invalid matrix binding for %s", binding.Name)
		}
		return kr.copyMatrixFromDevice(matrix, mem, binding.DeviceType)
	}

	// Handle regular arrays
	return kr.copyArrayFromDevice(binding, mem)
}

// copyArrayToDevice handles non-partitioned array copies with type conversion
func (kr *Runner) copyArrayToDevice(binding *DeviceBinding, mem *gocca.OCCAMemory) error {
	// Check if conversion is needed
	if binding.HostType == binding.DeviceType {
		// Direct copy without conversion
		return kr.copyDirectToDevice(binding.HostBinding, mem)
	}

	// Type conversion needed
	return kr.copyWithTypeConversion(binding.HostBinding, mem, binding.HostType, binding.DeviceType)
}

// copyArrayFromDevice handles non-partitioned array copies from device with type conversion
func (kr *Runner) copyArrayFromDevice(binding *DeviceBinding, mem *gocca.OCCAMemory) error {
	// Calculate total size in bytes
	totalBytes := binding.Size * int64(binding.ElementSize)

	// Check if conversion is needed
	if binding.HostType == binding.DeviceType {
		// Direct copy without conversion
		return kr.copyDirectFromDevice(binding.HostBinding, mem, totalBytes)
	}

	// Type conversion needed
	return kr.copyFromDeviceWithTypeConversion(mem, binding.HostBinding, binding.DeviceType, binding.HostType, totalBytes)
}

// copyPartitionedDataToDevice handles partitioned data copy to device
func (kr *Runner) copyPartitionedDataToDevice(binding *DeviceBinding, deviceMem *gocca.OCCAMemory) error {
	// Get partition offsets
	offsets, err := kr.readPartitionOffsets(binding.Name)
	if err != nil {
		return fmt.Errorf("failed to read partition offsets: %w", err)
	}

	// Determine if conversion is needed
	needsConversion := binding.HostType != binding.DeviceType

	// Handle different partitioned types
	switch data := binding.HostBinding.(type) {
	case [][]float64:
		return kr.copyFloat64PartitionsToDevice(data, deviceMem, offsets, needsConversion, binding.DeviceType)
	case [][]float32:
		return kr.copyFloat32PartitionsToDevice(data, deviceMem, offsets, needsConversion, binding.DeviceType)
	case [][]int32:
		return kr.copyInt32PartitionsToDevice(data, deviceMem, offsets, needsConversion, binding.DeviceType)
	case [][]int64:
		return kr.copyInt64PartitionsToDevice(data, deviceMem, offsets, needsConversion, binding.DeviceType)
	case []mat.Matrix:
		return kr.copyMatrixPartitionsToDevice(data, deviceMem, offsets, needsConversion, binding.DeviceType)
	default:
		return fmt.Errorf("unsupported partitioned type: %T", data)
	}
}

// copyPartitionedDataFromDeviceBinding handles partitioned data copy from device
func (kr *Runner) copyPartitionedDataFromDeviceBinding(binding *DeviceBinding, deviceMem *gocca.OCCAMemory) error {
	// Get partition offsets
	offsets, err := kr.readPartitionOffsets(binding.Name)
	if err != nil {
		return fmt.Errorf("failed to read partition offsets: %w", err)
	}

	// Determine if conversion is needed
	needsConversion := binding.HostType != binding.DeviceType

	// Handle different partitioned types
	switch data := binding.HostBinding.(type) {
	case [][]float64:
		return kr.copyFloat64PartitionsFromDevice(data, deviceMem, offsets, needsConversion, binding.DeviceType)
	case [][]float32:
		return kr.copyFloat32PartitionsFromDevice(data, deviceMem, offsets, needsConversion, binding.DeviceType)
	case [][]int32:
		return kr.copyInt32PartitionsFromDevice(data, deviceMem, offsets, needsConversion, binding.DeviceType)
	case [][]int64:
		return kr.copyInt64PartitionsFromDevice(data, deviceMem, offsets, needsConversion, binding.DeviceType)
	case []mat.Matrix:
		return kr.copyMatrixPartitionsFromDevice(data, deviceMem, offsets, needsConversion, binding.DeviceType)
	default:
		return fmt.Errorf("unsupported partitioned type for copy from device: %T", data)
	}
}

// Simple copy methods for single parameters

// CopyToDevice copies a single parameter from host to device
func (kr *Runner) CopyToDevice(name string) error {
	binding := kr.GetBinding(name)
	if binding == nil {
		return fmt.Errorf("binding %s not found", name)
	}

	return kr.executeCopyActions([]ParameterUsage{
		{Binding: binding, Actions: CopyTo},
	})
}

// CopyFromDevice copies a single parameter from device to host
func (kr *Runner) CopyFromDevice(name string) error {
	binding := kr.GetBinding(name)
	if binding == nil {
		return fmt.Errorf("binding %s not found", name)
	}

	return kr.executeCopyActions([]ParameterUsage{
		{Binding: binding, Actions: CopyBack},
	})
}

// copyWithTypeConversion handles type conversion during host→device copy
func (kr *Runner) copyWithTypeConversion(hostData interface{}, mem *gocca.OCCAMemory,
	hostType, deviceType builder.DataType) error {

	// Handle all conversion cases
	switch hostType {
	case builder.Float64:
		data := hostData.([]float64)
		if deviceType == builder.Float32 {
			// Convert float64 → float32
			converted := make([]float32, len(data))
			for i, v := range data {
				converted[i] = float32(v)
			}
			mem.CopyFrom(unsafe.Pointer(&converted[0]), int64(len(converted)*4))
			return nil
		}

	case builder.Float32:
		data := hostData.([]float32)
		if deviceType == builder.Float64 {
			// Convert float32 → float64
			converted := make([]float64, len(data))
			for i, v := range data {
				converted[i] = float64(v)
			}
			mem.CopyFrom(unsafe.Pointer(&converted[0]), int64(len(converted)*8))
			return nil
		}

	case builder.INT64:
		data := hostData.([]int64)
		if deviceType == builder.INT32 {
			// Convert int64 → int32
			converted := make([]int32, len(data))
			for i, v := range data {
				converted[i] = int32(v)
			}
			mem.CopyFrom(unsafe.Pointer(&converted[0]), int64(len(converted)*4))
			return nil
		}

	case builder.INT32:
		data := hostData.([]int32)
		if deviceType == builder.INT64 {
			// Convert int32 → int64
			converted := make([]int64, len(data))
			for i, v := range data {
				converted[i] = int64(v)
			}
			mem.CopyFrom(unsafe.Pointer(&converted[0]), int64(len(converted)*8))
			return nil
		}
	}

	return fmt.Errorf("unsupported type conversion: %v → %v", hostType, deviceType)
}

// copyFromDeviceWithTypeConversion handles type conversion during device→host copy
func (kr *Runner) copyFromDeviceWithTypeConversion(mem *gocca.OCCAMemory, hostData interface{},
	deviceType, hostType builder.DataType, totalBytes int64) error {

	// Handle all conversion cases
	switch deviceType {
	case builder.Float32:
		if hostType == builder.Float64 {
			// Read float32 from device, convert to float64
			data := hostData.([]float64)
			temp := make([]float32, len(data))
			mem.CopyTo(unsafe.Pointer(&temp[0]), int64(len(temp)*4))
			for i, v := range temp {
				data[i] = float64(v)
			}
			return nil
		}

	case builder.Float64:
		if hostType == builder.Float32 {
			// Read float64 from device, convert to float32
			data := hostData.([]float32)
			temp := make([]float64, len(data))
			mem.CopyTo(unsafe.Pointer(&temp[0]), int64(len(temp)*8))
			for i, v := range temp {
				data[i] = float32(v)
			}
			return nil
		}

	case builder.INT32:
		if hostType == builder.INT64 {
			// Read int32 from device, convert to int64
			data := hostData.([]int64)
			temp := make([]int32, len(data))
			mem.CopyTo(unsafe.Pointer(&temp[0]), int64(len(temp)*4))
			for i, v := range temp {
				data[i] = int64(v)
			}
			return nil
		}

	case builder.INT64:
		if hostType == builder.INT32 {
			// Read int64 from device, convert to int32
			data := hostData.([]int32)
			temp := make([]int64, len(data))
			mem.CopyTo(unsafe.Pointer(&temp[0]), int64(len(temp)*8))
			for i, v := range temp {
				data[i] = int32(v)
			}
			return nil
		}
	}

	return fmt.Errorf("unsupported type conversion: %v → %v", deviceType, hostType)
}

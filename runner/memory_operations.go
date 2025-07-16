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

// isMatrixType checks if the host binding is a mat.Matrix type
// This is used to determine copy behavior (transpose vs non-transpose)
// IMPORTANT: This is completely independent of the IsMatrix flag
func isMatrixType(hostBinding interface{}) bool {
	_, ok := hostBinding.(mat.Matrix)
	return ok
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
	// NOTE: Memory location logic uses IsMatrix flag for naming conventions only
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

	// CRITICAL: Check actual type for copy behavior, NOT IsMatrix flag
	// The IsMatrix flag is ONLY for MATMUL macro generation
	if isMatrixType(binding.HostBinding) {
		// All mat.Matrix types must go through transpose path
		matrix := binding.HostBinding.(mat.Matrix)
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
	// NOTE: Memory location logic uses IsMatrix flag for naming conventions only
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

	// CRITICAL: Check actual type for copy behavior, NOT IsMatrix flag
	// The IsMatrix flag is ONLY for MATMUL macro generation
	if isMatrixType(binding.HostBinding) {
		// All mat.Matrix types must go through transpose path
		matrix := binding.HostBinding.(mat.Matrix)
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
		switch deviceType {
		case builder.Float32:
			// Convert float64 to float32
			converted := make([]float32, len(data))
			for i, v := range data {
				converted[i] = float32(v)
			}
			mem.CopyFrom(unsafe.Pointer(&converted[0]), int64(len(converted)*4))
		case builder.Float64:
			// Same type - direct copy
			mem.CopyFrom(unsafe.Pointer(&data[0]), int64(len(data)*8))
		default:
			return fmt.Errorf("unsupported conversion from float64 to %v", deviceType)
		}

	case builder.Float32:
		data := hostData.([]float32)
		switch deviceType {
		case builder.Float64:
			// Convert float32 to float64
			converted := make([]float64, len(data))
			for i, v := range data {
				converted[i] = float64(v)
			}
			mem.CopyFrom(unsafe.Pointer(&converted[0]), int64(len(converted)*8))
		case builder.Float32:
			// Same type - direct copy
			mem.CopyFrom(unsafe.Pointer(&data[0]), int64(len(data)*4))
		default:
			return fmt.Errorf("unsupported conversion from float32 to %v", deviceType)
		}

	case builder.INT32:
		data := hostData.([]int32)
		switch deviceType {
		case builder.INT64:
			// Convert int32 to int64
			converted := make([]int64, len(data))
			for i, v := range data {
				converted[i] = int64(v)
			}
			mem.CopyFrom(unsafe.Pointer(&converted[0]), int64(len(converted)*8))
		case builder.INT32:
			// Same type - direct copy
			mem.CopyFrom(unsafe.Pointer(&data[0]), int64(len(data)*4))
		default:
			return fmt.Errorf("unsupported conversion from int32 to %v", deviceType)
		}

	case builder.INT64:
		data := hostData.([]int64)
		switch deviceType {
		case builder.INT32:
			// Convert int64 to int32 (with potential data loss)
			converted := make([]int32, len(data))
			for i, v := range data {
				converted[i] = int32(v)
			}
			mem.CopyFrom(unsafe.Pointer(&converted[0]), int64(len(converted)*4))
		case builder.INT64:
			// Same type - direct copy
			mem.CopyFrom(unsafe.Pointer(&data[0]), int64(len(data)*8))
		default:
			return fmt.Errorf("unsupported conversion from int64 to %v", deviceType)
		}

	default:
		return fmt.Errorf("unsupported host type for conversion: %v", hostType)
	}

	return nil
}

// copyFromDeviceWithTypeConversion handles type conversion during device→host copy
func (kr *Runner) copyFromDeviceWithTypeConversion(mem *gocca.OCCAMemory, hostData interface{},
	deviceType, hostType builder.DataType, bytes int64) error {

	// Handle all conversion cases
	switch deviceType {
	case builder.Float32:
		// Read float32 from device
		elemCount := bytes / 4
		deviceData := make([]float32, elemCount)
		mem.CopyTo(unsafe.Pointer(&deviceData[0]), bytes)

		switch hostType {
		case builder.Float64:
			// Convert to float64
			hostSlice := hostData.([]float64)
			for i, v := range deviceData {
				hostSlice[i] = float64(v)
			}
		case builder.Float32:
			// Same type - direct copy
			hostSlice := hostData.([]float32)
			copy(hostSlice, deviceData)
		default:
			return fmt.Errorf("unsupported conversion from device float32 to host %v", hostType)
		}

	case builder.Float64:
		// Read float64 from device
		elemCount := bytes / 8
		deviceData := make([]float64, elemCount)
		mem.CopyTo(unsafe.Pointer(&deviceData[0]), bytes)

		switch hostType {
		case builder.Float32:
			// Convert to float32
			hostSlice := hostData.([]float32)
			for i, v := range deviceData {
				hostSlice[i] = float32(v)
			}
		case builder.Float64:
			// Same type - direct copy
			hostSlice := hostData.([]float64)
			copy(hostSlice, deviceData)
		default:
			return fmt.Errorf("unsupported conversion from device float64 to host %v", hostType)
		}

	case builder.INT32:
		// Read int32 from device
		elemCount := bytes / 4
		deviceData := make([]int32, elemCount)
		mem.CopyTo(unsafe.Pointer(&deviceData[0]), bytes)

		switch hostType {
		case builder.INT64:
			// Convert to int64
			hostSlice := hostData.([]int64)
			for i, v := range deviceData {
				hostSlice[i] = int64(v)
			}
		case builder.INT32:
			// Same type - direct copy
			hostSlice := hostData.([]int32)
			copy(hostSlice, deviceData)
		default:
			return fmt.Errorf("unsupported conversion from device int32 to host %v", hostType)
		}

	case builder.INT64:
		// Read int64 from device
		elemCount := bytes / 8
		deviceData := make([]int64, elemCount)
		mem.CopyTo(unsafe.Pointer(&deviceData[0]), bytes)

		switch hostType {
		case builder.INT32:
			// Convert to int32
			hostSlice := hostData.([]int32)
			for i, v := range deviceData {
				hostSlice[i] = int32(v)
			}
		case builder.INT64:
			// Same type - direct copy
			hostSlice := hostData.([]int64)
			copy(hostSlice, deviceData)
		default:
			return fmt.Errorf("unsupported conversion from device int64 to host %v", hostType)
		}

	default:
		return fmt.Errorf("unsupported device type for conversion: %v", deviceType)
	}

	return nil
}

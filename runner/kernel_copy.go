package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/gocca"
	"gonum.org/v1/gonum/mat"
	"unsafe"
)

// copyToDeviceWithConversion handles host→device copy with optional type conversion
func (kr *Runner) copyToDeviceWithConversion(spec *ParamSpec) error {
	mem := kr.GetMemory(spec.Name)
	if mem == nil {
		return fmt.Errorf("memory for %s not found", spec.Name)
	}

	// Handle type conversion if needed
	if spec.ConvertType != 0 && spec.ConvertType != spec.DataType {
		// Perform conversion during copy
		return kr.copyWithTypeConversion(spec.HostBinding, mem, spec.DataType, spec.ConvertType)
	}

	// Direct copy based on type
	switch data := spec.HostBinding.(type) {
	case []float32:
		mem.CopyFrom(unsafe.Pointer(&data[0]), int64(len(data)*4))
	case []float64:
		mem.CopyFrom(unsafe.Pointer(&data[0]), int64(len(data)*8))
	case []int32:
		mem.CopyFrom(unsafe.Pointer(&data[0]), int64(len(data)*4))
	case []int64:
		mem.CopyFrom(unsafe.Pointer(&data[0]), int64(len(data)*8))
		// Add this case to the type switch in copyToDeviceWithConversion method in runner/kernel_definition.go
	case mat.Matrix:
		// Handle mat.Matrix as a flat array
		rows, cols := data.Dims()
		size := rows * cols

		// Extract matrix data as a flat array (column-major)
		flatData := make([]float64, size)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				// Column-major: column j, row i goes to position j*rows + i
				flatData[j*rows+i] = data.At(i, j)
			}
		}
		// Copy the flattened data to device
		mem.CopyFrom(unsafe.Pointer(&flatData[0]), int64(size*8))
	default:
		return fmt.Errorf("unsupported type for copy: %T", data)
	}

	return nil
}

// copyWithTypeConversion performs type conversion during copy
func (kr *Runner) copyWithTypeConversion(hostData interface{}, deviceMem *gocca.OCCAMemory,
	fromType, toType builder.DataType) error {
	// This is a simplified example - full implementation would handle all conversions
	switch fromType {
	case builder.Float64:
		if toType == builder.Float32 {
			// Convert float64 → float32
			src := hostData.([]float64)
			dst := make([]float32, len(src))
			for i, v := range src {
				dst[i] = float32(v)
			}
			deviceMem.CopyFrom(unsafe.Pointer(&dst[0]), int64(len(dst)*4))
			return nil
		}
	case builder.Float32:
		if toType == builder.Float64 {
			// Convert float32 → float64
			src := hostData.([]float32)
			dst := make([]float64, len(src))
			for i, v := range src {
				dst[i] = float64(v)
			}
			deviceMem.CopyFrom(unsafe.Pointer(&dst[0]), int64(len(dst)*8))
			return nil
		}
	}

	return fmt.Errorf("unsupported conversion from %v to %v", fromType, toType)
}

// copyFromDeviceWithConversion handles device→host copy with optional type conversion
func (kr *Runner) copyFromDeviceWithConversion(spec *ParamSpec) error {
	mem := kr.GetMemory(spec.Name)
	if mem == nil {
		return fmt.Errorf("memory for %s not found", spec.Name)
	}

	// For now, use the existing CopyArrayToHost for full arrays
	// This would be enhanced to handle partial copies and conversions

	hostBinding := spec.HostBinding
	if hostBinding == nil {
		// Try to get from stored bindings
		if binding, exists := kr.hostBindings[spec.Name]; exists {
			hostBinding = binding
		} else {
			return fmt.Errorf("no host binding for %s", spec.Name)
		}
	}

	// Get total size
	totalSize := spec.Size * sizeOfType(spec.GetEffectiveType())

	// Handle type conversion if needed
	if spec.ConvertType != 0 && spec.ConvertType != spec.DataType {
		// Perform conversion during copy back
		return kr.copyFromDeviceWithTypeConversion(mem, hostBinding, spec.ConvertType, spec.DataType, totalSize)
	}

	// Direct copy based on type
	switch data := hostBinding.(type) {
	case []float32:
		mem.CopyTo(unsafe.Pointer(&data[0]), totalSize)
	case []float64:
		mem.CopyTo(unsafe.Pointer(&data[0]), totalSize)
	case []int32:
		mem.CopyTo(unsafe.Pointer(&data[0]), totalSize)
	case []int64:
		mem.CopyTo(unsafe.Pointer(&data[0]), totalSize)
	case mat.Matrix:
		// Handle mat.Matrix for copy back
		rows, cols := data.Dims()
		size := rows * cols

		// Create temporary buffer to receive device data
		flatData := make([]float64, size)
		mem.CopyTo(unsafe.Pointer(&flatData[0]), totalSize)

		// Type assert to a mutable matrix type
		switch m := data.(type) {
		case *mat.Dense:
			// Unpack column-major data back into the Dense matrix
			for i := 0; i < rows; i++ {
				for j := 0; j < cols; j++ {
					// Column-major: position j*rows + i contains element (i,j)
					val := flatData[j*rows+i]
					m.Set(i, j, val)
				}
			}
		}
	default:
		return fmt.Errorf("unsupported type for copy: %T", data)
	}

	return nil
}

// copyFromDeviceWithTypeConversion performs type conversion during device→host copy
func (kr *Runner) copyFromDeviceWithTypeConversion(deviceMem *gocca.OCCAMemory, hostData interface{},
	fromType, toType builder.DataType, totalSize int64) error {
	// This is a simplified example - full implementation would handle all conversions
	switch fromType {
	case builder.Float32:
		if toType == builder.Float64 {
			// Device has float32, host wants float64
			temp := make([]float32, totalSize/4)
			deviceMem.CopyTo(unsafe.Pointer(&temp[0]), totalSize)

			dst := hostData.([]float64)
			for i, v := range temp {
				dst[i] = float64(v)
			}
			return nil
		}
	case builder.Float64:
		if toType == builder.Float32 {
			// Device has float64, host wants float32
			temp := make([]float64, totalSize/8)
			deviceMem.CopyTo(unsafe.Pointer(&temp[0]), totalSize)

			dst := hostData.([]float32)
			for i, v := range temp {
				dst[i] = float32(v)
			}
			return nil
		}
	}

	return fmt.Errorf("unsupported conversion from %v to %v", fromType, toType)
}

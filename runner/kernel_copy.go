// runner/kernel_copy.go
package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/gocca"
	"gonum.org/v1/gonum/mat"
	"unsafe"
)

// ============================================================================
// Main Entry Points
// ============================================================================

// copyToDeviceWithConversion handles host→device copy with optional type conversion
func (kr *Runner) copyToDeviceWithConversion(spec *builder.ParamSpec) error {
	mem := kr.GetMemory(spec.Name)
	if mem == nil {
		return fmt.Errorf("memory for %s not found", spec.Name)
	}

	// Handle partitioned data
	if spec.IsPartitioned {
		return kr.copyPartitionedData(spec, mem)
	}

	// Handle type conversion if needed
	if spec.ConvertType != 0 && spec.ConvertType != spec.DataType {
		// Perform conversion during copy
		return kr.copyWithTypeConversion(spec.HostBinding, mem, spec.DataType, spec.ConvertType)
	}

	// Direct copy based on type
	return kr.copyDirectToDevice(spec.HostBinding, mem)
}

// copyFromDeviceWithConversion handles device→host copy with optional type conversion
func (kr *Runner) copyFromDeviceWithConversion(spec *builder.ParamSpec) error {
	mem := kr.GetMemory(spec.Name)
	if mem == nil {
		return fmt.Errorf("memory for %s not found", spec.Name)
	}

	// Handle partitioned data
	if spec.IsPartitioned {
		return kr.copyPartitionedDataFromDevice(spec, mem)
	}

	// Get host binding
	hostBinding := spec.HostBinding
	if hostBinding == nil {
		if binding, exists := kr.hostBindings[spec.Name]; exists {
			hostBinding = binding
		} else {
			return fmt.Errorf("no host binding for %s", spec.Name)
		}
	}

	// Get total size based on effective type
	effectiveType := spec.DataType
	if spec.ConvertType != 0 {
		effectiveType = spec.ConvertType
	}
	totalSize := spec.Size * int64(sizeOfType(effectiveType))

	// Handle type conversion if needed
	if spec.ConvertType != 0 && spec.ConvertType != spec.DataType {
		// Perform conversion during copy back
		return kr.copyFromDeviceWithTypeConversion(mem, hostBinding, spec.ConvertType, spec.DataType, totalSize)
	}

	// Direct copy from device
	return kr.copyDirectFromDevice(hostBinding, mem, totalSize)
}

// ============================================================================
// Partitioned Data Handling
// ============================================================================

// copyPartitionedData handles partitioned data copy with optional type conversion
func (kr *Runner) copyPartitionedData(spec *builder.ParamSpec, deviceMem *gocca.OCCAMemory) error {
	offsets, err := kr.readPartitionOffsets(spec.Name)
	if err != nil {
		return err
	}

	needsConversion := spec.ConvertType != 0 && spec.ConvertType != spec.DataType
	targetType := spec.DataType
	if needsConversion {
		targetType = spec.ConvertType
	}

	// Copy each partition with optional conversion
	switch data := spec.HostBinding.(type) {
	case [][]float64:
		return kr.copyFloat64PartitionsToDevice(data, deviceMem, offsets, needsConversion, targetType)
	case [][]float32:
		return kr.copyFloat32PartitionsToDevice(data, deviceMem, offsets, needsConversion, targetType)
	case [][]int32:
		return kr.copyInt32PartitionsToDevice(data, deviceMem, offsets, needsConversion, targetType)
	case [][]int64:
		return kr.copyInt64PartitionsToDevice(data, deviceMem, offsets, needsConversion, targetType)
	case []mat.Matrix:
		return kr.copyMatrixPartitionsToDevice(data, deviceMem, offsets, needsConversion, targetType)
	default:
		return fmt.Errorf("unsupported partitioned type: %T", data)
	}
}

// copyPartitionedDataFromDevice handles partitioned data copy from device with optional type conversion
func (kr *Runner) copyPartitionedDataFromDevice(spec *builder.ParamSpec, deviceMem *gocca.OCCAMemory) error {
	offsets, err := kr.readPartitionOffsets(spec.Name)
	if err != nil {
		return err
	}

	needsConversion := spec.ConvertType != 0 && spec.ConvertType != spec.DataType
	sourceType := spec.DataType
	if spec.ConvertType != 0 {
		sourceType = spec.ConvertType
	}

	// Copy each partition with optional conversion
	switch data := spec.HostBinding.(type) {
	case [][]float64:
		return kr.copyFloat64PartitionsFromDevice(data, deviceMem, offsets, needsConversion, sourceType)
	case [][]float32:
		return kr.copyFloat32PartitionsFromDevice(data, deviceMem, offsets, needsConversion, sourceType)
	case [][]int32:
		return kr.copyInt32PartitionsFromDevice(data, deviceMem, offsets, needsConversion, sourceType)
	case [][]int64:
		return kr.copyInt64PartitionsFromDevice(data, deviceMem, offsets, needsConversion, sourceType)
	case []mat.Matrix:
		return kr.copyMatrixPartitionsFromDevice(data, deviceMem, offsets, needsConversion, sourceType)
	default:
		return fmt.Errorf("unsupported partitioned type for copy back: %T", data)
	}
}

// ============================================================================
// Helper Methods - Eliminate Duplication
// ============================================================================

// readPartitionOffsets reads partition offset arrays (coalesced duplicate logic)
func (kr *Runner) readPartitionOffsets(name string) ([]int64, error) {
	offsetsMem := kr.PooledMemory[name+"_offsets"]
	if offsetsMem == nil {
		return nil, fmt.Errorf("offsets for %s not found", name)
	}

	var offsets []int64
	if kr.GetIntSize() == 4 {
		offsets32 := make([]int32, kr.NumPartitions+1)
		offsetsMem.CopyTo(unsafe.Pointer(&offsets32[0]), int64(len(offsets32)*4))
		offsets = make([]int64, len(offsets32))
		for i, v := range offsets32 {
			offsets[i] = int64(v)
		}
	} else {
		offsets = make([]int64, kr.NumPartitions+1)
		offsetsMem.CopyTo(unsafe.Pointer(&offsets[0]), int64(len(offsets)*8))
	}

	return offsets, nil
}

// copyDirectToDevice performs direct copy to device without conversion
func (kr *Runner) copyDirectToDevice(hostData interface{}, mem *gocca.OCCAMemory) error {
	switch data := hostData.(type) {
	case []float32:
		mem.CopyFrom(unsafe.Pointer(&data[0]), int64(len(data)*4))
	case []float64:
		mem.CopyFrom(unsafe.Pointer(&data[0]), int64(len(data)*8))
	case []int32:
		mem.CopyFrom(unsafe.Pointer(&data[0]), int64(len(data)*4))
	case []int64:
		mem.CopyFrom(unsafe.Pointer(&data[0]), int64(len(data)*8))
	case mat.Matrix:
		flatData := kr.flattenMatrix(data)
		mem.CopyFrom(unsafe.Pointer(&flatData[0]), int64(len(flatData)*8))
	default:
		return fmt.Errorf("unsupported type for copy: %T", hostData)
	}
	return nil
}

// copyDirectFromDevice performs direct copy from device without conversion
func (kr *Runner) copyDirectFromDevice(hostData interface{}, mem *gocca.OCCAMemory, totalSize int64) error {
	switch data := hostData.(type) {
	case []float32:
		mem.CopyTo(unsafe.Pointer(&data[0]), totalSize)
	case []float64:
		mem.CopyTo(unsafe.Pointer(&data[0]), totalSize)
	case []int32:
		mem.CopyTo(unsafe.Pointer(&data[0]), totalSize)
	case []int64:
		mem.CopyTo(unsafe.Pointer(&data[0]), totalSize)
	case mat.Matrix:
		return kr.copyMatrixFromDevice(data, mem)
	default:
		return fmt.Errorf("unsupported type for copy: %T", hostData)
	}
	return nil
}

// flattenMatrix converts mat.Matrix to flat array (coalesced duplicate logic)
func (kr *Runner) flattenMatrix(data mat.Matrix) []float64 {
	rows, cols := data.Dims()
	size := rows * cols
	flatData := make([]float64, size)

	// Column-major order
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			flatData[j*rows+i] = data.At(i, j)
		}
	}

	return flatData
}

// copyMatrixFromDevice copies from device to mat.Matrix
func (kr *Runner) copyMatrixFromDevice(matrix mat.Matrix, mem *gocca.OCCAMemory) error {
	rows, cols := matrix.Dims()
	size := rows * cols

	// Create temporary buffer to receive device data
	flatData := make([]float64, size)
	mem.CopyTo(unsafe.Pointer(&flatData[0]), int64(size*8))

	// Type assert to mutable matrix
	m, ok := matrix.(*mat.Dense)
	if !ok {
		return fmt.Errorf("matrix must be *mat.Dense for modification")
	}

	// Unpack column-major data back into the Dense matrix
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.Set(i, j, flatData[j*rows+i])
		}
	}

	return nil
}

// ============================================================================
// Type Conversion (unified to eliminate duplication)
// ============================================================================

// copyWithTypeConversion performs type conversion during host→device copy
func (kr *Runner) copyWithTypeConversion(hostData interface{}, deviceMem *gocca.OCCAMemory,
	fromType, toType builder.DataType) error {

	switch fromType {
	case builder.Float64:
		if toType == builder.Float32 {
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

// copyFromDeviceWithTypeConversion performs type conversion during device→host copy
func (kr *Runner) copyFromDeviceWithTypeConversion(deviceMem *gocca.OCCAMemory, hostData interface{},
	fromType, toType builder.DataType, totalSize int64) error {

	switch fromType {
	case builder.Float32:
		if toType == builder.Float64 {
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

// ============================================================================
// Partitioned Copy Methods (specialized but using common helpers)
// ============================================================================

// copyFloat64PartitionsToDevice handles float64 partition copies to device
func (kr *Runner) copyFloat64PartitionsToDevice(data [][]float64, deviceMem *gocca.OCCAMemory,
	offsets []int64, needsConversion bool, targetType builder.DataType) error {

	for i, partition := range data {
		if i >= kr.NumPartitions {
			break
		}

		var ptr unsafe.Pointer
		var bytes int64

		if needsConversion && targetType == builder.Float32 {
			// Convert float64 to float32
			converted := make([]float32, len(partition))
			for j, v := range partition {
				converted[j] = float32(v)
			}
			ptr = unsafe.Pointer(&converted[0])
			bytes = int64(len(converted) * 4)
		} else {
			// Direct copy
			ptr = unsafe.Pointer(&partition[0])
			bytes = int64(len(partition) * 8)
		}

		offsetBytes := offsets[i] * int64(sizeOfType(targetType))
		deviceMem.CopyFromWithOffset(ptr, bytes, offsetBytes)
	}

	return nil
}

// copyFloat32PartitionsToDevice handles float32 partition copies to device
func (kr *Runner) copyFloat32PartitionsToDevice(data [][]float32, deviceMem *gocca.OCCAMemory,
	offsets []int64, needsConversion bool, targetType builder.DataType) error {

	for i, partition := range data {
		if i >= kr.NumPartitions {
			break
		}

		var ptr unsafe.Pointer
		var bytes int64

		if needsConversion && targetType == builder.Float64 {
			// Convert float32 to float64
			converted := make([]float64, len(partition))
			for j, v := range partition {
				converted[j] = float64(v)
			}
			ptr = unsafe.Pointer(&converted[0])
			bytes = int64(len(converted) * 8)
		} else {
			// Direct copy
			ptr = unsafe.Pointer(&partition[0])
			bytes = int64(len(partition) * 4)
		}

		offsetBytes := offsets[i] * int64(sizeOfType(targetType))
		deviceMem.CopyFromWithOffset(ptr, bytes, offsetBytes)
	}

	return nil
}

// copyInt64PartitionsToDevice handles int64 partition copies to device
func (kr *Runner) copyInt64PartitionsToDevice(data [][]int64, deviceMem *gocca.OCCAMemory,
	offsets []int64, needsConversion bool, targetType builder.DataType) error {

	for i, partition := range data {
		if i >= kr.NumPartitions {
			break
		}

		var ptr unsafe.Pointer
		var bytes int64

		if needsConversion && targetType == builder.INT32 {
			// Convert int64 to int32
			converted := make([]int32, len(partition))
			for j, v := range partition {
				converted[j] = int32(v)
			}
			ptr = unsafe.Pointer(&converted[0])
			bytes = int64(len(converted) * 4)
		} else {
			// Direct copy
			ptr = unsafe.Pointer(&partition[0])
			bytes = int64(len(partition) * 8)
		}

		offsetBytes := offsets[i] * int64(sizeOfType(targetType))
		deviceMem.CopyFromWithOffset(ptr, bytes, offsetBytes)
	}

	return nil
}

// copyInt32PartitionsToDevice handles int32 partition copies to device
func (kr *Runner) copyInt32PartitionsToDevice(data [][]int32, deviceMem *gocca.OCCAMemory,
	offsets []int64, needsConversion bool, targetType builder.DataType) error {

	for i, partition := range data {
		if i >= kr.NumPartitions {
			break
		}

		var ptr unsafe.Pointer
		var bytes int64

		if needsConversion && targetType == builder.INT64 {
			// Convert int32 to int64
			converted := make([]int64, len(partition))
			for j, v := range partition {
				converted[j] = int64(v)
			}
			ptr = unsafe.Pointer(&converted[0])
			bytes = int64(len(converted) * 8)
		} else {
			// Direct copy
			ptr = unsafe.Pointer(&partition[0])
			bytes = int64(len(partition) * 4)
		}

		offsetBytes := offsets[i] * int64(sizeOfType(targetType))
		deviceMem.CopyFromWithOffset(ptr, bytes, offsetBytes)
	}

	return nil
}

// copyMatrixPartitionsToDevice handles matrix partition copies to device
func (kr *Runner) copyMatrixPartitionsToDevice(data []mat.Matrix, deviceMem *gocca.OCCAMemory,
	offsets []int64, needsConversion bool, targetType builder.DataType) error {

	for i, matrix := range data {
		if i >= kr.NumPartitions {
			break
		}

		flatData := kr.flattenMatrix(matrix)

		var ptr unsafe.Pointer
		var bytes int64

		if needsConversion && targetType == builder.Float32 {
			// Convert float64 to float32
			converted := make([]float32, len(flatData))
			for j, v := range flatData {
				converted[j] = float32(v)
			}
			ptr = unsafe.Pointer(&converted[0])
			bytes = int64(len(converted) * 4)
		} else {
			// Direct copy
			ptr = unsafe.Pointer(&flatData[0])
			bytes = int64(len(flatData) * 8)
		}

		offsetBytes := offsets[i] * int64(sizeOfType(targetType))
		deviceMem.CopyFromWithOffset(ptr, bytes, offsetBytes)
	}

	return nil
}

// copyFloat64PartitionsFromDevice handles float64 partition copies from device
func (kr *Runner) copyFloat64PartitionsFromDevice(data [][]float64, deviceMem *gocca.OCCAMemory,
	offsets []int64, needsConversion bool, sourceType builder.DataType) error {

	for i, partition := range data {
		if i >= kr.NumPartitions {
			break
		}

		if needsConversion && sourceType == builder.Float32 {
			// Read as float32 and convert to float64
			temp := make([]float32, len(partition))
			partitionBytes := int64(len(temp) * 4)
			offsetBytes := offsets[i] * 4
			deviceMem.CopyToWithOffset(unsafe.Pointer(&temp[0]), partitionBytes, offsetBytes)

			// Convert to float64
			for j, v := range temp {
				partition[j] = float64(v)
			}
		} else {
			// Direct copy
			partitionBytes := int64(len(partition) * 8)
			offsetBytes := offsets[i] * 8
			deviceMem.CopyToWithOffset(unsafe.Pointer(&partition[0]), partitionBytes, offsetBytes)
		}
	}

	return nil
}

// copyFloat32PartitionsFromDevice handles float32 partition copies from device
func (kr *Runner) copyFloat32PartitionsFromDevice(data [][]float32, deviceMem *gocca.OCCAMemory,
	offsets []int64, needsConversion bool, sourceType builder.DataType) error {

	for i, partition := range data {
		if i >= kr.NumPartitions {
			break
		}

		if needsConversion && sourceType == builder.Float64 {
			// Read as float64 and convert to float32
			temp := make([]float64, len(partition))
			partitionBytes := int64(len(temp) * 8)
			offsetBytes := offsets[i] * 8
			deviceMem.CopyToWithOffset(unsafe.Pointer(&temp[0]), partitionBytes, offsetBytes)

			// Convert to float32
			for j, v := range temp {
				partition[j] = float32(v)
			}
		} else {
			// Direct copy
			partitionBytes := int64(len(partition) * 4)
			offsetBytes := offsets[i] * 4
			deviceMem.CopyToWithOffset(unsafe.Pointer(&partition[0]), partitionBytes, offsetBytes)
		}
	}

	return nil
}

// copyInt64PartitionsFromDevice handles int64 partition copies from device
func (kr *Runner) copyInt64PartitionsFromDevice(data [][]int64, deviceMem *gocca.OCCAMemory,
	offsets []int64, needsConversion bool, sourceType builder.DataType) error {

	for i, partition := range data {
		if i >= kr.NumPartitions {
			break
		}

		if needsConversion && sourceType == builder.INT32 {
			// Read as int32 and convert to int64
			temp := make([]int32, len(partition))
			partitionBytes := int64(len(temp) * 4)
			offsetBytes := offsets[i] * 4
			deviceMem.CopyToWithOffset(unsafe.Pointer(&temp[0]), partitionBytes, offsetBytes)

			// Convert to int64
			for j, v := range temp {
				partition[j] = int64(v)
			}
		} else {
			// Direct copy
			partitionBytes := int64(len(partition) * 8)
			offsetBytes := offsets[i] * 8
			deviceMem.CopyToWithOffset(unsafe.Pointer(&partition[0]), partitionBytes, offsetBytes)
		}
	}

	return nil
}

// copyInt32PartitionsFromDevice handles int32 partition copies from device
func (kr *Runner) copyInt32PartitionsFromDevice(data [][]int32, deviceMem *gocca.OCCAMemory,
	offsets []int64, needsConversion bool, sourceType builder.DataType) error {

	for i, partition := range data {
		if i >= kr.NumPartitions {
			break
		}

		if needsConversion && sourceType == builder.INT64 {
			// Read as int64 and convert to int32
			temp := make([]int64, len(partition))
			partitionBytes := int64(len(temp) * 8)
			offsetBytes := offsets[i] * 8
			deviceMem.CopyToWithOffset(unsafe.Pointer(&temp[0]), partitionBytes, offsetBytes)

			// Convert to int32
			for j, v := range temp {
				partition[j] = int32(v)
			}
		} else {
			// Direct copy
			partitionBytes := int64(len(partition) * 4)
			offsetBytes := offsets[i] * 4
			deviceMem.CopyToWithOffset(unsafe.Pointer(&partition[0]), partitionBytes, offsetBytes)
		}
	}

	return nil
}

// copyMatrixPartitionsFromDevice handles matrix partition copies from device
func (kr *Runner) copyMatrixPartitionsFromDevice(data []mat.Matrix, deviceMem *gocca.OCCAMemory,
	offsets []int64, needsConversion bool, sourceType builder.DataType) error {

	for i, matrix := range data {
		if i >= kr.NumPartitions {
			break
		}

		rows, cols := matrix.Dims()
		size := rows * cols

		if needsConversion && sourceType == builder.Float32 {
			// Read as float32 and convert to float64
			temp := make([]float32, size)
			partitionBytes := int64(size * 4)
			offsetBytes := offsets[i] * 4
			deviceMem.CopyToWithOffset(unsafe.Pointer(&temp[0]), partitionBytes, offsetBytes)

			// Type assert to mutable matrix
			if m, ok := matrix.(*mat.Dense); ok {
				// Convert and unpack
				for r := 0; r < rows; r++ {
					for c := 0; c < cols; c++ {
						val := float64(temp[c*rows+r])
						m.Set(r, c, val)
					}
				}
			}
		} else {
			// Direct copy
			flatData := make([]float64, size)
			partitionBytes := int64(size * 8)
			offsetBytes := offsets[i] * 8
			deviceMem.CopyToWithOffset(unsafe.Pointer(&flatData[0]), partitionBytes, offsetBytes)

			// Type assert to mutable matrix
			if m, ok := matrix.(*mat.Dense); ok {
				// Unpack column-major data back into the Dense matrix
				for r := 0; r < rows; r++ {
					for c := 0; c < cols; c++ {
						val := flatData[c*rows+r]
						m.Set(r, c, val)
					}
				}
			}
		}
	}

	return nil
}

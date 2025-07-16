// runner/kernel_copy.go
package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/gocca"
	"gonum.org/v1/gonum/mat"
	"unsafe"
)

// copyMatrixPartitionsFromDevice handles matrix partition copies from device
func (kr *Runner) copyMatrixPartitionsFromDevice(data []mat.Matrix, deviceMem *gocca.OCCAMemory,
	offsets []int64, needsConversion bool, sourceType builder.DataType) error {

	// fmt.Printf("\n=== BEGIN copyMatrixPartitionsFromDevice DEBUG ===\n")
	// fmt.Printf("Num partitions: %d, needsConversion: %v, sourceType: %v\n",
	// 	kr.NumPartitions, needsConversion, sourceType)
	// fmt.Printf("Offsets: %v\n", offsets)

	for i, matrix := range data {
		if i >= kr.NumPartitions {
			break
		}

		// fmt.Printf("\nPartition %d:\n", i)
		rows, cols := matrix.Dims()
		// fmt.Printf("  Matrix dims: %dx%d = %d elements\n", rows, cols, rows*cols)
		size := rows * cols

		if needsConversion && sourceType == builder.Float32 {
			// Read as float32 and convert
			temp := make([]float32, size)
			partitionBytes := int64(size * 4)
			offsetBytes := offsets[i] * 4

			// fmt.Printf("  Conversion path (float32 -> float64)\n")
			// fmt.Printf("  Partition bytes: %d\n", partitionBytes)
			// fmt.Printf("  Offset bytes: %d\n", offsetBytes)
			// fmt.Printf("  Calling CopyToWithOffset...\n")

			deviceMem.CopyToWithOffset(unsafe.Pointer(&temp[0]), partitionBytes, offsetBytes)

			// fmt.Printf("  CopyToWithOffset succeeded\n")

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

			// fmt.Printf("  Direct copy path (no conversion)\n")
			// fmt.Printf("  Partition bytes: %d\n", partitionBytes)
			// fmt.Printf("  Offset bytes: %d\n", offsetBytes)
			// fmt.Printf("  Access range: [%d, %d)\n", offsetBytes, offsetBytes+partitionBytes)

			// DEBUG: Check if we have a next offset to verify bounds
			if i+1 < len(offsets) {
				nextOffsetBytes := offsets[i+1] * 8
				// fmt.Printf("  Next partition starts at: %d bytes\n", nextOffsetBytes)
				if offsetBytes+partitionBytes > nextOffsetBytes {
					// fmt.Printf("  *** WARNING: Would overrun into next partition! ***\n")
				}
			}

			// This is where the error occurs
			// fmt.Printf("  Calling CopyToWithOffset...\n")
			deviceMem.CopyToWithOffset(unsafe.Pointer(&flatData[0]), partitionBytes, offsetBytes)
			// fmt.Printf("  CopyToWithOffset succeeded\n")

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
		// fmt.Printf("\n")
	}

	// fmt.Printf("=== END copyMatrixPartitionsFromDevice DEBUG ===\n\n")
	return nil
}

// copyMatrixToDevice copies a matrix to device with transposition
func (kr *Runner) copyMatrixToDevice(matrix mat.Matrix, mem *gocca.OCCAMemory, dataType builder.DataType) error {
	rows, cols := matrix.Dims()
	totalElements := rows * cols

	if dataType == builder.Float64 {
		// Transpose to column-major
		transposed := make([]float64, totalElements)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				transposed[j*rows+i] = matrix.At(i, j)
			}
		}
		mem.CopyFrom(unsafe.Pointer(&transposed[0]), int64(totalElements*8))
	} else {
		// Convert and transpose
		transposed := make([]float32, totalElements)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				transposed[j*rows+i] = float32(matrix.At(i, j))
			}
		}
		mem.CopyFrom(unsafe.Pointer(&transposed[0]), int64(totalElements*4))
	}
	return nil
}

// copyMatrixFromDevice copies a matrix from device with transposition
func (kr *Runner) copyMatrixFromDevice(matrix mat.Matrix, mem *gocca.OCCAMemory, dataType builder.DataType) error {
	// Type assert to get mutable matrix
	m, ok := matrix.(*mat.Dense)
	if !ok {
		return fmt.Errorf("matrix must be *mat.Dense for copy back")
	}

	rows, cols := m.Dims()
	totalElements := rows * cols

	if dataType == builder.Float64 {
		// Read column-major data
		transposed := make([]float64, totalElements)
		mem.CopyTo(unsafe.Pointer(&transposed[0]), int64(totalElements*8))

		// Transpose back to row-major
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				m.Set(i, j, transposed[j*rows+i])
			}
		}
	} else {
		// Read and convert
		transposed := make([]float32, totalElements)
		mem.CopyTo(unsafe.Pointer(&transposed[0]), int64(totalElements*4))

		// Transpose and convert back
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				m.Set(i, j, float64(transposed[j*rows+i]))
			}
		}
	}
	return nil
}

func (kr *Runner) copyToDeviceWithConversion(spec *builder.ParamSpec) error {
	if spec.HostBinding == nil {
		return nil // Nothing to copy
	}

	// Get memory - this handles both partitioned and non-partitioned cases
	mem := kr.GetMemory(spec.Name)
	if mem == nil {
		return fmt.Errorf("no device memory allocated for %s", spec.Name)
	}

	// Handle partitioned data types ([][]T format)
	if spec.IsPartitioned {
		return kr.copyPartitionedData(spec, mem)
	}

	// Handle matrix types specially
	if matrix, ok := spec.HostBinding.(mat.Matrix); ok {
		return kr.copyMatrixToDevice(matrix, mem, spec.GetEffectiveType())
	}

	// For non-matrix, non-partitioned types
	hostType := spec.DataType
	deviceType := spec.GetEffectiveType()

	if hostType == deviceType {
		// No conversion needed
		return kr.copyDirectToDevice(spec.HostBinding, mem)
	}

	// Type conversion needed
	return kr.copyWithTypeConversion(spec.HostBinding, mem, hostType, deviceType)
}

func (kr *Runner) copyFromDeviceWithConversion(spec *builder.ParamSpec) error {
	if spec.HostBinding == nil {
		return nil // No destination
	}

	// Get memory - this handles both partitioned and non-partitioned cases
	mem := kr.GetMemory(spec.Name)
	if mem == nil {
		return fmt.Errorf("no device memory allocated for %s", spec.Name)
	}

	// Handle partitioned data types ([][]T and []mat.Matrix formats)
	if spec.IsPartitioned {
		return kr.copyPartitionedDataFromDevice(spec, mem)
	}

	// Handle single matrix types specially
	if matrix, ok := spec.HostBinding.(mat.Matrix); ok {
		return kr.copyMatrixFromDevice(matrix, mem, spec.GetEffectiveType())
	}

	// Determine types for non-matrix, non-partitioned data
	hostType := spec.DataType
	deviceType := spec.GetEffectiveType()

	// Calculate size based on device type
	var elementSize int64
	switch deviceType {
	case builder.Float32, builder.INT32:
		elementSize = 4
	case builder.Float64, builder.INT64:
		elementSize = 8
	default:
		elementSize = 8
	}
	totalSize := spec.Size * elementSize

	if hostType == deviceType {
		// No conversion needed
		return kr.copyDirectFromDevice(spec.HostBinding, mem, totalSize)
	}

	// Type conversion needed
	return kr.copyFromDeviceWithTypeConversion(mem, spec.HostBinding, deviceType, hostType, totalSize)
}

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
	sourceType := spec.GetEffectiveType() // What's actually stored on device
	// sourceType := spec.DataType
	// if spec.ConvertType != 0 {
	// 	sourceType = spec.ConvertType
	// }

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

// flattenMatrix converts a matrix to a flat array in column-major order
func (kr *Runner) flattenMatrix(matrix mat.Matrix) []float64 {
	rows, cols := matrix.Dims()
	flat := make([]float64, rows*cols)

	// Transpose to column-major
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			flat[j*rows+i] = matrix.At(i, j)
		}
	}
	return flat
}

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

		offsetBytes := offsets[i] * SizeOfType(targetType)
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

		offsetBytes := offsets[i] * SizeOfType(targetType)
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

		offsetBytes := offsets[i] * SizeOfType(targetType)
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

		offsetBytes := offsets[i] * SizeOfType(targetType)
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

		offsetBytes := offsets[i] * SizeOfType(targetType)
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
			offsetBytes := offsets[i] * 4 // FIX: Use sourceType size (4 bytes for float32)
			deviceMem.CopyToWithOffset(unsafe.Pointer(&temp[0]), partitionBytes, offsetBytes)

			// Convert to float64
			for j, v := range temp {
				partition[j] = float64(v)
			}
		} else {
			// Direct copy - device has float64, host has float64
			partitionBytes := int64(len(partition) * 8)
			offsetBytes := offsets[i] * 8 // offsets[i] is in elements, multiply by element size
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
			// WRONG: Device has float64, but output specified Float32 allocation
			// This shouldn't happen with proper GetEffectiveType implementation
			return fmt.Errorf("invalid conversion: output array on device should match Convert type")
		} else {
			// Direct copy - device has float32, host has float32
			partitionBytes := int64(len(partition) * 4)
			offsetBytes := offsets[i] * 4 // offsets[i] is in elements, multiply by element size
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

func (kr *Runner) copyDirectToDevice(hostData interface{}, mem *gocca.OCCAMemory) error {
	switch data := hostData.(type) {
	case []float64:
		mem.CopyFrom(unsafe.Pointer(&data[0]), int64(len(data)*8))
	case []float32:
		mem.CopyFrom(unsafe.Pointer(&data[0]), int64(len(data)*4))
	case []int32:
		mem.CopyFrom(unsafe.Pointer(&data[0]), int64(len(data)*4))
	case []int64:
		mem.CopyFrom(unsafe.Pointer(&data[0]), int64(len(data)*8))
	case mat.Matrix:
		// This should NEVER happen - matrices must go through copyMatrixToDevice
		return fmt.Errorf("INTERNAL ERROR: mat.Matrix reached copyDirectToDevice - this indicates a bug in copy logic. Matrices must use copyMatrixToDevice for transpose")
	case [][]float64, [][]float32, [][]int32, [][]int64, []mat.Matrix:
		return fmt.Errorf("partitioned data requires special handling - use copyPartitionedData directly")
	default:
		return fmt.Errorf("unsupported type for direct copy: %T", data)
	}
	return nil
}

func (kr *Runner) copyDirectFromDevice(hostData interface{}, mem *gocca.OCCAMemory, size int64) error {
	switch data := hostData.(type) {
	case []float64:
		mem.CopyTo(unsafe.Pointer(&data[0]), size)
	case []float32:
		mem.CopyTo(unsafe.Pointer(&data[0]), size)
	case []int32:
		mem.CopyTo(unsafe.Pointer(&data[0]), size)
	case []int64:
		mem.CopyTo(unsafe.Pointer(&data[0]), size)
	case mat.Matrix:
		// This should NEVER happen - matrices must go through copyMatrixFromDevice
		return fmt.Errorf("INTERNAL ERROR: mat.Matrix reached copyDirectFromDevice - this indicates a bug in copy logic. Matrices must use copyMatrixFromDevice for transpose")
	case [][]float64, [][]float32, [][]int32, [][]int64, []mat.Matrix:
		return fmt.Errorf("partitioned data requires special handling - use copyPartitionedDataFromDevice directly")
	default:
		return fmt.Errorf("unsupported type for direct copy from device: %T", data)
	}
	return nil
}

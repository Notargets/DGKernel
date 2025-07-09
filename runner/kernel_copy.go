package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/gocca"
	"gonum.org/v1/gonum/mat"
	"unsafe"
)

// copyToDeviceWithConversion handles host→device copy with optional type conversion
func (kr *Runner) copyToDeviceWithConversion(spec *builder.ParamSpec) error {
	mem := kr.GetMemory(spec.Name)
	if mem == nil {
		return fmt.Errorf("memory for %s not found", spec.Name)
	}

	// NEW: Handle partitioned data
	if spec.IsPartitioned {
		return kr.copyPartitionedData(spec, mem)
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

// NEW: Add function to copy partitioned data
func (kr *Runner) copyPartitionedData(spec *builder.ParamSpec, deviceMem *gocca.OCCAMemory) error {
	// Get offset array
	offsetsMem := kr.PooledMemory[spec.Name+"_offsets"]
	if offsetsMem == nil {
		return fmt.Errorf("offsets for %s not found", spec.Name)
	}

	// Read offsets
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

	// Copy each partition to its aligned offset
	switch data := spec.HostBinding.(type) {
	case [][]float64:
		for i, partition := range data {
			if i >= kr.NumPartitions {
				break
			}
			partitionBytes := int64(len(partition) * 8)
			offsetBytes := offsets[i] * 8
			deviceMem.CopyFromWithOffset(unsafe.Pointer(&partition[0]), partitionBytes, offsetBytes)
		}
	case [][]float32:
		for i, partition := range data {
			if i >= kr.NumPartitions {
				break
			}
			partitionBytes := int64(len(partition) * 4)
			offsetBytes := offsets[i] * 4
			deviceMem.CopyFromWithOffset(unsafe.Pointer(&partition[0]), partitionBytes, offsetBytes)
		}
	case [][]int32:
		for i, partition := range data {
			if i >= kr.NumPartitions {
				break
			}
			partitionBytes := int64(len(partition) * 4)
			offsetBytes := offsets[i] * 4
			deviceMem.CopyFromWithOffset(unsafe.Pointer(&partition[0]), partitionBytes, offsetBytes)
		}
	case [][]int64:
		for i, partition := range data {
			if i >= kr.NumPartitions {
				break
			}
			partitionBytes := int64(len(partition) * 8)
			offsetBytes := offsets[i] * 8
			deviceMem.CopyFromWithOffset(unsafe.Pointer(&partition[0]), partitionBytes, offsetBytes)
		}
	case []mat.Matrix:
		for i, matrix := range data {
			if i >= kr.NumPartitions {
				break
			}
			rows, cols := matrix.Dims()
			size := rows * cols

			// Extract matrix data as flat array (column-major)
			flatData := make([]float64, size)
			for r := 0; r < rows; r++ {
				for c := 0; c < cols; c++ {
					flatData[c*rows+r] = matrix.At(r, c)
				}
			}

			partitionBytes := int64(size * 8)
			offsetBytes := offsets[i] * 8
			deviceMem.CopyFromWithOffset(unsafe.Pointer(&flatData[0]), partitionBytes, offsetBytes)
		}
	default:
		return fmt.Errorf("unsupported partitioned type: %T", data)
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

// NEW: Add function to copy partitioned data from device
func (kr *Runner) copyPartitionedDataFromDevice(spec *builder.ParamSpec, deviceMem *gocca.OCCAMemory) error {
	// Get offset array
	offsetsMem := kr.PooledMemory[spec.Name+"_offsets"]
	if offsetsMem == nil {
		return fmt.Errorf("offsets for %s not found", spec.Name)
	}

	// Read offsets
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

	// Copy each partition from its aligned offset
	switch data := spec.HostBinding.(type) {
	case [][]float64:
		for i, partition := range data {
			if i >= kr.NumPartitions {
				break
			}
			partitionBytes := int64(len(partition) * 8)
			offsetBytes := offsets[i] * 8
			deviceMem.CopyToWithOffset(unsafe.Pointer(&partition[0]), partitionBytes, offsetBytes)
		}
	case [][]float32:
		for i, partition := range data {
			if i >= kr.NumPartitions {
				break
			}
			partitionBytes := int64(len(partition) * 4)
			offsetBytes := offsets[i] * 4
			deviceMem.CopyToWithOffset(unsafe.Pointer(&partition[0]), partitionBytes, offsetBytes)
		}
	case [][]int32:
		for i, partition := range data {
			if i >= kr.NumPartitions {
				break
			}
			partitionBytes := int64(len(partition) * 4)
			offsetBytes := offsets[i] * 4
			deviceMem.CopyToWithOffset(unsafe.Pointer(&partition[0]), partitionBytes, offsetBytes)
		}
	case [][]int64:
		for i, partition := range data {
			if i >= kr.NumPartitions {
				break
			}
			partitionBytes := int64(len(partition) * 8)
			offsetBytes := offsets[i] * 8
			deviceMem.CopyToWithOffset(unsafe.Pointer(&partition[0]), partitionBytes, offsetBytes)
		}
	case []mat.Matrix:
		for i, matrix := range data {
			if i >= kr.NumPartitions {
				break
			}
			rows, cols := matrix.Dims()
			size := rows * cols

			// Create temporary buffer to receive device data
			flatData := make([]float64, size)
			partitionBytes := int64(size * 8)
			offsetBytes := offsets[i] * 8
			deviceMem.CopyToWithOffset(unsafe.Pointer(&flatData[0]), partitionBytes, offsetBytes)

			// Type assert to mutable matrix
			if m, ok := matrix.(*mat.Dense); ok {
				// Unpack column-major data back into the Dense matrix
				for r := 0; r < rows; r++ {
					for c := 0; c < cols; c++ {
						// Column-major: position c*rows + r contains element (r,c)
						val := flatData[c*rows+r]
						m.Set(r, c, val)
					}
				}
			}
		}
	default:
		return fmt.Errorf("unsupported partitioned type for copy back: %T", data)
	}

	return nil
}

// copyFromDeviceWithConversion handles device→host copy with optional type conversion
func (kr *Runner) copyFromDeviceWithConversion(spec *builder.ParamSpec) error {
	mem := kr.GetMemory(spec.Name)
	if mem == nil {
		return fmt.Errorf("memory for %s not found", spec.Name)
	}
	// NEW: Handle partitioned data
	if spec.IsPartitioned {
		return kr.copyPartitionedDataFromDevice(spec, mem)
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

// Helper function to get offsets for a named array
func (kr *Runner) getPartitionOffsets(name string) ([]int64, error) {
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

// Partitioned float64 to float32 conversion
func (kr *Runner) copyPartitionedFloat64ToFloat32(partitions [][]float64, deviceMem *gocca.OCCAMemory) error {
	// Note: We need to find the array name from metadata - this is a limitation
	// For now, assume we can get it from context or pass it as parameter
	// This would need refactoring to pass the spec or name

	// For each partition
	for i, partition := range partitions {
		if i >= kr.NumPartitions {
			break
		}

		// Convert float64 to float32
		converted := make([]float32, len(partition))
		for j, v := range partition {
			converted[j] = float32(v)
		}

		// Calculate offset manually based on partition sizes
		offset := int64(0)
		for p := 0; p < i; p++ {
			offset += int64(kr.K[p] * len(partition) / kr.K[i])
		}

		partitionBytes := int64(len(converted) * 4)
		offsetBytes := offset * 4
		deviceMem.CopyFromWithOffset(unsafe.Pointer(&converted[0]), partitionBytes, offsetBytes)
	}

	return nil
}

// Partitioned float32 to float64 conversion
func (kr *Runner) copyPartitionedFloat32ToFloat64(partitions [][]float32, deviceMem *gocca.OCCAMemory) error {
	for i, partition := range partitions {
		if i >= kr.NumPartitions {
			break
		}

		// Convert float32 to float64
		converted := make([]float64, len(partition))
		for j, v := range partition {
			converted[j] = float64(v)
		}

		// Calculate offset manually
		offset := int64(0)
		for p := 0; p < i; p++ {
			offset += int64(kr.K[p] * len(partition) / kr.K[i])
		}

		partitionBytes := int64(len(converted) * 8)
		offsetBytes := offset * 8
		deviceMem.CopyFromWithOffset(unsafe.Pointer(&converted[0]), partitionBytes, offsetBytes)
	}

	return nil
}

// Partitioned int64 to int32 conversion
func (kr *Runner) copyPartitionedInt64ToInt32(partitions [][]int64, deviceMem *gocca.OCCAMemory) error {
	for i, partition := range partitions {
		if i >= kr.NumPartitions {
			break
		}

		// Convert int64 to int32
		converted := make([]int32, len(partition))
		for j, v := range partition {
			converted[j] = int32(v)
		}

		// Calculate offset manually
		offset := int64(0)
		for p := 0; p < i; p++ {
			offset += int64(kr.K[p] * len(partition) / kr.K[i])
		}

		partitionBytes := int64(len(converted) * 4)
		offsetBytes := offset * 4
		deviceMem.CopyFromWithOffset(unsafe.Pointer(&converted[0]), partitionBytes, offsetBytes)
	}

	return nil
}

// Partitioned int32 to int64 conversion
func (kr *Runner) copyPartitionedInt32ToInt64(partitions [][]int32, deviceMem *gocca.OCCAMemory) error {
	for i, partition := range partitions {
		if i >= kr.NumPartitions {
			break
		}

		// Convert int32 to int64
		converted := make([]int64, len(partition))
		for j, v := range partition {
			converted[j] = int64(v)
		}

		// Calculate offset manually
		offset := int64(0)
		for p := 0; p < i; p++ {
			offset += int64(kr.K[p] * len(partition) / kr.K[i])
		}

		partitionBytes := int64(len(converted) * 8)
		offsetBytes := offset * 8
		deviceMem.CopyFromWithOffset(unsafe.Pointer(&converted[0]), partitionBytes, offsetBytes)
	}

	return nil
}

// Update copyFromDeviceWithTypeConversion to handle partitioned data
func (kr *Runner) copyFromDeviceWithTypeConversion(deviceMem *gocca.OCCAMemory, hostData interface{},
	fromType, toType builder.DataType, totalSize int64) error {

	// Handle partitioned data with type conversion
	switch data := hostData.(type) {
	case [][]float64:
		if fromType == builder.Float32 {
			return kr.copyPartitionedFloat32ToFloat64FromDevice(data, deviceMem)
		}
		return fmt.Errorf("unsupported conversion from %v to [][]float64", fromType)
	case [][]float32:
		if fromType == builder.Float64 {
			return kr.copyPartitionedFloat64ToFloat32FromDevice(data, deviceMem)
		}
		return fmt.Errorf("unsupported conversion from %v to [][]float32", fromType)
	case [][]int64:
		if fromType == builder.INT32 {
			return kr.copyPartitionedInt32ToInt64FromDevice(data, deviceMem)
		}
		return fmt.Errorf("unsupported conversion from %v to [][]int64", fromType)
	case [][]int32:
		if fromType == builder.INT64 {
			return kr.copyPartitionedInt64ToInt32FromDevice(data, deviceMem)
		}
		return fmt.Errorf("unsupported conversion from %v to [][]int32", fromType)
	}

	// Existing conversion logic for non-partitioned data
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

// Partitioned float32 to float64 conversion from device
func (kr *Runner) copyPartitionedFloat32ToFloat64FromDevice(partitions [][]float64, deviceMem *gocca.OCCAMemory) error {
	for i, partition := range partitions {
		if i >= kr.NumPartitions {
			break
		}

		// Read as float32 and convert to float64
		temp := make([]float32, len(partition))

		// Calculate offset manually
		offset := int64(0)
		for p := 0; p < i; p++ {
			offset += int64(kr.K[p] * len(partition) / kr.K[i])
		}

		partitionBytes := int64(len(temp) * 4)
		offsetBytes := offset * 4
		deviceMem.CopyToWithOffset(unsafe.Pointer(&temp[0]), partitionBytes, offsetBytes)

		// Convert to float64
		for j, v := range temp {
			partition[j] = float64(v)
		}
	}

	return nil
}

// Partitioned float64 to float32 conversion from device
func (kr *Runner) copyPartitionedFloat64ToFloat32FromDevice(partitions [][]float32, deviceMem *gocca.OCCAMemory) error {
	for i, partition := range partitions {
		if i >= kr.NumPartitions {
			break
		}

		// Read as float64 and convert to float32
		temp := make([]float64, len(partition))

		// Calculate offset manually
		offset := int64(0)
		for p := 0; p < i; p++ {
			offset += int64(kr.K[p] * len(partition) / kr.K[i])
		}

		partitionBytes := int64(len(temp) * 8)
		offsetBytes := offset * 8
		deviceMem.CopyToWithOffset(unsafe.Pointer(&temp[0]), partitionBytes, offsetBytes)

		// Convert to float32
		for j, v := range temp {
			partition[j] = float32(v)
		}
	}

	return nil
}

// Partitioned int32 to int64 conversion from device
func (kr *Runner) copyPartitionedInt32ToInt64FromDevice(partitions [][]int64, deviceMem *gocca.OCCAMemory) error {
	for i, partition := range partitions {
		if i >= kr.NumPartitions {
			break
		}

		// Read as int32 and convert to int64
		temp := make([]int32, len(partition))

		// Calculate offset manually
		offset := int64(0)
		for p := 0; p < i; p++ {
			offset += int64(kr.K[p] * len(partition) / kr.K[i])
		}

		partitionBytes := int64(len(temp) * 4)
		offsetBytes := offset * 4
		deviceMem.CopyToWithOffset(unsafe.Pointer(&temp[0]), partitionBytes, offsetBytes)

		// Convert to int64
		for j, v := range temp {
			partition[j] = int64(v)
		}
	}

	return nil
}

// Partitioned int64 to int32 conversion from device
func (kr *Runner) copyPartitionedInt64ToInt32FromDevice(partitions [][]int32, deviceMem *gocca.OCCAMemory) error {
	for i, partition := range partitions {
		if i >= kr.NumPartitions {
			break
		}

		// Read as int64 and convert to int32
		temp := make([]int64, len(partition))

		// Calculate offset manually
		offset := int64(0)
		for p := 0; p < i; p++ {
			offset += int64(kr.K[p] * len(partition) / kr.K[i])
		}

		partitionBytes := int64(len(temp) * 8)
		offsetBytes := offset * 8
		deviceMem.CopyToWithOffset(unsafe.Pointer(&temp[0]), partitionBytes, offsetBytes)

		// Convert to int32
		for j, v := range temp {
			partition[j] = int32(v)
		}
	}

	return nil
}

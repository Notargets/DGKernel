package runner

import (
	"fmt"
	"unsafe"
)

// ============================================================================
// Public API for copying data between host and device
// ============================================================================

// CopyArrayToHost copies all partitions' data from device to host as a contiguous array
func CopyArrayToHost[T any](kr *Runner, name string) ([]T, error) {
	// Check if array exists
	metadata, exists := kr.arrayMetadata[name]
	if !exists {
		return nil, fmt.Errorf("array %s not found", name)
	}

	// Verify type matches
	var sample T
	requestedType := GetDataTypeFromSample(sample)
	if requestedType != metadata.dataType {
		return nil, fmt.Errorf("type mismatch: array is %v, requested %v",
			metadata.dataType, requestedType)
	}

	// Get memory and offsets
	memory := kr.GetMemory(name)
	if memory == nil {
		return nil, fmt.Errorf("memory for %s not found", name)
	}

	offsetsMem := kr.PooledMemory[name+"_offsets"]
	if offsetsMem == nil {
		return nil, fmt.Errorf("offsets for %s not found", name)
	}

	// Read offsets to determine actual data locations
	numOffsets := kr.NumPartitions + 1
	var offsets []int64

	if kr.GetIntSize() == 4 {
		offsets32 := make([]int32, numOffsets)
		offsetsMem.CopyTo(unsafe.Pointer(&offsets32[0]), int64(numOffsets*4))
		offsets = make([]int64, numOffsets)
		for i, v := range offsets32 {
			offsets[i] = int64(v)
		}
	} else {
		offsets = make([]int64, numOffsets)
		offsetsMem.CopyTo(unsafe.Pointer(&offsets[0]), int64(numOffsets*8))
	}

	// Calculate total elements (sum of K values)
	totalElements := kr.GetTotalElements()
	result := make([]T, totalElements)

	// Get element size
	elementSize := int64(unsafe.Sizeof(sample))

	// Copy each partition contiguously
	destIndex := 0
	for i := 0; i < kr.NumPartitions; i++ {
		partitionElements := kr.K[i]
		partitionBytes := int64(partitionElements) * elementSize
		sourceOffset := offsets[i] * elementSize

		// Copy this partition's data
		memory.CopyToWithOffset(
			unsafe.Pointer(&result[destIndex]),
			partitionBytes,
			sourceOffset,
		)

		destIndex += partitionElements
	}

	return result, nil
}

// CopyPartitionToHost copies a specific partition's data from device to host
func CopyPartitionToHost[T any](kr *Runner, name string, partitionID int) ([]T, error) {
	if partitionID < 0 || partitionID >= kr.NumPartitions {
		return nil, fmt.Errorf("invalid partition ID: %d", partitionID)
	}

	// Check if array exists
	metadata, exists := kr.arrayMetadata[name]
	if !exists {
		return nil, fmt.Errorf("array %s not found", name)
	}

	// Verify type matches
	var sample T
	requestedType := GetDataTypeFromSample(sample)
	if requestedType != metadata.dataType {
		return nil, fmt.Errorf("type mismatch: array is %v, requested %v",
			metadata.dataType, requestedType)
	}

	// Get memory and offsets
	memory := kr.GetMemory(name)
	if memory == nil {
		return nil, fmt.Errorf("memory for %s not found", name)
	}

	offsetsMem := kr.PooledMemory[name+"_offsets"]
	if offsetsMem == nil {
		return nil, fmt.Errorf("offsets for %s not found", name)
	}

	// Read just the offsets we need
	var partitionOffset int64
	if kr.GetIntSize() == 4 {
		offsets32 := make([]int32, 2)
		offsetsBytes := int64(8) // 2 * 4 bytes
		sourceOffset := int64(partitionID * 4)
		offsetsMem.CopyToWithOffset(unsafe.Pointer(&offsets32[0]), offsetsBytes, sourceOffset)
		partitionOffset = int64(offsets32[0])
	} else {
		offsets := make([]int64, 2)
		offsetsBytes := int64(16) // 2 * 8 bytes
		sourceOffset := int64(partitionID * 8)
		offsetsMem.CopyToWithOffset(unsafe.Pointer(&offsets[0]), offsetsBytes, sourceOffset)
		partitionOffset = offsets[0]
	}

	// Create result array for this partition
	partitionElements := kr.K[partitionID]
	result := make([]T, partitionElements)

	// Copy partition data
	elementSize := int64(unsafe.Sizeof(sample))
	partitionBytes := int64(partitionElements) * elementSize
	sourceOffset := partitionOffset * elementSize

	memory.CopyToWithOffset(
		unsafe.Pointer(&result[0]),
		partitionBytes,
		sourceOffset,
	)

	return result, nil
}

package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"math"
	"testing"
	"unsafe"
)

// ============================================================================
// Section 1: Fundamental Tests - Basic Operations
// Following Unit Testing Principle: Start with fundamentals
// ============================================================================

// Test 1.1: GetArrayType basic functionality
func TestGetArrayType_BasicFunctionality(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K:         []int{10},
		FloatType: builder.Float64,
		IntType:   builder.INT32,
	})
	defer kp.Free()

	// Allocate arrays with different types
	specs := []builder.ArraySpec{
		{Name: "float64_array", Size: 10 * 8, DataType: builder.Float64, Alignment: builder.NoAlignment},
		{Name: "float32_array", Size: 10 * 4, DataType: builder.Float32, Alignment: builder.NoAlignment},
		{Name: "int32_array", Size: 10 * 4, DataType: builder.INT32, Alignment: builder.NoAlignment},
		{Name: "int64_array", Size: 10 * 8, DataType: builder.INT64, Alignment: builder.NoAlignment},
	}

	err := kp.AllocateArrays(specs)
	if err != nil {
		t.Fatalf("Failed to allocate arrays: %v", err)
	}

	// Test each array type
	testCases := []struct {
		name     string
		expected builder.DataType
	}{
		{"float64_array", builder.Float64},
		{"float32_array", builder.Float32},
		{"int32_array", builder.INT32},
		{"int64_array", builder.INT64},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			dataType, err := kp.GetArrayType(tc.name)
			if err != nil {
				t.Errorf("GetArrayType failed: %v", err)
			}
			if dataType != tc.expected {
				t.Errorf("Expected type %v, got %v", tc.expected, dataType)
			}
		})
	}
}

// Test 1.2: GetArrayType error cases
func TestGetArrayType_ErrorCases(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{K: []int{10}})
	defer kp.Free()

	// Test non-existent array
	_, err := kp.GetArrayType("non_existent")
	if err == nil {
		t.Error("Expected error for non-existent array")
	}
}

// Test 1.3: GetArrayLogicalSize basic functionality
func TestGetArrayLogicalSize_BasicFunctionality(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{K: []int{10}})
	defer kp.Free()

	// Test arrays of different sizes
	testCases := []struct {
		name         string
		size         int64
		dataType     builder.DataType
		expectedSize int
	}{
		{"small_float64", 10 * 8, builder.Float64, 10},
		{"large_float64", 100 * 8, builder.Float64, 100},
		{"small_float32", 20 * 4, builder.Float32, 20},
		{"small_int32", 15 * 4, builder.INT32, 15},
		{"small_int64", 25 * 8, builder.INT64, 25},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			spec := builder.ArraySpec{
				Name:      tc.name,
				Size:      tc.size,
				DataType:  tc.dataType,
				Alignment: builder.NoAlignment,
			}

			err := kp.AllocateArrays([]builder.ArraySpec{spec})
			if err != nil {
				t.Fatalf("Failed to allocate: %v", err)
			}

			size, err := kp.GetArrayLogicalSize(tc.name)
			if err != nil {
				t.Errorf("GetArrayLogicalSize failed: %v", err)
			}
			if size != tc.expectedSize {
				t.Errorf("Expected size %d, got %d", tc.expectedSize, size)
			}
		})
	}
}

// ============================================================================
// Section 2: Progressive Complexity - Multiple Partitions
// Following Unit Testing Principle: Build systematically
// ============================================================================

// Test 2.1: GetArrayLogicalSize with multiple partitions
func TestGetArrayLogicalSize_MultiplePartitions(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	// Test incrementally: 1, 2, 3, ... partitions
	testCases := []struct {
		name string
		k    []int
	}{
		{"2_partitions", []int{3, 4}},
		{"3_partitions", []int{2, 3, 4}},
		{"4_partitions", []int{1, 2, 3, 4}},
		{"5_partitions", []int{1, 1, 1, 1, 1}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			kp := NewRunner(device, builder.Config{K: tc.k})
			defer kp.Free()

			totalK := 0
			for _, k := range tc.k {
				totalK += k
			}

			// Allocate with total size = totalK elements * 8 bytes each
			spec := builder.ArraySpec{
				Name:      "test_array",
				Size:      int64(totalK * 8),
				DataType:  builder.Float64,
				Alignment: builder.NoAlignment,
			}

			err := kp.AllocateArrays([]builder.ArraySpec{spec})
			if err != nil {
				t.Fatalf("Failed to allocate: %v", err)
			}

			size, err := kp.GetArrayLogicalSize("test_array")
			if err != nil {
				t.Errorf("GetArrayLogicalSize failed: %v", err)
			}

			// Mathematical property: size = total bytes / element size
			if size != totalK {
				t.Errorf("Expected size %d, got %d", totalK, size)
			}
		})
	}
}

// Test 2.2: CopyArrayToHost single partition
func TestCopyArrayToHost_SinglePartition(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K:         []int{5},
		FloatType: builder.Float64,
	})
	defer kp.Free()

	// Allocate and initialize array
	spec := builder.ArraySpec{
		Name:      "data",
		Size:      5 * 8,
		DataType:  builder.Float64,
		Alignment: builder.NoAlignment,
	}
	err := kp.AllocateArrays([]builder.ArraySpec{spec})
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Write test data
	testData := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	mem := kp.GetMemory("data")
	mem.CopyFrom(unsafe.Pointer(&testData[0]), int64(5*8))

	// Copy back using generic method
	result, err := CopyArrayToHost[float64](kp, "data")
	if err != nil {
		t.Fatalf("CopyArrayToHost failed: %v", err)
	}

	// Verify
	if len(result) != 5 {
		t.Errorf("Expected 5 elements, got %d", len(result))
	}

	for i := 0; i < 5; i++ {
		if math.Abs(result[i]-testData[i]) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, testData[i], result[i])
		}
	}
}

// ============================================================================
// Section 3: Type Safety and Verification
// Following Unit Testing Principle: Specific property testing
// ============================================================================

// Test 3.1: CopyArrayToHost type verification
func TestCopyArrayToHost_TypeVerification(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{K: []int{10}})
	defer kp.Free()

	// Allocate Float64 array
	spec := builder.ArraySpec{
		Name:      "float64_data",
		Size:      10 * 8,
		DataType:  builder.Float64,
		Alignment: builder.NoAlignment,
	}
	err := kp.AllocateArrays([]builder.ArraySpec{spec})
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Try to copy as wrong type (float32)
	_, err = CopyArrayToHost[float32](kp, "float64_data")
	if err == nil {
		t.Error("Expected type mismatch error")
	}

	// Try to copy as int32
	_, err = CopyArrayToHost[int32](kp, "float64_data")
	if err == nil {
		t.Error("Expected type mismatch error")
	}

	// Correct type should work
	_, err = CopyArrayToHost[float64](kp, "float64_data")
	if err != nil {
		t.Errorf("Correct type failed: %v", err)
	}
}

// Test 3.2: All supported types
func TestCopyArrayToHost_AllTypes(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{K: []int{5}})
	defer kp.Free()

	// Test all data types
	testCases := []struct {
		name     string
		dataType builder.DataType
		size     int64
		copyFunc func() (interface{}, error)
	}{
		{
			name:     "float32",
			dataType: builder.Float32,
			size:     5 * 4,
			copyFunc: func() (interface{}, error) {
				return CopyArrayToHost[float32](kp, "float32")
			},
		},
		{
			name:     "float64",
			dataType: builder.Float64,
			size:     5 * 8,
			copyFunc: func() (interface{}, error) {
				return CopyArrayToHost[float64](kp, "float64")
			},
		},
		{
			name:     "int32",
			dataType: builder.INT32,
			size:     5 * 4,
			copyFunc: func() (interface{}, error) {
				return CopyArrayToHost[int32](kp, "int32")
			},
		},
		{
			name:     "int64",
			dataType: builder.INT64,
			size:     5 * 8,
			copyFunc: func() (interface{}, error) {
				return CopyArrayToHost[int64](kp, "int64")
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			spec := builder.ArraySpec{
				Name:      tc.name,
				Size:      tc.size,
				DataType:  tc.dataType,
				Alignment: builder.NoAlignment,
			}
			err := kp.AllocateArrays([]builder.ArraySpec{spec})
			if err != nil {
				t.Fatalf("Failed to allocate: %v", err)
			}

			result, err := tc.copyFunc()
			if err != nil {
				t.Errorf("Copy failed: %v", err)
			}
			if result == nil {
				t.Error("Result is nil")
			}
		})
	}
}

// ============================================================================
// Section 4: Alignment and Padding Removal
// Following Unit Testing Principle: Test specific mathematical properties
// ============================================================================

// Test 4.1: CopyArrayToHost removes padding correctly
func TestCopyArrayToHost_PaddingRemoval(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	// Use odd-sized partitions to ensure padding
	k := []int{3, 5, 7}
	totalElements := 15

	kp := NewRunner(device, builder.Config{
		K:         k,
		FloatType: builder.Float64,
	})
	defer kp.Free()

	// Allocate with 64-byte alignment
	spec := builder.ArraySpec{
		Name:      "aligned_data",
		Size:      int64(totalElements * 8),
		DataType:  builder.Float64,
		Alignment: builder.CacheLineAlign,
	}
	err := kp.AllocateArrays([]builder.ArraySpec{spec})
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Get offsets to understand padding
	offsetsMem := kp.PooledMemory["aligned_data_offsets"]
	numOffsets := len(k) + 1
	offsets := make([]int64, numOffsets)
	offsetsMem.CopyTo(unsafe.Pointer(&offsets[0]), int64(numOffsets*8))

	// Write test pattern that includes padding areas
	_, totalSize := kp.CalculateAlignedOffsetsAndSize(spec)
	paddedData := make([]float64, totalSize/8)

	// Fill with sentinel values
	for i := range paddedData {
		paddedData[i] = -999.0
	}

	// Fill actual data areas with sequential values
	dataValue := 0.0
	for part := 0; part < len(k); part++ {
		startIdx := offsets[part]
		for elem := 0; elem < k[part]; elem++ {
			paddedData[startIdx+int64(elem)] = dataValue
			dataValue++
		}
	}

	// Write to Device
	mem := kp.GetMemory("aligned_data")
	mem.CopyFrom(unsafe.Pointer(&paddedData[0]), totalSize)

	// Copy back without padding
	result, err := CopyArrayToHost[float64](kp, "aligned_data")
	if err != nil {
		t.Fatalf("CopyArrayToHost failed: %v", err)
	}

	// Verify: should have exactly totalElements, no padding
	if len(result) != totalElements {
		t.Errorf("Expected %d elements, got %d", totalElements, len(result))
	}

	// Verify sequential values with no gaps
	for i := 0; i < totalElements; i++ {
		if math.Abs(result[i]-float64(i)) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, float64(i), result[i])
		}
	}
}

// ============================================================================
// Section 5: CopyPartitionToHost Tests
// Following Unit Testing Principle: Incremental validation
// ============================================================================

// Test 5.1: CopyPartitionToHost single partition
func TestCopyPartitionToHost_SinglePartition(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{K: []int{5}})
	defer kp.Free()

	spec := builder.ArraySpec{
		Name:      "data",
		Size:      5 * 8,
		DataType:  builder.Float64,
		Alignment: builder.NoAlignment,
	}
	err := kp.AllocateArrays([]builder.ArraySpec{spec})
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Write test data
	testData := []float64{10, 20, 30, 40, 50}
	mem := kp.GetMemory("data")
	mem.CopyFrom(unsafe.Pointer(&testData[0]), int64(5*8))

	// Copy partition 0
	result, err := CopyPartitionToHost[float64](kp, "data", 0)
	if err != nil {
		t.Fatalf("CopyPartitionToHost failed: %v", err)
	}

	// Verify
	if len(result) != 5 {
		t.Errorf("Expected 5 elements, got %d", len(result))
	}

	for i := 0; i < 5; i++ {
		if math.Abs(result[i]-testData[i]) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, testData[i], result[i])
		}
	}
}

// Test 5.2: CopyPartitionToHost multiple partitions incrementally
func TestCopyPartitionToHost_MultiplePartitions(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	k := []int{3, 4, 5}
	kp := NewRunner(device, builder.Config{
		K:         k,
		FloatType: builder.Float64,
	})
	defer kp.Free()

	totalElements := 12
	spec := builder.ArraySpec{
		Name:      "data",
		Size:      int64(totalElements * 8),
		DataType:  builder.Float64,
		Alignment: builder.NoAlignment,
	}
	err := kp.AllocateArrays([]builder.ArraySpec{spec})
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Write unique values to each partition
	fullData := make([]float64, totalElements)
	idx := 0
	for part, partK := range k {
		for elem := 0; elem < partK; elem++ {
			fullData[idx] = float64(part*100 + elem)
			idx++
		}
	}

	mem := kp.GetMemory("data")
	mem.CopyFrom(unsafe.Pointer(&fullData[0]), int64(totalElements*8))

	// Test each partition
	for partID, partK := range k {
		t.Run(fmt.Sprintf("partition_%d", partID), func(t *testing.T) {
			result, err := CopyPartitionToHost[float64](kp, "data", partID)
			if err != nil {
				t.Fatalf("CopyPartitionToHost failed: %v", err)
			}

			// Verify size
			if len(result) != partK {
				t.Errorf("Expected %d elements, got %d", partK, len(result))
			}

			// Verify values
			for elem := 0; elem < partK; elem++ {
				expected := float64(partID*100 + elem)
				if math.Abs(result[elem]-expected) > 1e-10 {
					t.Errorf("Partition %d, element %d: expected %f, got %f",
						partID, elem, expected, result[elem])
				}
			}
		})
	}
}

// Test 5.3: CopyPartitionToHost with alignment
func TestCopyPartitionToHost_WithAlignment(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	k := []int{3, 5, 7} // Odd sizes to test padding
	kp := NewRunner(device, builder.Config{
		K:         k,
		FloatType: builder.Float64,
	})
	defer kp.Free()

	totalElements := 15
	spec := builder.ArraySpec{
		Name:      "aligned_data",
		Size:      int64(totalElements * 8),
		DataType:  builder.Float64,
		Alignment: builder.CacheLineAlign,
	}
	err := kp.AllocateArrays([]builder.ArraySpec{spec})
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Get aligned memory layout info
	_, totalSize := kp.CalculateAlignedOffsetsAndSize(spec)
	offsetsMem := kp.PooledMemory["aligned_data_offsets"]
	offsets := make([]int64, len(k)+1)
	offsetsMem.CopyTo(unsafe.Pointer(&offsets[0]), int64((len(k)+1)*8))

	// Create padded buffer with test data
	paddedBuffer := make([]float64, totalSize/8)

	// Fill with sentinel values
	for i := range paddedBuffer {
		paddedBuffer[i] = -1.0
	}

	// Write actual data
	for part, partK := range k {
		startIdx := offsets[part]
		for elem := 0; elem < partK; elem++ {
			paddedBuffer[startIdx+int64(elem)] = float64(part*1000 + elem)
		}
	}

	// Write to Device
	mem := kp.GetMemory("aligned_data")
	mem.CopyFrom(unsafe.Pointer(&paddedBuffer[0]), totalSize)

	// Test each partition copy
	for partID, partK := range k {
		result, err := CopyPartitionToHost[float64](kp, "aligned_data", partID)
		if err != nil {
			t.Fatalf("CopyPartitionToHost failed for partition %d: %v", partID, err)
		}

		// Verify correct data extraction despite padding
		if len(result) != partK {
			t.Errorf("Partition %d: expected %d elements, got %d", partID, partK, len(result))
		}

		for elem := 0; elem < partK; elem++ {
			expected := float64(partID*1000 + elem)
			if math.Abs(result[elem]-expected) > 1e-10 {
				t.Errorf("Partition %d, element %d: expected %f, got %f",
					partID, elem, expected, result[elem])
			}
		}
	}
}

// ============================================================================
// Section 6: Edge Cases and Error Conditions
// Following Unit Testing Principle: Defensive testing
// ============================================================================

// Test 6.1: Invalid partition ID
func TestCopyPartitionToHost_InvalidPartitionID(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{K: []int{5, 10}})
	defer kp.Free()

	spec := builder.ArraySpec{
		Name:      "data",
		Size:      15 * 8,
		DataType:  builder.Float64,
		Alignment: builder.NoAlignment,
	}
	err := kp.AllocateArrays([]builder.ArraySpec{spec})
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Test negative partition ID
	_, err = CopyPartitionToHost[float64](kp, "data", -1)
	if err == nil {
		t.Error("Expected error for negative partition ID")
	}

	// Test partition ID >= NumPartitions
	_, err = CopyPartitionToHost[float64](kp, "data", 2)
	if err == nil {
		t.Error("Expected error for partition ID >= NumPartitions")
	}

	// Test large partition ID
	_, err = CopyPartitionToHost[float64](kp, "data", 100)
	if err == nil {
		t.Error("Expected error for large partition ID")
	}
}

// Test 6.2: Degenerate cases
func TestCopyMethods_DegenerateCases(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	// Empty partition (K[i] = 0)
	kp := NewRunner(device, builder.Config{K: []int{0, 5, 0}})
	defer kp.Free()

	totalElements := 5
	spec := builder.ArraySpec{
		Name:      "data",
		Size:      int64(totalElements * 8),
		DataType:  builder.Float64,
		Alignment: builder.NoAlignment,
	}
	err := kp.AllocateArrays([]builder.ArraySpec{spec})
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Test CopyPartitionToHost on empty partition
	result, err := CopyPartitionToHost[float64](kp, "data", 0)
	if err != nil {
		t.Fatalf("CopyPartitionToHost failed on empty partition: %v", err)
	}
	if len(result) != 0 {
		t.Errorf("Expected 0 elements for empty partition, got %d", len(result))
	}

	// Test GetArrayLogicalSize
	size, err := kp.GetArrayLogicalSize("data")
	if err != nil {
		t.Fatalf("GetArrayLogicalSize failed: %v", err)
	}
	if size != totalElements {
		t.Errorf("Expected size %d, got %d", totalElements, size)
	}
}

// ============================================================================
// Section 7: Integration Tests with Different Offset Types
// Following Unit Testing Principle: Real-world scenarios
// ============================================================================

// Test 7.1: Test with Int32 offsets
func TestCopyMethods_Int32Offsets(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K:       []int{100, 200, 150},
		IntType: builder.INT32, // Force 32-bit offsets
	})
	defer kp.Free()

	totalElements := 450
	spec := builder.ArraySpec{
		Name:      "large_data",
		Size:      int64(totalElements * 8),
		DataType:  builder.Float64,
		Alignment: builder.NoAlignment,
	}
	err := kp.AllocateArrays([]builder.ArraySpec{spec})
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Initialize with sequential values
	testData := make([]float64, totalElements)
	for i := 0; i < totalElements; i++ {
		testData[i] = float64(i)
	}
	mem := kp.GetMemory("large_data")
	mem.CopyFrom(unsafe.Pointer(&testData[0]), int64(totalElements*8))

	// Test CopyArrayToHost
	result, err := CopyArrayToHost[float64](kp, "large_data")
	if err != nil {
		t.Fatalf("CopyArrayToHost failed: %v", err)
	}

	// Verify all data
	for i := 0; i < totalElements; i++ {
		if math.Abs(result[i]-float64(i)) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, float64(i), result[i])
		}
	}

	// Test partition copy with int32 offsets
	part1Result, err := CopyPartitionToHost[float64](kp, "large_data", 1)
	if err != nil {
		t.Fatalf("CopyPartitionToHost failed: %v", err)
	}
	if len(part1Result) != 200 {
		t.Errorf("Expected 200 elements in partition 1, got %d", len(part1Result))
	}
}

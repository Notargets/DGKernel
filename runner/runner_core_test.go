package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"math"
	"testing"
)

// ============================================================================
// Section 1: Basic Creation and Configuration Tests
// These remain largely unchanged as they test core functionality
// ============================================================================

func TestRunner_Creation(t *testing.T) {
	// Test nil Device
	t.Run("NilDevice", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic for nil Device")
			}
		}()
		NewRunner(nil, builder.Config{K: []int{10}})
	})

	// Test empty K array
	t.Run("EmptyKArray", func(t *testing.T) {
		device := utils.CreateTestDevice()
		defer device.Free()

		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic for empty K array")
			}
		}()
		NewRunner(device, builder.Config{K: []int{}})
	})
}

func TestRunner_SinglePartition(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K:         []int{100},
		FloatType: builder.Float64,
		IntType:   builder.INT64,
	})
	defer kp.Free()

	// Verify basic properties
	if kp.NumPartitions != 1 {
		t.Errorf("Expected NumPartitions=1, got %d", kp.NumPartitions)
	}
	if kp.K[0] != 100 {
		t.Errorf("Expected K[0]=100, got %d", kp.K[0])
	}
	if kp.KpartMax != 100 {
		t.Errorf("Expected KpartMax=100, got %d", kp.KpartMax)
	}
}

func TestRunner_KpartMaxComputation(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	testCases := []struct {
		name         string
		k            []int
		expectedKMax int
	}{
		{"uniform", []int{10, 10, 10}, 10},
		{"ascending", []int{5, 10, 15, 20}, 20},
		{"descending", []int{20, 15, 10, 5}, 20},
		{"mixed", []int{10, 25, 15, 30, 20}, 30},
		{"single_large", []int{5, 5, 100, 5}, 100},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			kp := NewRunner(device, builder.Config{K: tc.k})
			defer kp.Free()

			if kp.KpartMax != tc.expectedKMax {
				t.Errorf("Expected KpartMax=%d, got %d", tc.expectedKMax, kp.KpartMax)
			}
		})
	}
}

// ============================================================================
// Section 2: Kernel Execution with New API
// Replaces old explicit allocation tests
// ============================================================================

func TestRunner_BasicComputation(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{K: []int{10}})
	defer kp.Free()

	// Host data
	hostData := make([]float64, 10)
	for i := range hostData {
		hostData[i] = float64(i)
	}

	// Define kernel with new API
	err := kp.DefineKernel("setValues",
		InOut("data").Bind(hostData).Copy(), // Copy in and out
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	// Get signature for kernel
	signature, _ := kp.GetKernelSignature("setValues")

	// Simple kernel that squares values
	kernelSource := fmt.Sprintf(`
@kernel void setValues(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		real_t* data = data_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				data[i] = data[i] * data[i];
			}
		}
	}
}`, signature)

	_, err = kp.BuildKernel(kernelSource, "setValues")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Execute kernel
	err = kp.RunKernel("setValues")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify results - data should be squared due to Copy()
	for i := 0; i < 10; i++ {
		expected := float64(i * i)
		if math.Abs(hostData[i]-expected) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, hostData[i])
		}
	}
}

func TestRunner_MultiplePartitions(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	k := []int{5, 10, 8}
	totalElements := 23

	kp := NewRunner(device, builder.Config{
		K:         k,
		FloatType: builder.Float64,
	})
	defer kp.Free()

	// Host arrays
	hostA := make([]float64, totalElements)
	hostB := make([]float64, totalElements)
	hostC := make([]float64, totalElements)

	// Initialize
	for i := range hostA {
		hostA[i] = float64(i)
		hostB[i] = float64(i * 2)
	}

	// Define kernel
	err := kp.DefineKernel("vectorAdd",
		Input("A").Bind(hostA).CopyTo(),
		Input("B").Bind(hostB).CopyTo(),
		Output("C").Bind(hostC),
		Scalar("alpha").Bind(1.5),
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignature("vectorAdd")

	kernelSource := fmt.Sprintf(`
@kernel void vectorAdd(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* A = A_PART(part);
		const real_t* B = B_PART(part);
		real_t* C = C_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				C[i] = alpha * A[i] + B[i];
			}
		}
	}
}`, signature)

	_, err = kp.BuildKernel(kernelSource, "vectorAdd")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.RunKernel("vectorAdd")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Copy result back
	result, err := CopyArrayToHost[float64](kp, "C")
	if err != nil {
		t.Fatalf("Failed to copy result: %v", err)
	}

	// Verify across all partitions
	for i := 0; i < totalElements; i++ {
		expected := 1.5*float64(i) + float64(i*2)
		if math.Abs(result[i]-expected) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, result[i])
		}
	}
}

// ============================================================================
// Section 3: Edge Cases and Error Handling
// Ported from copy tests
// ============================================================================

func TestRunner_EmptyPartition(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	// K array with empty partition
	k := []int{0, 5, 0, 3}
	kp := NewRunner(device, builder.Config{K: k})
	defer kp.Free()

	totalElements := 8
	hostData := make([]float64, totalElements)

	// Define kernel
	err := kp.DefineKernel("processData",
		InOut("data").Bind(hostData).Copy(),
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignature("processData")

	// Kernel that increments non-empty partitions
	kernelSource := fmt.Sprintf(`
@kernel void processData(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		real_t* data = data_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				data[i] += 1.0;
			}
		}
	}
}`, signature)

	_, err = kp.BuildKernel(kernelSource, "processData")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Initialize data
	idx := 0
	for partID, partK := range k {
		for i := 0; i < partK; i++ {
			hostData[idx] = float64(partID*100 + i)
			idx++
		}
	}

	err = kp.RunKernel("processData")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Test partition copies including empty ones
	for partID, partK := range k {
		result, err := CopyPartitionToHost[float64](kp, "data", partID)
		if err != nil {
			t.Fatalf("Failed to copy partition %d: %v", partID, err)
		}

		if len(result) != partK {
			t.Errorf("Partition %d: expected %d elements, got %d", partID, partK, len(result))
		}

		// Verify data in non-empty partitions
		for i := 0; i < partK; i++ {
			expected := float64(partID*100+i) + 1.0 // incremented by kernel
			if math.Abs(result[i]-expected) > 1e-10 {
				t.Errorf("Partition %d, element %d: expected %f, got %f",
					partID, i, expected, result[i])
			}
		}
	}
}

func TestRunner_TypeMismatch(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K:         []int{10},
		FloatType: builder.Float64,
	})
	defer kp.Free()

	// Define with float64
	hostData := make([]float64, 10)
	err := kp.DefineKernel("test",
		Input("data").Bind(hostData).CopyTo(),
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	// Try to copy as wrong type
	_, err = CopyArrayToHost[float32](kp, "data")
	if err == nil {
		t.Error("Expected type mismatch error")
	}

	// Correct type should work
	_, err = CopyArrayToHost[float64](kp, "data")
	if err != nil {
		t.Errorf("Correct type failed: %v", err)
	}
}

// ============================================================================
// Section 4: CUDA-specific Tests
// ============================================================================

func TestRunner_CUDAInnerLimit(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	if device.Mode() != "CUDA" {
		t.Skip("Skipping CUDA-specific test")
	}

	// Should panic with K > 1024 on CUDA
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for K > 1024 on CUDA")
		}
	}()

	NewRunner(device, builder.Config{K: []int{100, 1025, 500}})
}

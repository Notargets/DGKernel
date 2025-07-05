package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"math"
	"testing"
)

// ============================================================================
// Data Movement Tests - Testing copy semantics with new API
// ============================================================================

func TestRunner_CopySemantics(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{K: []int{10}})
	defer kp.Free()

	t.Run("CopyTo_Only", func(t *testing.T) {
		hostData := make([]float64, 10)
		for i := range hostData {
			hostData[i] = float64(i)
		}

		err := kp.DefineKernel("copyto_test",
			Input("data").Bind(hostData).CopyTo(), // Only copies to device
		)
		if err != nil {
			t.Fatalf("Failed to define kernel: %v", err)
		}

		// Run kernel that modifies data
		signature, _ := kp.GetKernelSignature("copyto_test")
		kernelSource := fmt.Sprintf(`
@kernel void copyto_test(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* data = data_PART(part);
		// Read-only access
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				// Just read, don't modify
				volatile real_t val = data[i];
			}
		}
	}
}`, signature)

		kp.BuildKernel(kernelSource, "copyto_test")
		kp.RunKernel("copyto_test")

		// Host data should be unchanged
		for i := range hostData {
			if hostData[i] != float64(i) {
				t.Errorf("Host data modified at %d: expected %f, got %f",
					i, float64(i), hostData[i])
			}
		}
	})

	t.Run("Copy_Bidirectional", func(t *testing.T) {
		hostData := make([]float64, 10)
		for i := range hostData {
			hostData[i] = float64(i)
		}

		err := kp.DefineKernel("copy_test",
			InOut("data").Bind(hostData).Copy(), // Copies both ways
		)
		if err != nil {
			t.Fatalf("Failed to define kernel: %v", err)
		}

		signature, _ := kp.GetKernelSignature("copy_test")
		kernelSource := fmt.Sprintf(`
@kernel void copy_test(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		real_t* data = data_PART(part);
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				data[i] = data[i] * 2.0;
			}
		}
	}
}`, signature)

		kp.BuildKernel(kernelSource, "copy_test")
		kp.RunKernel("copy_test")

		// Host data should be automatically updated
		for i := range hostData {
			expected := float64(i) * 2.0
			if math.Abs(hostData[i]-expected) > 1e-10 {
				t.Errorf("Copy back failed at %d: expected %f, got %f",
					i, expected, hostData[i])
			}
		}
	})

	t.Run("NoCopy", func(t *testing.T) {
		hostData := make([]float64, 10)

		err := kp.DefineKernel("nocopy_test",
			Output("data").Bind(hostData).NoCopy(),
		)
		if err != nil {
			t.Fatalf("Failed to define kernel: %v", err)
		}

		// Write to device
		signature, _ := kp.GetKernelSignature("nocopy_test")
		kernelSource := fmt.Sprintf(`
@kernel void nocopy_test(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		real_t* data = data_PART(part);
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				data[i] = (real_t)(i * i);
			}
		}
	}
}`, signature)

		kp.BuildKernel(kernelSource, "nocopy_test")
		kp.RunKernel("nocopy_test")

		// Host should still be zeros
		for i := range hostData {
			if hostData[i] != 0 {
				t.Errorf("NoCopy failed - host modified at %d", i)
			}
		}

		// But device should have data - verify with explicit copy
		result, err := CopyArrayToHost[float64](kp, "data")
		if err != nil {
			t.Fatalf("Failed to copy: %v", err)
		}

		for i := 0; i < 10; i++ {
			if result[i] != float64(i*i) {
				t.Errorf("Device data wrong at %d: expected %f, got %f",
					i, float64(i*i), result[i])
			}
		}
	})
}

func TestRunner_PartitionCopy(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	k := []int{3, 5, 7}
	kp := NewRunner(device, builder.Config{
		K:         k,
		FloatType: builder.Float64,
	})
	defer kp.Free()

	totalElements := 15
	hostData := make([]float64, totalElements)

	// Initialize with partition-specific patterns
	idx := 0
	for part, partK := range k {
		for elem := 0; elem < partK; elem++ {
			hostData[idx] = float64(part*100 + elem)
			idx++
		}
	}

	err := kp.DefineKernel("partition_test",
		InOut("data").Bind(hostData).Copy(),
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignature("partition_test")
	kernelSource := fmt.Sprintf(`
@kernel void partition_test(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		real_t* data = data_PART(part);
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				data[i] += 1000.0; // Add 1000 to each element
			}
		}
	}
}`, signature)

	kp.BuildKernel(kernelSource, "partition_test")
	kp.RunKernel("partition_test")

	// Test partition-wise copy
	for partID, partK := range k {
		result, err := CopyPartitionToHost[float64](kp, "data", partID)
		if err != nil {
			t.Fatalf("Failed to copy partition %d: %v", partID, err)
		}

		if len(result) != partK {
			t.Errorf("Partition %d: expected %d elements, got %d",
				partID, partK, len(result))
		}

		for elem := 0; elem < partK; elem++ {
			expected := float64(partID*100+elem) + 1000.0
			if math.Abs(result[elem]-expected) > 1e-10 {
				t.Errorf("Partition %d, element %d: expected %f, got %f",
					partID, elem, expected, result[elem])
			}
		}
	}
}

func TestRunner_TypeConversion(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K:         []int{100},
		FloatType: builder.Float32, // Device uses float32
	})
	defer kp.Free()

	// Host uses float64 for precision
	hostData64 := make([]float64, 100)
	hostResult32 := make([]float32, 100)

	// Initialize with values that show precision loss
	for i := range hostData64 {
		hostData64[i] = float64(i) * 1.123456789
	}

	err := kp.DefineKernel("convert_test",
		Input("input").Bind(hostData64).CopyTo().Convert(builder.Float32),
		Output("output").Bind(hostResult32), // No conversion needed
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignature("convert_test")
	kernelSource := fmt.Sprintf(`
@kernel void convert_test(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* input = input_PART(part);
		real_t* output = output_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				output[i] = input[i] * 2.0f;
			}
		}
	}
}`, signature)

	kp.BuildKernel(kernelSource, "convert_test")
	kp.RunKernel("convert_test")

	// Get result
	result, err := CopyArrayToHost[float32](kp, "output")
	if err != nil {
		t.Fatalf("Failed to copy result: %v", err)
	}

	// Verify with appropriate float32 precision
	for i := 0; i < 100; i++ {
		expected := float32(hostData64[i] * 2.0)
		if abs32(result[i]-expected) > 1e-6 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, result[i])
		}
	}
}

func TestRunner_WorkingArrays(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{K: []int{50}})
	defer kp.Free()

	hostInput := make([]float64, 50)
	hostOutput := make([]float64, 50)

	for i := range hostInput {
		hostInput[i] = float64(i)
	}

	// Define kernel with working arrays
	err := kp.DefineKernel("working_test",
		Input("input").Bind(hostInput).CopyTo(),
		Output("output").Bind(hostOutput),
		Temp("scratch1").Type(builder.Float64).Size(50),
		Temp("scratch2").Type(builder.Float64).Size(50),
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignature("working_test")
	kernelSource := fmt.Sprintf(`
@kernel void working_test(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* input = input_PART(part);
		real_t* output = output_PART(part);
		real_t* scratch1 = scratch1_PART(part);
		real_t* scratch2 = scratch2_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				// Multi-stage computation using scratch arrays
				scratch1[i] = input[i] * 2.0;
				scratch2[i] = scratch1[i] + 10.0;
				output[i] = scratch2[i] * scratch1[i];
			}
		}
	}
}`, signature)

	kp.BuildKernel(kernelSource, "working_test")
	kp.RunKernel("working_test")

	result, err := CopyArrayToHost[float64](kp, "output")
	if err != nil {
		t.Fatalf("Failed to copy result: %v", err)
	}

	// Verify: output = (input*2 + 10) * (input*2)
	for i := 0; i < 50; i++ {
		scratch1 := hostInput[i] * 2.0
		scratch2 := scratch1 + 10.0
		expected := scratch2 * scratch1
		if math.Abs(result[i]-expected) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, result[i])
		}
	}
}

// Test invalid partition access
func TestRunner_InvalidPartition(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{K: []int{5, 10}})
	defer kp.Free()

	hostData := make([]float64, 15)
	kp.DefineKernel("test",
		Input("data").Bind(hostData).CopyTo(),
	)

	// Build dummy kernel
	signature, _ := kp.GetKernelSignature("test")
	kernelSource := fmt.Sprintf(`@kernel void test(%s) {}`, signature)
	kp.BuildKernel(kernelSource, "test")
	kp.RunKernel("test")

	// Test invalid partition IDs
	_, err := CopyPartitionToHost[float64](kp, "data", -1)
	if err == nil {
		t.Error("Expected error for negative partition ID")
	}

	_, err = CopyPartitionToHost[float64](kp, "data", 2)
	if err == nil {
		t.Error("Expected error for partition ID >= NumPartitions")
	}

	_, err = CopyPartitionToHost[float64](kp, "data", 100)
	if err == nil {
		t.Error("Expected error for large partition ID")
	}
}

// Helper functions
func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

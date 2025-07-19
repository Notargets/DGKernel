// Test file to verify .DeviceMemType() functionality
package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"math"
	"testing"
)

// TestConvert_BasicFloat64ToFloat32 tests basic type conversion
func TestConvert_BasicFloat64ToFloat32(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer kp.Free()

	// Host data in float64
	hostInput := make([]float64, 10)
	hostOutput := make([]float64, 10)
	for i := range hostInput {
		hostInput[i] = float64(i) * 1.123456789 // Precision that will be lost
	}

	// Phase 1: Define bindings with conversion
	err := kp.DefineBindings(
		builder.Input("input").Bind(hostInput).DeviceMemType(builder.Float32),
		builder.Output("output").Bind(hostOutput).DeviceMemType(builder.Float32),
	)
	if err != nil {
		t.Fatalf("Failed to define bindings: %v", err)
	}

	// Phase 1: Allocate device memory
	err = kp.AllocateDevice()
	if err != nil {
		t.Fatalf("Failed to allocate device: %v", err)
	}

	// Phase 2: Configure kernel
	kernelName := "convert_test"
	_, err = kp.ConfigureKernel(kernelName,
		kp.Param("input").CopyTo(),
		kp.Param("output").CopyBack(),
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	// Get signature to verify types
	signature, err := kp.GetKernelSignatureForConfig(kernelName)
	if err != nil {
		t.Fatalf("Failed to get signature: %v", err)
	}
	t.Logf("Generated signature:\n%s", signature)

	// Simple kernel that copies data
	kernelSource := fmt.Sprintf(`
@kernel void convert_test(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const float* input = input_PART(part);
		float* output = output_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				output[i] = input[i] * 2.0f;
			}
		}
	}
}`, signature)

	_, err = kp.BuildKernel(kernelSource, kernelName)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.ExecuteKernel(kernelName)
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify results - should have precision loss from float64->float32->float64
	for i := range hostOutput {
		expected := hostInput[i] * 2.0
		// DeviceMemType to float32 and back to simulate precision loss
		expectedWithLoss := float64(float32(expected))

		if math.Abs(hostOutput[i]-expectedWithLoss) > 1e-6 {
			t.Errorf("Element %d: expected %f (with precision loss), got %f",
				i, expectedWithLoss, hostOutput[i])
		}
	}
	t.Log("✓ Float64 to Float32 conversion works correctly")
}

// TestConvert_PartitionedData tests conversion with partitioned arrays
func TestConvert_PartitionedData(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	k := []int{5, 7, 6}
	kp := NewRunner(device, builder.Config{
		K: k,
	})
	defer kp.Free()

	// Host data in float32 (partitioned)
	hostInput := [][]float32{
		make([]float32, k[0]),
		make([]float32, k[1]),
		make([]float32, k[2]),
	}
	hostOutput := [][]float32{
		make([]float32, k[0]),
		make([]float32, k[1]),
		make([]float32, k[2]),
	}

	// Initialize
	for p := range k {
		for i := 0; i < k[p]; i++ {
			hostInput[p][i] = float32(p*100 + i)
		}
	}

	// Phase 1: Define bindings with conversion for partitioned data
	err := kp.DefineBindings(
		builder.Input("input").Bind(hostInput).DeviceMemType(builder.Float64),    // DeviceMemType to float64 on device
		builder.Output("output").Bind(hostOutput).DeviceMemType(builder.Float64), // Device will be float64
	)
	if err != nil {
		t.Fatalf("Failed to define bindings: %v", err)
	}

	// Phase 1: Allocate device memory
	err = kp.AllocateDevice()
	if err != nil {
		t.Fatalf("Failed to allocate device: %v", err)
	}

	// Phase 2: Configure kernel
	kernelName := "partitioned_convert"
	_, err = kp.ConfigureKernel(kernelName,
		kp.Param("input").CopyTo(),
		kp.Param("output").CopyBack(),
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	signature, err := kp.GetKernelSignatureForConfig(kernelName)
	if err != nil {
		t.Fatalf("Failed to get signature: %v", err)
	}

	kernelSource := fmt.Sprintf(`
@kernel void partitioned_convert(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* input = input_PART(part);
		double* output = output_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				output[i] = input[i] + 1000.0;
			}
		}
	}
}`, signature)

	_, err = kp.BuildKernel(kernelSource, kernelName)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.ExecuteKernel(kernelName)
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify
	for p := range k {
		for i := 0; i < k[p]; i++ {
			expected := hostInput[p][i] + 1000.0
			if math.Abs(float64(hostOutput[p][i]-expected)) > 1e-5 {
				t.Errorf("Partition %d, element %d: expected %f, got %f",
					p, i, expected, hostOutput[p][i])
			}
		}
	}
	t.Log("✓ Partitioned data conversion works correctly")
}

// TestConvert_AllCombinations tests all type conversion combinations
func TestConvert_AllCombinations(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	testCases := []struct {
		name       string
		deviceType builder.DataType
		hostType   builder.DataType
		convertTo  builder.DataType
	}{
		{"float64_to_float32", builder.Float64, builder.Float64, builder.Float32},
		{"float32_to_float64", builder.Float32, builder.Float32, builder.Float64},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			kp := NewRunner(device, builder.Config{
				K: []int{5},
			})
			defer kp.Free()

			// Create appropriate host data based on type
			var hostData interface{}
			switch tc.hostType {
			case builder.Float64:
				data := make([]float64, 5)
				for i := range data {
					data[i] = float64(i) * 1.5
				}
				hostData = data
			case builder.Float32:
				data := make([]float32, 5)
				for i := range data {
					data[i] = float32(i) * 1.5
				}
				hostData = data
			case builder.INT64:
				data := make([]int64, 5)
				for i := range data {
					data[i] = int64(i * 10)
				}
				hostData = data
			case builder.INT32:
				data := make([]int32, 5)
				for i := range data {
					data[i] = int32(i * 10)
				}
				hostData = data
			}

			// Phase 1: Define bindings
			err := kp.DefineBindings(
				builder.InOut("data").Bind(hostData).DeviceMemType(tc.convertTo),
			)
			if err != nil {
				t.Fatalf("Failed to define bindings: %v", err)
			}

			// Phase 1: Allocate device memory
			err = kp.AllocateDevice()
			if err != nil {
				t.Fatalf("Failed to allocate device: %v", err)
			}

			// Phase 2: Configure kernel
			_, err = kp.ConfigureKernel("test",
				kp.Param("data").Copy(),
			)
			if err != nil {
				t.Fatalf("Failed to configure kernel: %v", err)
			}

			// Verify allocation used the converted type
			binding := kp.GetBinding("data")
			if binding.DeviceType != tc.convertTo {
				t.Errorf("Expected device type %v, got %v", tc.convertTo, binding.DeviceType)
			}
		})
	}
}

// TestConvert_MixedPrecisionKernel tests kernel with mixed precision arrays
func TestConvert_MixedPrecisionKernel(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer kp.Free()

	// Different precision host arrays
	input64 := make([]float64, 10)
	weights32 := make([]float32, 10)
	output64 := make([]float64, 10)

	for i := range input64 {
		input64[i] = float64(i) * 1.1
		weights32[i] = float32(i) * 0.5
	}

	// Phase 1: Define bindings with different types
	err := kp.DefineBindings(
		builder.Input("input").Bind(input64),                                    // Keep as float64
		builder.Input("weights").Bind(weights32).DeviceMemType(builder.Float64), // DeviceMemType to float64
		builder.Output("output").Bind(output64),                                 // Keep as float64
	)
	if err != nil {
		t.Fatalf("Failed to define bindings: %v", err)
	}

	// Phase 1: Allocate device memory
	err = kp.AllocateDevice()
	if err != nil {
		t.Fatalf("Failed to allocate device: %v", err)
	}

	// Phase 2: Configure kernel
	kernelName := "mixed_precision"
	_, err = kp.ConfigureKernel(kernelName,
		kp.Param("input").CopyTo(),
		kp.Param("weights").CopyTo(),
		kp.Param("output").CopyBack(),
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignatureForConfig(kernelName)
	kernelSource := fmt.Sprintf(`
@kernel void mixed_precision(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* input = input_PART(part);
		const double* weights = weights_PART(part);  // Converted to double
		double* output = output_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				output[i] = input[i] * weights[i];
			}
		}
	}
}`, signature)

	_, err = kp.BuildKernel(kernelSource, kernelName)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.ExecuteKernel(kernelName)
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify results
	for i := range output64 {
		expected := input64[i] * float64(weights32[i])
		if math.Abs(output64[i]-expected) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, output64[i])
		}
	}
	t.Log("✓ Mixed precision kernel works correctly")
}

// TestConvert_VerifyMemoryAllocation tests that conversion affects memory allocation
func TestConvert_VerifyMemoryAllocation(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K: []int{1000}, // Large array to make size difference clear
	})
	defer kp.Free()

	hostData := make([]float64, 1000)

	// Phase 1: Define bindings with conversion
	err := kp.DefineBindings(
		builder.Input("data_f32").Bind(hostData).DeviceMemType(builder.Float32),
		builder.Input("data_f64").Bind(hostData), // No conversion
	)
	if err != nil {
		t.Fatalf("Failed to define bindings: %v", err)
	}

	// Phase 1: Allocate device memory
	err = kp.AllocateDevice()
	if err != nil {
		t.Fatalf("Failed to allocate device: %v", err)
	}

	// Check memory allocations
	memF32 := kp.GetMemory("data_f32")
	memF64 := kp.GetMemory("data_f64")

	if memF32 == nil || memF64 == nil {
		t.Fatal("Failed to get memory allocations")
	}

	// Float32 should use half the memory of float64
	// We can't directly check size, but we can verify the binding metadata
	bindingF32 := kp.GetBinding("data_f32")
	bindingF64 := kp.GetBinding("data_f64")

	if bindingF32.ElementSize != 4 {
		t.Errorf("Float32 binding should have element size 4, got %d", bindingF32.ElementSize)
	}
	if bindingF64.ElementSize != 8 {
		t.Errorf("Float64 binding should have element size 8, got %d", bindingF64.ElementSize)
	}

	t.Log("✓ Memory allocation respects type conversion")
}

// TestConvert_RoundTripAccuracy tests accuracy of round-trip conversions
func TestConvert_RoundTripAccuracy(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K: []int{100},
	})
	defer kp.Free()

	// Test data with values that lose precision in float32
	hostInput := make([]float64, 100)
	hostOutput := make([]float64, 100)

	// Use values that demonstrate precision loss
	for i := range hostInput {
		// These values have more precision than float32 can represent
		hostInput[i] = 1.0 + float64(i)*1e-7
	}

	// Phase 1: Define bindings
	err := kp.DefineBindings(
		builder.InOut("data").Bind(hostInput).DeviceMemType(builder.Float32),
		builder.Output("output").Bind(hostOutput),
	)
	if err != nil {
		t.Fatalf("Failed to define bindings: %v", err)
	}

	// Phase 1: Allocate device memory
	err = kp.AllocateDevice()
	if err != nil {
		t.Fatalf("Failed to allocate device: %v", err)
	}

	// First kernel: Copy input to device (with conversion)
	_, err = kp.ConfigureKernel("copy_to",
		kp.Param("data").CopyTo(),
	)
	if err != nil {
		t.Fatalf("Failed to configure copy_to kernel: %v", err)
	}

	sig1, _ := kp.GetKernelSignatureForConfig("copy_to")
	kernel1Src := fmt.Sprintf(`
@kernel void copy_to(%s) {
		int ii;
		for (int i=0; i<10; ++i; @outer) {
			for (int j=0; j<10; ++j; @inner) {
				ii++;
			}
		}
	// No-op kernel just to trigger copy
}`, sig1)

	_, err = kp.BuildKernel(kernel1Src, "copy_to")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.ExecuteKernel("copy_to")
	if err != nil {
		t.Fatalf("Failed to execute copy_to: %v", err)
	}

	// Second kernel: Process on device and copy back
	_, err = kp.ConfigureKernel("process",
		kp.Param("data").CopyBack(), // Copy back with conversion
		kp.Param("output").CopyBack(),
	)
	if err != nil {
		t.Fatalf("Failed to configure process kernel: %v", err)
	}

	sig2, _ := kp.GetKernelSignatureForConfig("process")
	kernel2Src := fmt.Sprintf(`
@kernel void process(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		float* data = data_PART(part);
		double* output = output_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				// Store float32 value into float64 output
				output[i] = (double)data[i];
			}
		}
	}
}`, sig2)

	_, err = kp.BuildKernel(kernel2Src, "process")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.ExecuteKernel("process")
	if err != nil {
		t.Fatalf("Failed to execute process: %v", err)
	}

	// Verify precision loss
	precisionLossDetected := false
	for i := range hostInput {
		// hostOutput should have the float32-truncated value
		originalValue := 1.0 + float64(i)*1e-7
		float32Value := float32(originalValue)
		expectedValue := float64(float32Value)

		if math.Abs(hostOutput[i]-expectedValue) > 1e-10 {
			t.Errorf("Element %d: expected %v, got %v", i, expectedValue, hostOutput[i])
		}

		// Check that precision was actually lost
		if hostOutput[i] != originalValue && !precisionLossDetected {
			precisionLossDetected = true
			t.Logf("Precision loss detected at element %d: original %v, recovered %v",
				i, originalValue, hostOutput[i])
		}
	}

	if !precisionLossDetected {
		t.Error("Expected to detect precision loss in float64->float32->float64 conversion")
	}

	t.Log("✓ Round-trip conversion accuracy verified")
}

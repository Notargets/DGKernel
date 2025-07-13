// Test file to verify .Convert() functionality
package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"math"
	"testing"
	"unsafe"
)

// TestConvert_BasicFloat64ToFloat32 tests basic type conversion
func _TestConvert_BasicFloat66ToFloat32(t *testing.T) {
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

	// Define kernel with conversion
	err := kp.DefineKernel("convert_test",
		builder.Input("input").Bind(hostInput).CopyTo().Convert(builder.Float32),
		builder.Output("output").Bind(hostOutput).CopyBack().Convert(builder.Float32), // Changed!
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	// Get signature to verify types
	signature, _ := kp.GetKernelSignature("convert_test")
	t.Logf("Generated signature:\n%s", signature)

	// Simple kernel that copies data
	kernelSource := fmt.Sprintf(`
@kernel void convert_test(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* input = input_PART(part);
		double* output = output_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				output[i] = input[i] * 2.0f;
			}
		}
	}
}`, signature)

	_, err = kp.BuildKernel(kernelSource, "convert_test")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.RunKernel("convert_test")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify results - should have precision loss from float64->float32->float64
	for i := range hostOutput {
		expected := hostInput[i] * 2.0
		// Convert to float32 and back to simulate precision loss
		expectedWithLoss := float64(float32(expected))

		if math.Abs(hostOutput[i]-expectedWithLoss) > 1e-6 {
			t.Errorf("Element %d: expected %f (with precision loss), got %f",
				i, expectedWithLoss, hostOutput[i])
		}
	}
	t.Log("✓ Float64 to Float32 conversion works correctly")
}

// TestConvert_PartitionedData tests conversion with partitioned arrays
func _TestConvert_PartitionedData(t *testing.T) {
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

	// Define kernel with conversion for partitioned data
	err := kp.DefineKernel("partitioned_convert",
		builder.Input("input").Bind(hostInput).CopyTo().Convert(builder.Float64),      // Convert to float64 on device
		builder.Output("output").Bind(hostOutput).CopyBack().Convert(builder.Float32), // Convert back to float32
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignature("partitioned_convert")
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

	_, err = kp.BuildKernel(kernelSource, "partitioned_convert")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.RunKernel("partitioned_convert")
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
func _TestConvert_AllCombinations(t *testing.T) {
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
		{"int64_to_int32", builder.Float64, builder.INT64, builder.INT32},
		{"int32_to_int64", builder.Float64, builder.INT32, builder.INT64},
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

			err := kp.DefineKernel("test",
				builder.InOut("data").Bind(hostData).Copy().Convert(tc.convertTo),
			)
			if err != nil {
				t.Fatalf("Failed to define kernel: %v", err)
			}

			// Verify allocation used the converted type
			meta, exists := kp.GetArrayMetadata("data")
			if !exists {
				t.Fatal("Array metadata not found")
			}
			if meta.dataType != tc.convertTo {
				t.Errorf("Expected device type %v, got %v", tc.convertTo, meta.dataType)
			}

			t.Logf("✓ %s conversion setup works", tc.name)
		})
	}
}

// TestConvert_MemoryEfficiency verifies that conversion saves memory
func TestConvert_MemoryEfficiency(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K: []int{1000},
	})
	defer kp.Free()

	// Large float64 array on host
	hostData := make([]float64, 1000)

	// Without conversion - would use 8KB on device
	// With conversion - uses only 4KB on device
	err := kp.DefineKernel("memory_test",
		builder.Input("data").Bind(hostData).CopyTo().Convert(builder.Float32),
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	// Check that device allocation is float32 (4 bytes per element)
	meta, _ := kp.GetArrayMetadata("data")
	if meta.dataType != builder.Float32 {
		t.Error("Conversion did not change device allocation type")
	}

	expectedSize := int64(1000 * 4) // 4 bytes per float32
	if meta.spec.Size != expectedSize {
		t.Errorf("Expected allocation size %d, got %d", expectedSize, meta.spec.Size)
	}

	t.Log("✓ Conversion reduces memory usage by 50% for float64→float32")
}

func _TestConvert_Debug(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K: []int{5}, // Smaller for easier debugging
	})
	defer kp.Free()

	// Host data in float64
	hostInput := make([]float64, 5)
	hostOutput := make([]float64, 5)
	for i := range hostInput {
		hostInput[i] = float64(i) + 0.5
	}

	t.Logf("Initial hostInput: %v", hostInput)

	// Define kernel with conversion
	err := kp.DefineKernel("debug_convert",
		builder.Input("input").Bind(hostInput).CopyTo().Convert(builder.Float32),
		builder.Output("output").Bind(hostOutput).CopyBack().Convert(builder.Float64),
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	// Check array metadata
	t.Log("\n=== Array Metadata ===")
	for name := range kp.arrayMetadata {
		meta, _ := kp.GetArrayMetadata(name)
		t.Logf("Array %s: dataType=%v, size=%d bytes, isOutput=%v",
			name, meta.dataType, meta.spec.Size, meta.isOutput)
		if meta.paramSpec != nil {
			t.Logf("  ParamSpec: DataType=%v, ConvertType=%v, EffectiveType=%v",
				meta.paramSpec.DataType, meta.paramSpec.ConvertType,
				meta.paramSpec.GetEffectiveType())
		}
	}

	// Check memory allocations
	t.Log("\n=== Memory Allocations ===")
	for key, mem := range kp.PooledMemory {
		if mem != nil {
			t.Logf("Memory %s: allocated", key)
		}
	}

	// Simple kernel that just copies
	signature, _ := kp.GetKernelSignature("debug_convert")
	t.Logf("\nGenerated signature:\n%s", signature)

	kernelSource := fmt.Sprintf(`
@kernel void debug_convert(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* input = input_PART(part);
		double* output = output_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				output[i] = input[i]; // Just copy
			}
		}
	}
}`, signature)

	_, err = kp.BuildKernel(kernelSource, "debug_convert")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Now let's trace the copy operations
	t.Log("\n=== Running Kernel ===")

	// Manually perform the pre-kernel copy to add logging
	def := kp.kernelDefinitions["debug_convert"]
	for _, param := range def.Parameters {
		if param.Name == "input" && param.NeedsCopyTo() {
			t.Logf("Copying %s to device: needsConversion=%v, hostType=%v, deviceType=%v",
				param.Name, param.ConvertType != 0, param.DataType, param.GetEffectiveType())

			// Check what copyToDeviceWithConversion will do
			mem := kp.GetMemory(param.Name)
			_ = mem
			if param.ConvertType != 0 && param.ConvertType != param.DataType {
				t.Logf("Will perform conversion from %v to %v", param.DataType, param.ConvertType)
			}
		}
	}

	err = kp.RunKernel("debug_convert")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	t.Logf("\nOutput after kernel: %v", hostOutput)

	// Let's also manually read from device to see what's there
	t.Log("\n=== Manual Device Read ===")
	outputMem := kp.GetMemory("output")
	if outputMem != nil {
		// Read as float32 (what's on device)
		deviceData32 := make([]float32, 5)
		outputMem.CopyTo(unsafe.Pointer(&deviceData32[0]), int64(5*4))
		t.Logf("Device data (as float32): %v", deviceData32)

		// Read as float64 (wrong, but let's see)
		deviceData64 := make([]float64, 5)
		outputMem.CopyTo(unsafe.Pointer(&deviceData64[0]), int64(5*8))
		t.Logf("Device data (as float64): %v", deviceData64)
	}
}

// Test without conversion to verify basic functionality
func _TestConvert_NoConversion(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K: []int{5},
	})
	defer kp.Free()

	// Host data in float32 (matches device)
	hostInput := make([]float32, 5)
	hostOutput := make([]float32, 5)
	for i := range hostInput {
		hostInput[i] = float32(i) + 0.5
	}

	err := kp.DefineKernel("no_convert",
		builder.Input("input").Bind(hostInput).CopyTo(),      // No conversion
		builder.Output("output").Bind(hostOutput).CopyBack(), // No conversion
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignature("no_convert")
	kernelSource := fmt.Sprintf(`
@kernel void no_convert(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* input = input_PART(part);
		double* output = output_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				output[i] = input[i] * 2.0f;
			}
		}
	}
}`, signature)

	_, err = kp.BuildKernel(kernelSource, "no_convert")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.RunKernel("no_convert")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify
	for i := range hostOutput {
		expected := hostInput[i] * 2.0
		if hostOutput[i] != expected {
			t.Errorf("Element %d: expected %f, got %f", i, expected, hostOutput[i])
		}
	}
	t.Log("✓ No conversion case works correctly")
}

// TestConvert_PartitionedDataDebug - Debug version to trace the issue
func _TestConvert_PartitionedDataDebug(t *testing.T) {
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

	// Initialize with distinct values
	for p := range k {
		for i := 0; i < k[p]; i++ {
			hostInput[p][i] = float32(p*100 + i)
		}
	}

	t.Logf("Initial hostInput:")
	for p := range k {
		t.Logf("  Partition %d: %v", p, hostInput[p])
	}

	// Define kernel with conversion for partitioned data
	err := kp.DefineKernel("partitioned_convert",
		builder.Input("input").Bind(hostInput).CopyTo().Convert(builder.Float64),      // Convert to float64 on device
		builder.Output("output").Bind(hostOutput).CopyBack().Convert(builder.Float32), // Device should use float32
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	// Check array metadata
	t.Log("\n=== Array Metadata ===")
	for name := range kp.arrayMetadata {
		meta, _ := kp.GetArrayMetadata(name)
		t.Logf("Array %s: dataType=%v, size=%d bytes",
			name, meta.dataType, meta.spec.Size)
		if meta.paramSpec != nil {
			t.Logf("  ParamSpec: DataType=%v, ConvertType=%v, EffectiveType=%v",
				meta.paramSpec.DataType, meta.paramSpec.ConvertType,
				meta.paramSpec.GetEffectiveType())
		}
	}

	// Check offsets
	t.Log("\n=== Partition Offsets ===")
	inputOffsets, _ := kp.readPartitionOffsets("input")
	outputOffsets, _ := kp.readPartitionOffsets("output")
	t.Logf("Input offsets: %v", inputOffsets)
	t.Logf("Output offsets: %v", outputOffsets)

	signature, _ := kp.GetKernelSignature("partitioned_convert")
	t.Logf("\nGenerated signature:\n%s", signature)

	// Simple kernel that adds 1000
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

	_, err = kp.BuildKernel(kernelSource, "partitioned_convert")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Run kernel
	err = kp.RunKernel("partitioned_convert")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Manual device read to check what's actually there
	t.Log("\n=== Manual Device Read ===")
	outputMem := kp.GetMemory("output")
	if outputMem != nil {
		// Read partition 0 as float32 (what should be on device)
		deviceData32 := make([]float32, k[0])
		outputMem.CopyTo(unsafe.Pointer(&deviceData32[0]), int64(k[0]*4))
		t.Logf("Partition 0 data (as float32): %v", deviceData32)

		// Also check the raw memory pattern
		rawBytes := make([]byte, 32) // First 32 bytes
		outputMem.CopyTo(unsafe.Pointer(&rawBytes[0]), 32)
		t.Logf("First 32 bytes of output memory: %x", rawBytes)
	}

	// Check results
	t.Log("\n=== Results ===")
	for p := range k {
		t.Logf("Partition %d output: %v", p, hostOutput[p])
	}
}

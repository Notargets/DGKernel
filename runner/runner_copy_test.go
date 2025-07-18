package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"gonum.org/v1/gonum/mat"
	"math"
	"testing"
)

// ============================================================================
// Data Movement Tests - Testing copy semantics with new API
// ============================================================================

func TestRunner_CopySemantics(t *testing.T) {
	// device := utils.CreateTestDevice()
	device := utils.CreateTestDevice(true)
	defer device.Free()

	kp := NewRunner(device, builder.Config{K: []int{10}})
	defer kp.Free()

	t.Run("CopyTo_Only", func(t *testing.T) {
		hostData := make([]float64, 10)
		for i := range hostData {
			hostData[i] = float64(i)
		}

		// Phase 1: Define bindings
		err := kp.DefineBindings(
			builder.Input("data").Bind(hostData),
		)
		if err != nil {
			t.Fatalf("Failed to define bindings: %v", err)
		}

		// Phase 1: Allocate device memory
		err = kp.AllocateDevice()
		if err != nil {
			t.Fatalf("Failed to allocate device: %v", err)
		}

		// Phase 2: Configure kernel with CopyTo only
		_, err = kp.ConfigureKernel("copyto_test",
			kp.Param("data").CopyTo(), // Only copies to device
		)
		if err != nil {
			t.Fatalf("Failed to configure kernel: %v", err)
		}

		// Run kernel that modifies data
		signature, _ := kp.GetKernelSignatureForConfig("copyto_test")
		kernelSource := fmt.Sprintf(`
@kernel void copyto_test(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* data = data_PART(part);
		// Read-only access
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				// Just read, don't modify
				volatile double val = data[i];
			}
		}
	}
}`, signature)

		kp.BuildKernel(kernelSource, "copyto_test")
		kp.ExecuteKernel("copyto_test")

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

		// Phase 1: Define bindings
		err := kp.DefineBindings(
			builder.InOut("data").Bind(hostData),
		)
		if err != nil {
			t.Fatalf("Failed to define bindings: %v", err)
		}

		// Phase 1: Allocate device memory
		err = kp.AllocateDevice()
		if err != nil {
			t.Fatalf("Failed to allocate device: %v", err)
		}

		// Phase 2: Configure kernel with bidirectional copy
		_, err = kp.ConfigureKernel("copy_test",
			kp.Param("data").Copy(), // Copies both ways
		)
		if err != nil {
			t.Fatalf("Failed to configure kernel: %v", err)
		}

		signature, _ := kp.GetKernelSignatureForConfig("copy_test")
		kernelSource := fmt.Sprintf(`
@kernel void copy_test(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		double* data = data_PART(part);
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				data[i] = data[i] * 2.0;
			}
		}
	}
}`, signature)

		kp.BuildKernel(kernelSource, "copy_test")
		kp.ExecuteKernel("copy_test")

		// Host data should be automatically updated
		for i := range hostData {
			expected := float64(i) * 2.0
			if math.Abs(hostData[i]-expected) > 1e-10 {
				t.Errorf("Copy back failed at %d: expected %f, got %f",
					i, expected, hostData[i])
			}
		}
	})

	t.Run("Copy_Bidirectional_Matrix", func(t *testing.T) {
		hostData := make([]float64, 10)
		for i := range hostData {
			hostData[i] = float64(i)
		}
		hostM := mat.NewDense(len(hostData), 1, hostData)

		// Phase 1: Define bindings
		err := kp.DefineBindings(
			builder.InOut("data").Bind(hostM),
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
		_, err = kp.ConfigureKernel("copy_test",
			kp.Param("data").Copy(), // Copies both ways
		)
		if err != nil {
			t.Fatalf("Failed to configure kernel: %v", err)
		}

		signature, _ := kp.GetKernelSignatureForConfig("copy_test")
		kernelSource := fmt.Sprintf(`
@kernel void copy_test(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		double* data = data_PART(part);
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				data[i] = data[i] * 2.0;
			}
		}
	}
}`, signature)

		kp.BuildKernel(kernelSource, "copy_test")
		kp.ExecuteKernel("copy_test")

		// Host data should be automatically updated
		for i, val := range hostM.RawMatrix().Data {
			expected := float64(i) * 2.0
			if math.Abs(val-expected) > 1e-10 {
				t.Errorf("Copy back failed at %d: expected %f, got %f",
					i, expected, val)
			}
		}
	})

	t.Run("CopyBack_Matrix", func(t *testing.T) {
		hostData := make([]float64, 10)
		for i := range hostData {
			hostData[i] = float64(i)
		}
		hostM := mat.NewDense(len(hostData), 1, hostData)
		returnM := mat.NewDense(len(hostData), 1, nil)

		// Phase 1: Define bindings
		err := kp.DefineBindings(
			builder.Input("data").Bind(hostM),
			builder.Output("ret").Bind(returnM),
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
		_, err = kp.ConfigureKernel("copy_test",
			kp.Param("data").CopyTo(),
			kp.Param("ret").CopyBack(),
		)
		if err != nil {
			t.Fatalf("Failed to configure kernel: %v", err)
		}

		signature, _ := kp.GetKernelSignatureForConfig("copy_test")
		kernelSource := fmt.Sprintf(`
@kernel void copy_test(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* data = data_PART(part);
		double* ret = ret_PART(part);
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				ret[i] = data[i] * 2.0;
			}
		}
	}
}`, signature)

		kp.BuildKernel(kernelSource, "copy_test")
		kp.ExecuteKernel("copy_test")

		// Host data should be automatically updated
		for i, val := range returnM.RawMatrix().Data {
			expected := float64(i) * 2.0
			if math.Abs(val-expected) > 1e-10 {
				t.Errorf("Copy back failed at %d: expected %f, got %f",
					i, expected, val)
			}
		}
	})
}

// ============================================================================
// Partition-wise copy tests
// ============================================================================

func TestRunner_PartitionCopy(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	k := []int{5, 10, 8}
	kp := NewRunner(device, builder.Config{K: k})
	defer kp.Free()

	// Create test data
	totalSize := 0
	for _, kval := range k {
		totalSize += kval
	}

	hostData := make([]float64, totalSize)
	idx := 0
	for part, kval := range k {
		for elem := 0; elem < kval; elem++ {
			hostData[idx] = float64(part*100 + elem)
			idx++
		}
	}

	// Phase 1: Define bindings
	err := kp.DefineBindings(
		builder.InOut("data").Bind(hostData),
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
	_, err = kp.ConfigureKernel("partition_test",
		kp.Param("data").Copy(),
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignatureForConfig("partition_test")
	kernelSource := fmt.Sprintf(`
@kernel void partition_test(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		double* data = data_PART(part);
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				data[i] += 1000.0; // Add 1000 to each element
			}
		}
	}
}`, signature)

	kp.BuildKernel(kernelSource, "partition_test")
	kp.ExecuteKernel("partition_test")

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
	device := utils.CreateTestDevice(true)
	// device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K: []int{100},
	})
	defer kp.Free()

	// Host uses float64 for precision
	hostData64 := make([]float64, 100)
	hostResult32 := make([]float32, 100)

	// Initialize with values that show precision loss
	for i := range hostData64 {
		hostData64[i] = float64(i) * 1.123456789
	}

	// Phase 1: Define bindings with conversion
	err := kp.DefineBindings(
		builder.Input("input").Bind(hostData64).Convert(builder.Float32),
		builder.Output("output").Bind(hostResult32), // No conversion needed
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
	_, err = kp.ConfigureKernel("convert_test",
		kp.Param("input").CopyTo(),
		kp.Param("output").CopyBack(),
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignatureForConfig("convert_test")
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

	kp.BuildKernel(kernelSource, "convert_test")
	kp.ExecuteKernel("convert_test")

	// Verify values were converted and computed correctly
	for i := range hostResult32 {
		expected := float32(hostData64[i]) * 2.0
		if math.Abs(float64(hostResult32[i]-expected)) > 1e-6 {
			t.Errorf("Element %d: expected %f, got %f",
				i, expected, hostResult32[i])
		}
	}
}

// TestRunner_ManualMemoryOperations tests manual memory copy operations
func TestRunner_ManualMemoryOperations(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer kp.Free()

	// Host data
	data := make([]float64, 10)
	for i := range data {
		data[i] = float64(i)
	}

	// Phase 1: Define bindings
	err := kp.DefineBindings(
		builder.InOut("data").Bind(data),
	)
	if err != nil {
		t.Fatalf("Failed to define bindings: %v", err)
	}

	// Phase 1: Allocate device memory
	err = kp.AllocateDevice()
	if err != nil {
		t.Fatalf("Failed to allocate device: %v", err)
	}

	t.Run("ManualCopyToDevice", func(t *testing.T) {
		// Manually copy to device
		err := kp.CopyToDevice("data")
		if err != nil {
			t.Fatalf("CopyToDevice failed: %v", err)
		}

		// Run kernel without any copy actions
		_, err = kp.ConfigureKernel("process",
			kp.Param("data"), // No copy actions
		)
		if err != nil {
			t.Fatalf("Failed to configure kernel: %v", err)
		}

		signature, _ := kp.GetKernelSignatureForConfig("process")
		kernelSource := fmt.Sprintf(`
@kernel void process(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		double* data = data_PART(part);
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				data[i] *= 3.0;
			}
		}
	}
}`, signature)

		kp.BuildKernel(kernelSource, "process")
		kp.ExecuteKernel("process")

		// Manually copy back
		err = kp.CopyFromDevice("data")
		if err != nil {
			t.Fatalf("CopyFromDevice failed: %v", err)
		}

		// Verify
		for i := range data {
			expected := float64(i) * 3.0
			if math.Abs(data[i]-expected) > 1e-10 {
				t.Errorf("Element %d: expected %f, got %f", i, expected, data[i])
			}
		}
	})
}

// TestRunner_ConfigureCopy tests batch copy configuration
func TestRunner_ConfigureCopy(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer kp.Free()

	// Host data
	a := make([]float64, 10)
	b := make([]float64, 10)
	c := make([]float64, 10)

	for i := range a {
		a[i] = float64(i)
		b[i] = float64(i * 2)
	}

	// Phase 1: Define bindings
	err := kp.DefineBindings(
		builder.Input("a").Bind(a),
		builder.Input("b").Bind(b),
		builder.Output("c").Bind(c),
	)
	if err != nil {
		t.Fatalf("Failed to define bindings: %v", err)
	}

	// Phase 1: Allocate device memory
	err = kp.AllocateDevice()
	if err != nil {
		t.Fatalf("Failed to allocate device: %v", err)
	}

	// Configure batch copy to device
	copyToConfig, err := kp.ConfigureCopy(
		kp.Param("a").CopyTo(),
		kp.Param("b").CopyTo(),
	)
	if err != nil {
		t.Fatalf("Failed to configure copy to device: %v", err)
	}

	// Execute batch copy
	err = kp.ExecuteCopy(copyToConfig)
	if err != nil {
		t.Fatalf("Failed to execute copy to device: %v", err)
	}

	// Run kernel without copy actions
	_, err = kp.ConfigureKernel("add",
		kp.Param("a"),
		kp.Param("b"),
		kp.Param("c"),
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignatureForConfig("add")
	kernelSource := fmt.Sprintf(`
@kernel void add(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* a = a_PART(part);
		const double* b = b_PART(part);
		double* c = c_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				c[i] = a[i] + b[i];
			}
		}
	}
}`, signature)

	kp.BuildKernel(kernelSource, "add")
	kp.ExecuteKernel("add")

	// Configure batch copy from device
	copyFromConfig, err := kp.ConfigureCopy(
		kp.Param("c").CopyBack(),
	)
	if err != nil {
		t.Fatalf("Failed to configure copy from device: %v", err)
	}

	// Execute batch copy
	err = kp.ExecuteCopy(copyFromConfig)
	if err != nil {
		t.Fatalf("Failed to execute copy from device: %v", err)
	}

	// Verify results
	for i := range c {
		expected := a[i] + b[i]
		if math.Abs(c[i]-expected) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, c[i])
		}
	}
}

// TestRunner_DeviceDataPersistence tests that device data persists between kernels
func TestRunner_DeviceDataPersistence(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer kp.Free()

	// Host data
	data := make([]float64, 10)
	intermediate := make([]float64, 10)
	result := make([]float64, 10)

	for i := range data {
		data[i] = float64(i)
	}

	// Phase 1: Define bindings
	err := kp.DefineBindings(
		builder.InOut("data").Bind(data),
		builder.InOut("intermediate").Bind(intermediate),
		builder.Output("result").Bind(result),
	)
	if err != nil {
		t.Fatalf("Failed to define bindings: %v", err)
	}

	// Phase 1: Allocate device memory
	err = kp.AllocateDevice()
	if err != nil {
		t.Fatalf("Failed to allocate device: %v", err)
	}

	// Kernel 1: Square the data and store in intermediate
	_, err = kp.ConfigureKernel("square",
		kp.Param("data").CopyTo(),
		kp.Param("intermediate"), // No copy - stays on device
	)
	if err != nil {
		t.Fatalf("Failed to configure square kernel: %v", err)
	}

	sig1, _ := kp.GetKernelSignatureForConfig("square")
	kernel1Src := fmt.Sprintf(`
@kernel void square(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* data = data_PART(part);
		double* intermediate = intermediate_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				intermediate[i] = data[i] * data[i];
			}
		}
	}
}`, sig1)

	kp.BuildKernel(kernel1Src, "square")
	kp.ExecuteKernel("square")

	// Kernel 2: Add original data to squared data
	_, err = kp.ConfigureKernel("combine",
		kp.Param("data"),         // Still on device from kernel 1
		kp.Param("intermediate"), // Still on device from kernel 1
		kp.Param("result").CopyBack(),
	)
	if err != nil {
		t.Fatalf("Failed to configure combine kernel: %v", err)
	}

	sig2, _ := kp.GetKernelSignatureForConfig("combine")
	kernel2Src := fmt.Sprintf(`
@kernel void combine(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* data = data_PART(part);
		const double* intermediate = intermediate_PART(part);
		double* result = result_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				result[i] = data[i] + intermediate[i]; // i + i^2
			}
		}
	}
}`, sig2)

	kp.BuildKernel(kernel2Src, "combine")
	kp.ExecuteKernel("combine")

	// Verify results: result[i] = i + i^2
	for i := range result {
		expected := float64(i) + float64(i*i)
		if math.Abs(result[i]-expected) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, result[i])
		}
	}
}

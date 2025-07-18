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
		K:       []int{100},
		IntType: builder.INT64,
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
	_, err = kp.ConfigureKernel("setValues",
		kp.Param("data").Copy(), // Copy in and out
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	// Get signature for kernel
	signature, err := kp.GetKernelSignatureForConfig("setValues")
	if err != nil {
		t.Fatalf("Failed to get signature: %v", err)
	}

	// Simple kernel that squares values
	kernelSource := fmt.Sprintf(`
@kernel void setValues(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		double* data = data_PART(part);
		
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
	err = kp.ExecuteKernel("setValues")
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

func TestRunner_MultiplePartitions_Core(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	k := []int{5, 10, 8}
	totalElements := 23

	kp := NewRunner(device, builder.Config{
		K: k,
	})
	defer kp.Free()

	// Host arrays - flat arrays
	hostA := make([]float64, totalElements)
	hostB := make([]float64, totalElements)
	hostC := make([]float64, totalElements)

	// Initialize
	for i := 0; i < totalElements; i++ {
		hostA[i] = float64(i)
		hostB[i] = float64(i * 2)
	}

	// Phase 1: Define bindings
	err := kp.DefineBindings(
		builder.Input("A").Bind(hostA),
		builder.Input("B").Bind(hostB),
		builder.Output("C").Bind(hostC),
		builder.Scalar("alpha").Bind(1.5),
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
	_, err = kp.ConfigureKernel("vectorAdd",
		kp.Param("A").CopyTo(),
		kp.Param("B").CopyTo(),
		kp.Param("C").CopyBack(),
		kp.Param("alpha"),
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignatureForConfig("vectorAdd")

	kernelSource := fmt.Sprintf(`
@kernel void vectorAdd(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* A = A_PART(part);
		const double* B = B_PART(part);
		double* C = C_PART(part);
		
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

	err = kp.ExecuteKernel("vectorAdd")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify results
	for i := 0; i < totalElements; i++ {
		expected := 1.5*float64(i) + float64(i*2)
		if math.Abs(hostC[i]-expected) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, hostC[i])
		}
	}
}

// ============================================================================
// Section 3: Edge Cases and Error Handling
// ============================================================================

func TestRunner_NonexistentParameter(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer kp.Free()

	// Define minimal bindings
	err := kp.DefineBindings(
		builder.Input("data").Bind(make([]float64, 10)),
	)
	if err != nil {
		t.Fatalf("Failed to define bindings: %v", err)
	}

	err = kp.AllocateDevice()
	if err != nil {
		t.Fatalf("Failed to allocate device: %v", err)
	}

	// Try to configure kernel with nonexistent parameter
	_, err = kp.ConfigureKernel("test",
		kp.Param("data").CopyTo(),
		kp.Param("nonexistent").CopyTo(), // This should fail
	)

	if err == nil {
		t.Error("Expected error for nonexistent parameter")
	}
}

func TestRunner_BeforeAllocation(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer kp.Free()

	// Define bindings but don't allocate
	err := kp.DefineBindings(
		builder.Input("data").Bind(make([]float64, 10)),
	)
	if err != nil {
		t.Fatalf("Failed to define bindings: %v", err)
	}

	// Try to configure kernel before allocation - should fail
	_, err = kp.ConfigureKernel("test",
		kp.Param("data").CopyTo(),
	)

	if err == nil {
		t.Error("Expected error when configuring kernel before allocation")
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

// ============================================================================
// Section 5: Scalar Parameter Tests
// ============================================================================

func TestRunner_ScalarParameters(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer kp.Free()

	// Various scalar types
	floatVal := 3.14
	intVal := int64(42)
	complexVal := complex(2.0, 3.0)

	result := make([]float64, 10)

	// Phase 1: Define bindings
	err := kp.DefineBindings(
		builder.Output("result").Bind(result),
		builder.Scalar("pi").Bind(floatVal),
		builder.Scalar("answer").Bind(intVal),
		builder.Scalar("comp").Bind(complexVal),
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
	_, err = kp.ConfigureKernel("scalars",
		kp.Param("result").CopyBack(),
		kp.Param("pi"),
		kp.Param("answer"),
		kp.Param("comp"),
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignatureForConfig("scalars")

	kernelSource := fmt.Sprintf(`
@kernel void scalars(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		double* result = result_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				result[i] = pi + (double)answer + comp.x + comp.y;
			}
		}
	}
}`, signature)

	_, err = kp.BuildKernel(kernelSource, "scalars")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.ExecuteKernel("scalars")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify
	expected := floatVal + float64(intVal) + real(complexVal) + imag(complexVal)
	for i := 0; i < 10; i++ {
		if math.Abs(result[i]-expected) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, result[i])
		}
	}
}

// ============================================================================
// Section 6: Temporary Array Tests
// ============================================================================

func TestRunner_TemporaryArrays(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K: []int{20},
	})
	defer kp.Free()

	input := make([]float64, 20)
	output := make([]float64, 20)

	for i := range input {
		input[i] = float64(i)
	}

	// Phase 1: Define bindings with temp arrays
	err := kp.DefineBindings(
		builder.Input("input").Bind(input),
		builder.Output("output").Bind(output),
		builder.Temp("scratch").Type(builder.Float64).Size(20),
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
	_, err = kp.ConfigureKernel("useScratch",
		kp.Param("input").CopyTo(),
		kp.Param("output").CopyBack(),
		kp.Param("scratch"),
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignatureForConfig("useScratch")

	kernelSource := fmt.Sprintf(`
@kernel void useScratch(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* input = input_PART(part);
		double* output = output_PART(part);
		double* scratch = scratch_PART(part);
		
		// First pass: square into scratch
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				scratch[i] = input[i] * input[i];
			}
		}
		
		// Second pass: add original
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				output[i] = scratch[i] + input[i];
			}
		}
	}
}`, signature)

	_, err = kp.BuildKernel(kernelSource, "useScratch")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.ExecuteKernel("useScratch")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify: output = input^2 + input
	for i := 0; i < 20; i++ {
		expected := float64(i*i) + float64(i)
		if math.Abs(output[i]-expected) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, output[i])
		}
	}
}

// ============================================================================
// Section 7: Multi-Kernel Workflow Tests
// ============================================================================

func TestRunner_MultiKernelWorkflow(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K: []int{15},
	})
	defer kp.Free()

	// Working arrays
	u := make([]float64, 15)
	v := make([]float64, 15)
	result := make([]float64, 15)

	// Initialize
	for i := range u {
		u[i] = float64(i)
		v[i] = float64(i) * 0.5
	}

	// Phase 1: Define all bindings once
	err := kp.DefineBindings(
		builder.InOut("u").Bind(u),
		builder.InOut("v").Bind(v),
		builder.Output("result").Bind(result),
		builder.Scalar("dt").Bind(0.1),
	)
	if err != nil {
		t.Fatalf("Failed to define bindings: %v", err)
	}

	// Phase 1: Allocate device memory once
	err = kp.AllocateDevice()
	if err != nil {
		t.Fatalf("Failed to allocate device: %v", err)
	}

	// Kernel 1: Update u based on v
	_, err = kp.ConfigureKernel("updateU",
		kp.Param("u").Copy(),   // Copy to device and back
		kp.Param("v").CopyTo(), // Only copy to device
		kp.Param("dt"),
	)
	if err != nil {
		t.Fatalf("Failed to configure updateU: %v", err)
	}

	sig1, _ := kp.GetKernelSignatureForConfig("updateU")
	kernel1Src := fmt.Sprintf(`
@kernel void updateU(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		double* u = u_PART(part);
		const double* v = v_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				u[i] = u[i] + dt * v[i];
			}
		}
	}
}`, sig1)

	_, err = kp.BuildKernel(kernel1Src, "updateU")
	if err != nil {
		t.Fatalf("Failed to build updateU: %v", err)
	}

	err = kp.ExecuteKernel("updateU")
	if err != nil {
		t.Fatalf("Failed to execute updateU: %v", err)
	}

	// Verify first kernel results
	for i := 0; i < 15; i++ {
		expected := float64(i) + 0.1*float64(i)*0.5
		if math.Abs(u[i]-expected) > 1e-10 {
			t.Errorf("After updateU, u[%d]: expected %f, got %f", i, expected, u[i])
		}
	}

	// Kernel 2: Compute result from u and v (which are still on device)
	_, err = kp.ConfigureKernel("computeResult",
		kp.Param("u"), // Already on device
		kp.Param("v"), // Already on device
		kp.Param("result").CopyBack(),
	)
	if err != nil {
		t.Fatalf("Failed to configure computeResult: %v", err)
	}

	sig2, _ := kp.GetKernelSignatureForConfig("computeResult")
	kernel2Src := fmt.Sprintf(`
@kernel void computeResult(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* u = u_PART(part);
		const double* v = v_PART(part);
		double* result = result_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				result[i] = u[i] * u[i] + v[i] * v[i];
			}
		}
	}
}`, sig2)

	_, err = kp.BuildKernel(kernel2Src, "computeResult")
	if err != nil {
		t.Fatalf("Failed to build computeResult: %v", err)
	}

	err = kp.ExecuteKernel("computeResult")
	if err != nil {
		t.Fatalf("Failed to execute computeResult: %v", err)
	}

	// Verify final results
	for i := 0; i < 15; i++ {
		u_val := float64(i) + 0.1*float64(i)*0.5 // Updated u from kernel 1
		v_val := float64(i) * 0.5                // Original v
		expected := u_val*u_val + v_val*v_val
		if math.Abs(result[i]-expected) > 1e-10 {
			t.Errorf("Final result[%d]: expected %f, got %f", i, expected, result[i])
		}
	}
}

package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"gonum.org/v1/gonum/mat"
	"math"
	"strings"
	"testing"
	"unsafe"
)

// ============================================================================
// Section 1: Basic Creation and Configuration Tests
// Following Unit Testing Principle: Start with fundamentals
// ============================================================================

// Test 1.1: Device validation
func TestDGKernel(t *testing.T) {
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

// Test 1.2: Single partition creation
func TestDGKernel_Creation_SinglePartition(t *testing.T) {
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

// Test 1.3: KpartMax computation with multiple partitions
func TestDGKernel_Creation_KpartMaxComputation(t *testing.T) {
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
// Section 2: Code Generation Tests
// Following Unit Testing Principle: Build systematically
// ============================================================================

// Test 2.1: Type definitions and constants generation
func TestDGKernel_CodeGen_TypesAndConstants(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K:       []int{5, 10, 7},
		IntType: builder.INT64,
	})
	defer kp.Free()

	preamble := kp.GeneratePreamble(kp.GetAllocatedArrays(),
		kp.collectArrayTypes())

	// Check type definitions
	expectedTypes := []string{
		"typedef long int_t",
	}

	for _, expected := range expectedTypes {
		if !strings.Contains(preamble, expected) {
			t.Errorf("Missing type definition: %s", expected)
		}
	}

	// Check constants
	if !strings.Contains(preamble, "#define NPART 3") {
		t.Error("Missing or incorrect NPART definition")
	}
	if !strings.Contains(preamble, "#define KpartMax 10") {
		t.Error("Missing or incorrect KpartMax definition")
	}
}

// Test 2.2: Matrix macro generation with @inner loop
func TestDGKernel_CodeGen_MatrixMacroStructure(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{K: []int{10, 20}})
	defer kp.Free()

	// Add a differentiation matrix
	Dr := mat.NewDense(3, 3, []float64{
		-1.0, 1.0, 0.0,
		-0.5, 0.0, 0.5,
		0.0, -1.0, 1.0,
	})
	kp.AddStaticMatrix("Dr", Dr)

	preamble := kp.GeneratePreamble(kp.GetAllocatedArrays(),
		kp.collectArrayTypes())

	// Verify matrix declaration
	if !strings.Contains(preamble, "const double Dr[3][3]") {
		t.Error("Missing Dr matrix declaration")
	}

	// Verify macro contains @inner loop and new signature
	requiredPatterns := []string{
		"#define MATMUL_Dr(IN, OUT, K_VAL)",
		"#define MATMUL_ADD_Dr(IN, OUT, K_VAL)",
		"for (int elem = 0; elem < KpartMax; ++elem; @inner)",
	}

	for _, pattern := range requiredPatterns {
		if !strings.Contains(preamble, pattern) {
			t.Errorf("Missing required pattern in macro: %s", pattern)
		}
	}
}

// ============================================================================
// Section 3: Kernel Execution Tests with New API
// Note: When using multiple partitions (len(K) > 1), use partitioned data:
//   - Non-partitioned: []float64 for single partition
//   - Partitioned: [][]float64 for multiple partitions
// ============================================================================

// Test 3.1: Basic kernel with new API (single partition)
func TestDGKernel_NewAPI_BasicKernel(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	// Single partition - use non-partitioned data
	kp := NewRunner(device, builder.Config{K: []int{10}})
	defer kp.Free()

	// Host data - non-partitioned for single partition
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
		kp.Param("data").Copy(),
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	// Get signature
	signature, _ := kp.GetKernelSignatureForConfig("setValues")

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

	// Verify results - data should be squared
	for i := 0; i < 10; i++ {
		expected := float64(i * i)
		if math.Abs(hostData[i]-expected) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, hostData[i])
		}
	}
}

// Test 3.2: Matrix operations with new API
func TestDGKernel_NewAPI_MatrixOperation(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	np := 4
	k := []int{5, 10}
	totalNodes := 15 * np

	kp := NewRunner(device, builder.Config{
		K: k,
	})
	defer kp.Free()

	// Add differentiation matrix
	Dr := mat.NewDense(np, np, []float64{
		-2.0, 3.0, -1.0, 0.0,
		-1.0, 0.0, 1.0, 0.0,
		0.0, -1.0, 0.0, 1.0,
		0.0, 1.0, -3.0, 2.0,
	})

	// Initialize test data as partitioned arrays (since we have multiple partitions)
	U := make([][]float64, len(k))
	Ur := make([][]float64, len(k))
	idx := 0
	for part := 0; part < len(k); part++ {
		U[part] = make([]float64, k[part]*np)
		Ur[part] = make([]float64, k[part]*np)
		for i := range U[part] {
			U[part][i] = float64(idx % 10)
			idx++
		}
	}

	// Phase 1: Define bindings
	err := kp.DefineBindings(
		builder.Input("Dr").Bind(Dr).ToMatrix().Static(),
		builder.Input("U").Bind(U),
		builder.Output("Ur").Bind(Ur),
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
	_, err = kp.ConfigureKernel("differentiate",
		kp.Param("Dr"),
		kp.Param("U").CopyTo(),
		kp.Param("Ur").CopyBack(),
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignatureForConfig("differentiate")

	// Kernel using differentiation matrix
	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void differentiate(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* U = U_PART(part);
		double* Ur = Ur_PART(part);
		MATMUL_Dr(U, Ur, K[part]);
	}
}
`, np, signature)

	_, err = kp.BuildKernel(kernelSource, "differentiate")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Execute differentiation
	err = kp.ExecuteKernel("differentiate")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify results make sense (not checking exact values, just sanity)
	result, err := CopyArrayToHost[float64](kp, "Ur")
	if err != nil {
		t.Fatalf("Failed to copy result: %v", err)
	}

	// Check that we got non-zero results
	hasNonZero := false
	for i := 0; i < totalNodes; i++ {
		if math.Abs(result[i]) > 1e-10 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("Differentiation produced all zeros")
	}
}

// Test 3.3: Multiple arrays with new API
func TestDGKernel_NewAPI_MultipleArrays(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	// Multiple partitions
	k := []int{5, 8, 6}
	kp := NewRunner(device, builder.Config{K: k})
	defer kp.Free()

	// Calculate total elements
	totalElements := 0
	for _, kval := range k {
		totalElements += kval
	}

	// Host arrays - using flat arrays for simplicity
	hostU := splitSlice(k, make([]float64, totalElements))
	hostV := splitSlice(k, make([]float64, totalElements))
	hostRHS := splitSlice(k, make([]float64, totalElements))

	// Initialize
	for ii, K := range k {
		for kk := 0; kk < K; kk++ {
			hostU[ii][kk] = float64(kk)
			hostV[ii][kk] = float64(kk) * 2.0
		}
	}

	// Phase 1: Define bindings
	err := kp.DefineBindings(
		builder.Input("U").Bind(hostU),
		builder.Input("V").Bind(hostV),
		builder.Output("RHS").Bind(hostRHS),
		builder.Scalar("alpha").Bind(1.5),
		builder.Scalar("beta").Bind(2.5),
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
	_, err = kp.ConfigureKernel("compute",
		kp.Param("U").CopyTo(),
		kp.Param("V").CopyTo(),
		kp.Param("RHS").CopyBack(),
		kp.Param("alpha"),
		kp.Param("beta"),
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignatureForConfig("compute")

	// Kernel that computes RHS = alpha*U + beta*V
	kernelSource := fmt.Sprintf(`
@kernel void compute(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* U = U_PART(part);
		const double* V = V_PART(part);
		double* RHS = RHS_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				RHS[i] = alpha * U[i] + beta * V[i];
			}
		}
	}
}`, signature)

	_, err = kp.BuildKernel(kernelSource, "compute")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Execute kernel
	err = kp.ExecuteKernel("compute")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify results
	for ii, K := range k {
		for j := 0; j < 1; j++ {
			for kk := 0; kk < K; kk++ {
				expected := 1.5*float64(kk) + 2.5*float64(kk)*2.0
				if math.Abs(hostRHS[ii][kk]-expected) > 1e-10 {
					t.Errorf("Element %d: expected %f, got %f", kk, expected,
						hostRHS[ii][kk])
				}
			}
		}
	}
}

// ============================================================================
// Section 4: Mathematical Properties Verification
// ============================================================================

// Test 4.1: Offset calculations with new API
func TestDGKernel_MathProperties_OffsetCalculations(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	k := []int{10, 15, 20}

	kp := NewRunner(device, builder.Config{K: k})
	defer kp.Free()

	// Create host data as partitioned arrays
	hostData := make([][]float64, len(k))
	for part := 0; part < len(k); part++ {
		hostData[part] = make([]float64, k[part])
	}

	// Phase 1: Define bindings to trigger array allocation
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

	// Check that offsets were created correctly
	offsetsMem := kp.GetOffsets("data")
	if offsetsMem == nil {
		t.Fatal("Offsets not allocated")
	}

	// Read offsets back
	offsets := make([]int64, kp.NumPartitions+1)
	offsetsMem.CopyTo(unsafe.Pointer(&offsets[0]), int64(len(offsets)*8))

	// Verify offset properties
	if offsets[0] != 0 {
		t.Errorf("First offset should be 0, got %d", offsets[0])
	}

	// Offsets should be monotonically increasing
	for i := 1; i < len(offsets); i++ {
		if offsets[i] <= offsets[i-1] {
			t.Errorf("Offsets not monotonic at %d: %d <= %d", i, offsets[i], offsets[i-1])
		}
	}

	// Check partition sizes match K values
	for i := 0; i < kp.NumPartitions; i++ {
		partSize := offsets[i+1] - offsets[i]
		if partSize != int64(k[i]) {
			t.Errorf("Partition %d size mismatch: expected %d, got %d", i, k[i], partSize)
		}
	}
}

// ============================================================================
// Section 5: State Persistence Tests
// ============================================================================

// Test 5.1: Device memory persistence across kernels
func TestDGKernel_StatePersistence_AcrossKernels(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{K: []int{10}})
	defer kp.Free()

	// Host data
	data := make([]float64, 10)
	temp := make([]float64, 10)
	result := make([]float64, 10)

	for i := range data {
		data[i] = float64(i)
	}

	// Phase 1: Define all bindings once
	err := kp.DefineBindings(
		builder.InOut("data").Bind(data),
		builder.InOut("temp").Bind(temp),
		builder.Output("result").Bind(result),
	)
	if err != nil {
		t.Fatalf("Failed to define bindings: %v", err)
	}

	// Phase 1: Allocate device memory once
	err = kp.AllocateDevice()
	if err != nil {
		t.Fatalf("Failed to allocate device: %v", err)
	}

	// Kernel 1: Square data and store in temp
	_, err = kp.ConfigureKernel("square",
		kp.Param("data").CopyTo(),
		kp.Param("temp"), // No copy - stays on device
	)
	if err != nil {
		t.Fatalf("Failed to configure square kernel: %v", err)
	}

	sig1, _ := kp.GetKernelSignatureForConfig("square")
	kernel1Src := fmt.Sprintf(`
@kernel void square(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* data = data_PART(part);
		double* temp = temp_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				temp[i] = data[i] * data[i];
			}
		}
	}
}`, sig1)

	_, err = kp.BuildKernel(kernel1Src, "square")
	if err != nil {
		t.Fatalf("Failed to build square kernel: %v", err)
	}

	err = kp.ExecuteKernel("square")
	if err != nil {
		t.Fatalf("Failed to execute square kernel: %v", err)
	}

	// Kernel 2: Add original data to squared data
	_, err = kp.ConfigureKernel("combine",
		kp.Param("data"), // Still on device from kernel 1
		kp.Param("temp"), // Still on device from kernel 1
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
		const double* temp = temp_PART(part);
		double* result = result_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				result[i] = data[i] + temp[i]; // i + i^2
			}
		}
	}
}`, sig2)

	_, err = kp.BuildKernel(kernel2Src, "combine")
	if err != nil {
		t.Fatalf("Failed to build combine kernel: %v", err)
	}

	err = kp.ExecuteKernel("combine")
	if err != nil {
		t.Fatalf("Failed to execute combine kernel: %v", err)
	}

	// Verify results: result[i] = i + i^2
	for i := range result {
		expected := float64(i) + float64(i*i)
		if math.Abs(result[i]-expected) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, result[i])
		}
	}
}

// ============================================================================
// Section 6: Advanced Features Tests
// ============================================================================

// Test 6.1: Type conversion with new API
func TestDGKernel_TypeConversion(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{K: []int{10}})
	defer kp.Free()

	// Host data in float64
	hostData64 := make([]float64, 10)
	for i := range hostData64 {
		hostData64[i] = float64(i) * 1.123456789
	}

	// Phase 1: Define bindings with conversion
	err := kp.DefineBindings(
		builder.InOut("data").Bind(hostData64).Convert(builder.Float32),
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
	_, err = kp.ConfigureKernel("process",
		kp.Param("data").Copy(),
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignatureForConfig("process")

	// Kernel works with float32 on device
	kernelSource := fmt.Sprintf(`
@kernel void process(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		float* data = data_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				data[i] = data[i] * 2.0f;
			}
		}
	}
}`, signature)

	_, err = kp.BuildKernel(kernelSource, "process")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.ExecuteKernel("process")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify with precision loss from float64->float32->float64
	for i := range hostData64 {
		original := float64(i) * 1.123456789
		expectedWithLoss := float64(float32(original) * 2.0)
		if math.Abs(hostData64[i]-expectedWithLoss) > 1e-6 {
			t.Errorf("Element %d: expected %f (with precision loss), got %f",
				i, expectedWithLoss, hostData64[i])
		}
	}
}

// Test 6.2: Manual memory operations
func TestDGKernel_ManualMemoryOps(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{K: []int{10}})
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

	// Manual copy to device
	err = kp.CopyToDevice("data")
	if err != nil {
		t.Fatalf("CopyToDevice failed: %v", err)
	}

	// Configure kernel without copy actions
	_, err = kp.ConfigureKernel("double",
		kp.Param("data"), // No copy actions
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	sig, _ := kp.GetKernelSignatureForConfig("double")
	kernelSrc := fmt.Sprintf(`
@kernel void double(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		double* data = data_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				data[i] = data[i] * 2.0;
			}
		}
	}
}`, sig)

	_, err = kp.BuildKernel(kernelSrc, "double")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.ExecuteKernel("double")
	if err != nil {
		t.Fatalf("Failed to execute kernel: %v", err)
	}

	// Manual copy from device
	err = kp.CopyFromDevice("data")
	if err != nil {
		t.Fatalf("CopyFromDevice failed: %v", err)
	}

	// Verify
	for i := range data {
		expected := float64(i) * 2.0
		if math.Abs(data[i]-expected) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, data[i])
		}
	}
}

// Test 6.3: Batch copy operations
func TestDGKernel_BatchCopyOps(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{K: []int{10}})
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

	// Configure batch copy
	copyOp, err := kp.ConfigureCopy(
		kp.Param("a").CopyTo(),
		kp.Param("b").CopyTo(),
	)
	if err != nil {
		t.Fatalf("Failed to configure copy: %v", err)
	}

	// Execute batch copy
	err = kp.ExecuteCopy(copyOp)
	if err != nil {
		t.Fatalf("Failed to execute copy: %v", err)
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

	sig, _ := kp.GetKernelSignatureForConfig("add")
	kernelSrc := fmt.Sprintf(`
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
}`, sig)

	_, err = kp.BuildKernel(kernelSrc, "add")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.ExecuteKernel("add")
	if err != nil {
		t.Fatalf("Failed to execute kernel: %v", err)
	}

	// Manual copy back
	err = kp.CopyFromDevice("c")
	if err != nil {
		t.Fatalf("Failed to copy from device: %v", err)
	}

	// Verify
	for i := range c {
		expected := a[i] + b[i]
		if math.Abs(c[i]-expected) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, c[i])
		}
	}
}

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
		K:         []int{5, 10, 7},
		FloatType: builder.Float64,
		IntType:   builder.INT64,
	})
	defer kp.Free()

	preamble := kp.GeneratePreamble()

	// Check type definitions
	expectedTypes := []string{
		"typedef double real_t",
		"typedef long int_t",
		"#define REAL_ZERO 0.0",
		"#define REAL_ONE 1.0",
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

	preamble := kp.GeneratePreamble()

	// Verify matrix declaration
	if !strings.Contains(preamble, "const double Dr[3][3]") {
		t.Error("Missing Dr matrix declaration")
	}

	// Verify macro contains @inner loop and new signature
	requiredPatterns := []string{
		"#define MATMUL_Dr(IN, OUT, K_VAL)",
		"#define MATMUL_ADD_Dr(IN, OUT, K_VAL)",
		"for (int elem = 0; elem < KpartMax; ++elem; @inner)",
		"if (elem < (K_VAL))",
	}

	for _, pattern := range requiredPatterns {
		if !strings.Contains(preamble, pattern) {
			t.Errorf("Missing required pattern in macro: %s", pattern)
		}
	}
}

// ============================================================================
// Section 3: Kernel Execution Tests with New API
// ============================================================================

// Test 3.1: Basic kernel with new API
func TestDGKernel_NewAPI_BasicKernel(t *testing.T) {
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
		builder.InOut("data").Bind(hostData).Copy(),
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	// Get signature
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
		K:         k,
		FloatType: builder.Float64,
	})
	defer kp.Free()

	// Add differentiation matrix
	Dr := mat.NewDense(np, np, []float64{
		-2.0, 3.0, -1.0, 0.0,
		-1.0, 0.0, 1.0, 0.0,
		0.0, -1.0, 0.0, 1.0,
		0.0, 1.0, -3.0, 2.0,
	})

	// Initialize test data
	U := make([]float64, totalNodes)
	Ur := make([]float64, totalNodes)
	for i := range U {
		U[i] = float64(i % 10)
	}

	// Define kernel with new API
	err := kp.DefineKernel("differentiate",
		builder.Input("Dr").Bind(Dr).ToMatrix().Static(),
		builder.Input("U").Bind(U).CopyTo(),
		builder.Output("Ur").Bind(Ur),
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignature("differentiate")

	// Kernel using differentiation matrix
	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void differentiate(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* U = U_PART(part);
		real_t* Ur = Ur_PART(part);
		MATMUL_Dr(U, Ur, K[part]);
	}
}
`, np, signature)

	_, err = kp.BuildKernel(kernelSource, "differentiate")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Execute differentiation
	err = kp.RunKernel("differentiate")
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

	k := []int{10, 15, 20}
	totalElements := 45

	kp := NewRunner(device, builder.Config{K: k})
	defer kp.Free()

	// Initialize host arrays
	hostU := make([]float64, totalElements)
	hostV := make([]float64, totalElements)
	hostW := make([]float64, totalElements)

	for i := range hostU {
		hostU[i] = float64(i)
		hostV[i] = float64(i * 2)
	}

	// Define kernel with new API
	err := kp.DefineKernel("vectorOps",
		builder.Input("U").Bind(hostU).CopyTo(),
		builder.Input("V").Bind(hostV).CopyTo(),
		builder.Output("W").Bind(hostW),
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignature("vectorOps")

	// Kernel that adds vectors
	kernelSource := fmt.Sprintf(`
@kernel void vectorOps(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* U = U_PART(part);
		const real_t* V = V_PART(part);
		real_t* W = W_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				W[i] = U[i] + V[i];
			}
		}
	}
}`, signature)

	_, err = kp.BuildKernel(kernelSource, "vectorOps")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Execute kernel
	err = kp.RunKernel("vectorOps")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify results
	result, err := CopyArrayToHost[float64](kp, "W")
	if err != nil {
		t.Fatalf("Failed to copy result: %v", err)
	}

	for i := 0; i < totalElements; i++ {
		expected := hostU[i] + hostV[i]
		if math.Abs(result[i]-expected) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, result[i])
		}
	}
}

// ============================================================================
// Section 4: Edge Cases and Degeneracies
// ============================================================================

// Test 4.1: Degenerate partition configurations
func TestDGKernel_EdgeCases_DegeneratePartitions(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	testCases := []struct {
		name         string
		k            []int
		expectedKMax int
	}{
		{"all_same", []int{10, 10, 10, 10}, 10},
		{"one_large", []int{1, 1, 100, 1, 1}, 100},
		{"powers_of_two", []int{1, 2, 4, 8, 16, 32}, 32},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			kp := NewRunner(device, builder.Config{K: tc.k})
			defer kp.Free()

			if kp.KpartMax != tc.expectedKMax {
				t.Errorf("Expected KpartMax=%d, got %d", tc.expectedKMax, kp.KpartMax)
			}

			// Verify preamble contains correct value
			preamble := kp.GeneratePreamble()
			expected := fmt.Sprintf("#define KpartMax %d", tc.expectedKMax)
			if !strings.Contains(preamble, expected) {
				t.Errorf("Preamble missing: %s", expected)
			}
		})
	}
}

// ============================================================================
// Section 5: Mathematical Properties Verification
// ============================================================================

// Test 5.1: Offset calculations with new API
func TestDGKernel_MathProperties_OffsetCalculations(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	k := []int{10, 15, 20}
	totalElements := 45

	kp := NewRunner(device, builder.Config{K: k})
	defer kp.Free()

	// Create host data
	hostData := make([]float64, totalElements)

	// Define kernel to trigger array allocation
	err := kp.DefineKernel("test",
		builder.Input("data").Bind(hostData).CopyTo(),
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
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

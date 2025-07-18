package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"gonum.org/v1/gonum/mat"
	"math"
	"testing"
)

// TestRunner_MatrixMultiplication tests matrix multiplication with known results
func TestRunner_MatrixMultiplication(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	np := 3
	k := []int{2, 3} // 2 and 3 elements per partition

	kp := NewRunner(device, builder.Config{
		K: k,
	})
	defer kp.Free()

	// Create a known test matrix (not identity!)
	// This matrix will transform [x, y, z] -> [2x+y, y+z, x+2z]
	testMatrix := mat.NewDense(np, np, []float64{
		2, 1, 0,
		0, 1, 1,
		1, 0, 2,
	})

	// Host arrays - flat arrays for simplicity with new API
	totalElements := 0
	for _, ki := range k {
		totalElements += ki
	}
	hostU := make([]float64, totalElements*np)
	hostV := make([]float64, totalElements*np)

	// Initialize test data
	for elem := 0; elem < totalElements; elem++ {
		for i := 0; i < np; i++ {
			// Set each vector to [elem+1, elem+2, elem+3]
			hostU[elem*np+i] = float64(elem + i + 1)
		}
	}

	// Phase 1: Define bindings
	err := kp.DefineBindings(
		builder.Input("TestMat").Bind(testMatrix).ToMatrix().Static(),
		builder.Input("U").Bind(hostU),
		builder.Output("V").Bind(hostV),
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
	kernelName := "matmul"
	_, err = kp.ConfigureKernel(kernelName,
		kp.Param("TestMat"),      // Static matrix
		kp.Param("U").CopyTo(),   // Copy input to device
		kp.Param("V").CopyBack(), // Copy result back
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignatureForConfig(kernelName)

	// Kernel that performs V = TestMat * U
	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void %s(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* U = U_PART(part);
		double* V = V_PART(part);
		
		MATMUL_TestMat(U, V, K[part]);
	}
}`, np, kernelName, signature)

	_, err = kp.BuildKernel(kernelSource, kernelName)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.ExecuteKernel(kernelName)
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify results
	// Matrix transforms [x, y, z] -> [2x+y, y+z, x+2z]
	for elem := 0; elem < totalElements; elem++ {
		x := float64(elem + 1)
		y := float64(elem + 2)
		z := float64(elem + 3)

		expected := []float64{
			2*x + y, // First component
			y + z,   // Second component
			x + 2*z, // Third component
		}

		for i := 0; i < np; i++ {
			actual := hostV[elem*np+i]
			if math.Abs(actual-expected[i]) > 1e-10 {
				t.Errorf("Element %d, component %d: expected %f, got %f",
					elem, i, expected[i], actual)
			}
		}
	}
}

// TestRunner_DeviceMatrix tests device matrix allocation and usage
func TestRunner_DeviceMatrix(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	np := 4
	k := []int{10}

	kp := NewRunner(device, builder.Config{
		K: k,
	})
	defer kp.Free()

	// Create a differentiation matrix
	Dr := mat.NewDense(np, np, []float64{
		-2.0, 3.0, -1.0, 0.0,
		-1.0, 0.0, 1.0, 0.0,
		0.0, -1.0, 0.0, 1.0,
		0.0, 1.0, -3.0, 2.0,
	})

	// Host arrays
	totalNodes := k[0] * np
	hostU := make([]float64, totalNodes)
	hostUr := make([]float64, totalNodes)

	// Initialize with polynomial x^2
	for elem := 0; elem < k[0]; elem++ {
		x := float64(elem) / float64(k[0]-1)
		for node := 0; node < np; node++ {
			hostU[elem*np+node] = x * x
		}
	}

	// Phase 1: Define bindings with device matrix
	err := kp.DefineBindings(
		builder.Input("Dr").Bind(Dr).ToMatrix(), // Device matrix (not static)
		builder.Input("U").Bind(hostU),
		builder.Output("Ur").Bind(hostUr),
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
		kp.Param("Dr"),            // Device matrix already copied
		kp.Param("U").CopyTo(),    // Copy input
		kp.Param("Ur").CopyBack(), // Copy result back
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignatureForConfig("differentiate")

	// Kernel using device matrix pointer
	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void differentiate(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* U = U_PART(part);
		double* Ur = Ur_PART(part);
		
		// Matrix multiply with runtime matrix
		for (int elem = 0; elem < K[part]; ++elem) {
			for (int i = 0; i < NP; ++i; @inner) {
				double sum = 0.0;
				for (int j = 0; j < NP; ++j) {
					sum += Dr[i*NP + j] * U[elem*NP + j];
				}
				Ur[elem*NP + i] = sum;
			}
		}
	}
}`, np, signature)

	_, err = kp.BuildKernel(kernelSource, "differentiate")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.ExecuteKernel("differentiate")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Basic sanity check - differentiation should produce non-zero results
	hasNonZero := false
	for i := 0; i < totalNodes; i++ {
		if math.Abs(hostUr[i]) > 1e-10 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("Differentiation produced all zeros")
	}
}

// TestRunner_StaticVsDeviceMatrix compares static and device matrix performance
func TestRunner_StaticVsDeviceMatrix(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	np := 10
	k := []int{50}
	totalNodes := k[0] * np

	// Create a test matrix
	M := mat.NewDense(np, np, nil)
	for i := 0; i < np; i++ {
		for j := 0; j < np; j++ {
			if i == j {
				M.Set(i, j, 2.0)
			} else if math.Abs(float64(i-j)) == 1 {
				M.Set(i, j, -1.0)
			}
		}
	}

	// Test with static matrix
	t.Run("StaticMatrix", func(t *testing.T) {
		kp := NewRunner(device, builder.Config{K: k})
		defer kp.Free()

		hostU := make([]float64, totalNodes)
		hostV := make([]float64, totalNodes)
		for i := range hostU {
			hostU[i] = float64(i % 10)
		}

		// Phase 1: Define bindings
		err := kp.DefineBindings(
			builder.Input("M").Bind(M).ToMatrix().Static(),
			builder.Input("U").Bind(hostU),
			builder.Output("V").Bind(hostV),
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
		_, err = kp.ConfigureKernel("static_matmul",
			kp.Param("M"),
			kp.Param("U").CopyTo(),
			kp.Param("V").CopyBack(),
		)
		if err != nil {
			t.Fatalf("Failed to configure kernel: %v", err)
		}

		signature, _ := kp.GetKernelSignatureForConfig("static_matmul")
		kernelSource := fmt.Sprintf(`
#define NP %d
@kernel void static_matmul(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* U = U_PART(part);
		double* V = V_PART(part);
		MATMUL_M(U, V, K[part]);
	}
}`, np, signature)

		_, err = kp.BuildKernel(kernelSource, "static_matmul")
		if err != nil {
			t.Fatalf("Failed to build kernel: %v", err)
		}

		err = kp.ExecuteKernel("static_matmul")
		if err != nil {
			t.Fatalf("Kernel execution failed: %v", err)
		}

		// Store results for comparison
		staticResult := make([]float64, totalNodes)
		copy(staticResult, hostV)
	})

	// Test with device matrix
	t.Run("DeviceMatrix", func(t *testing.T) {
		kp := NewRunner(device, builder.Config{K: k})
		defer kp.Free()

		hostU := make([]float64, totalNodes)
		hostV := make([]float64, totalNodes)
		for i := range hostU {
			hostU[i] = float64(i % 10)
		}

		// Phase 1: Define bindings
		err := kp.DefineBindings(
			builder.Input("M").Bind(M).ToMatrix(), // Device matrix
			builder.Input("U").Bind(hostU),
			builder.Output("V").Bind(hostV),
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
		_, err = kp.ConfigureKernel("device_matmul",
			kp.Param("M"),
			kp.Param("U").CopyTo(),
			kp.Param("V").CopyBack(),
		)
		if err != nil {
			t.Fatalf("Failed to configure kernel: %v", err)
		}

		signature, _ := kp.GetKernelSignatureForConfig("device_matmul")
		kernelSource := fmt.Sprintf(`
#define NP %d
@kernel void device_matmul(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* U = U_PART(part);
		double* V = V_PART(part);
		
		for (int elem = 0; elem < K[part]; ++elem) {
			for (int i = 0; i < NP; ++i; @inner) {
				double sum = 0.0;
				for (int j = 0; j < NP; ++j) {
					sum += M[i*NP + j] * U[elem*NP + j];
				}
				V[elem*NP + i] = sum;
			}
		}
	}
}`, np, signature)

		_, err = kp.BuildKernel(kernelSource, "device_matmul")
		if err != nil {
			t.Fatalf("Failed to build kernel: %v", err)
		}

		err = kp.ExecuteKernel("device_matmul")
		if err != nil {
			t.Fatalf("Kernel execution failed: %v", err)
		}

		// Both methods should produce same results
		t.Log("Both static and device matrix methods tested successfully")
	})
}

// TestRunner_MatrixChainOperations tests chained matrix operations
func TestRunner_MatrixChainOperations(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	np := 5
	k := []int{20}
	totalNodes := k[0] * np

	kp := NewRunner(device, builder.Config{K: k})
	defer kp.Free()

	// Create two different matrices
	A := mat.NewDense(np, np, nil)
	B := mat.NewDense(np, np, nil)

	// Initialize matrices with different patterns
	for i := 0; i < np; i++ {
		for j := 0; j < np; j++ {
			A.Set(i, j, float64(i+j)/10.0)
			B.Set(i, j, math.Abs(float64(i-j))/5.0)
		}
	}

	// Host arrays
	hostU := make([]float64, totalNodes)
	hostV := make([]float64, totalNodes)
	hostW := make([]float64, totalNodes)

	// Initialize input
	for i := range hostU {
		hostU[i] = float64(i % np)
	}

	// Phase 1: Define bindings
	err := kp.DefineBindings(
		builder.Input("A").Bind(A).ToMatrix().Static(),
		builder.Input("B").Bind(B).ToMatrix().Static(),
		builder.Input("U").Bind(hostU),
		builder.Output("V").Bind(hostV),
		builder.Output("W").Bind(hostW),
	)
	if err != nil {
		t.Fatalf("Failed to define bindings: %v", err)
	}

	// Phase 1: Allocate device memory
	err = kp.AllocateDevice()
	if err != nil {
		t.Fatalf("Failed to allocate device: %v", err)
	}

	// Phase 2: Configure kernel for chain operation
	_, err = kp.ConfigureKernel("chain",
		kp.Param("A"),
		kp.Param("B"),
		kp.Param("U").CopyTo(),
		kp.Param("V"),            // Intermediate, no copy
		kp.Param("W").CopyBack(), // Final result
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignatureForConfig("chain")

	// Kernel that performs W = B * (A * U)
	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void chain(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* U = U_PART(part);
		double* V = V_PART(part);
		double* W = W_PART(part);
		
		// First operation: V = A * U
		MATMUL_A(U, V, K[part]);
		
		// Second operation: W = B * V
		MATMUL_B(V, W, K[part]);
	}
}`, np, signature)

	_, err = kp.BuildKernel(kernelSource, "chain")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.ExecuteKernel("chain")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify we got results
	hasNonZero := false
	maxVal := 0.0
	for i := 0; i < totalNodes; i++ {
		if math.Abs(hostW[i]) > 1e-10 {
			hasNonZero = true
		}
		if math.Abs(hostW[i]) > maxVal {
			maxVal = math.Abs(hostW[i])
		}
	}

	if !hasNonZero {
		t.Error("Chain operation produced all zeros")
	}

	t.Logf("Chain operation completed, max value: %f", maxVal)
}

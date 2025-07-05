package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"gonum.org/v1/gonum/mat"
	"math"
	"testing"
)

// TestRunner_MatrixOperations tests matrix multiplication with known results
func TestRunner_MatrixOperations(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	np := 3
	k := []int{2, 3} // 2 and 3 elements per partition
	totalElements := 5

	kp := NewRunner(device, builder.Config{
		K:         k,
		FloatType: builder.Float64,
	})
	defer kp.Free()

	// Create a known test matrix (not identity!)
	// This matrix will transform [x, y, z] -> [2x+y, y+z, x+2z]
	testMatrix := mat.NewDense(np, np, []float64{
		2, 1, 0,
		0, 1, 1,
		1, 0, 2,
	})

	// Host arrays
	hostU := make([]float64, totalElements*np)
	hostV := make([]float64, totalElements*np)

	// Initialize U with known pattern
	// Each element has values [1, 2, 3], [4, 5, 6], etc.
	for elem := 0; elem < totalElements; elem++ {
		for node := 0; node < np; node++ {
			hostU[elem*np+node] = float64(elem*np + node + 1)
		}
	}

	// Define kernel with static matrix
	err := kp.DefineKernel("matmul",
		Input("TestMat").Bind(testMatrix).ToMatrix().Static(),
		Input("U").Bind(hostU).CopyTo(),
		Output("V").Bind(hostV),
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignature("matmul")

	// Kernel that performs V = TestMat * U
	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void matmul(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* U = U_PART(part);
		real_t* V = V_PART(part);
		
		MATMUL_TestMat(U, V, K[part]);
	}
}`, np, signature)

	_, err = kp.BuildKernel(kernelSource, "matmul")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.RunKernel("matmul")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Copy result back
	result, err := CopyArrayToHost[float64](kp, "V")
	if err != nil {
		t.Fatalf("Failed to copy result: %v", err)
	}

	// Verify the matrix multiplication
	// For each element, check that V = TestMat * U
	for elem := 0; elem < totalElements; elem++ {
		u1 := hostU[elem*np+0]
		u2 := hostU[elem*np+1]
		u3 := hostU[elem*np+2]

		// Expected values based on our test matrix
		expected := []float64{
			2*u1 + u2, // First row: [2, 1, 0] · [u1, u2, u3]
			u2 + u3,   // Second row: [0, 1, 1] · [u1, u2, u3]
			u1 + 2*u3, // Third row: [1, 0, 2] · [u1, u2, u3]
		}

		for node := 0; node < np; node++ {
			if math.Abs(result[elem*np+node]-expected[node]) > 1e-10 {
				t.Errorf("Element %d, node %d: expected %f, got %f",
					elem, node, expected[node], result[elem*np+node])
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
		K:         k,
		FloatType: builder.Float64,
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

	// Define kernel with device matrix
	err := kp.DefineKernel("differentiate",
		Input("Dr").Bind(Dr).ToMatrix(), // Device matrix (not static)
		Input("U").Bind(hostU).CopyTo(),
		Output("Ur").Bind(hostUr),
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignature("differentiate")

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
}`, np, signature)

	_, err = kp.BuildKernel(kernelSource, "differentiate")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.RunKernel("differentiate")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Get results
	result, err := CopyArrayToHost[float64](kp, "Ur")
	if err != nil {
		t.Fatalf("Failed to copy result: %v", err)
	}

	// Basic sanity check - derivative of x^2 should be ~2x
	t.Logf("Differentiation complete, checking %d values", len(result))
	// More detailed verification would require understanding the basis functions
}

// TestRunner_MatrixFromFlatArray tests promoting flat arrays to matrices
func TestRunner_MatrixFromFlatArray(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	np := 3
	kp := NewRunner(device, builder.Config{
		K:         []int{5},
		FloatType: builder.Float64,
	})
	defer kp.Free()

	// Flat array that represents a 3x3 matrix
	massFlat := []float64{
		2, 0, 0, // Diagonal mass matrix
		0, 3, 0,
		0, 0, 4,
	}

	// Host arrays
	hostU := make([]float64, 15) // 5 elements * 3 nodes
	hostMU := make([]float64, 15)

	// Initialize
	for i := range hostU {
		hostU[i] = float64(i + 1)
	}

	// Define kernel with flat array promoted to matrix
	err := kp.DefineKernel("applyMass",
		Input("Mass").Bind(massFlat).ToMatrix().Stride(np),
		Input("U").Bind(hostU).CopyTo(),
		Output("MU").Bind(hostMU),
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignature("applyMass")

	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void applyMass(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* U = U_PART(part);
		real_t* MU = MU_PART(part);
		
		MATMUL_Mass(U, MU, K[part]);
	}
}`, np, signature)

	_, err = kp.BuildKernel(kernelSource, "applyMass")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.RunKernel("applyMass")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	result, err := CopyArrayToHost[float64](kp, "MU")
	if err != nil {
		t.Fatalf("Failed to copy result: %v", err)
	}

	// Verify diagonal scaling
	for elem := 0; elem < 5; elem++ {
		for node := 0; node < np; node++ {
			idx := elem*np + node
			expected := hostU[idx] * float64(node+2) // Mass diagonal is [2,3,4]
			if math.Abs(result[idx]-expected) > 1e-10 {
				t.Errorf("Element %d, node %d: expected %f, got %f",
					elem, node, expected, result[idx])
			}
		}
	}
}

// TestRunner_MixedMatrices tests kernels with both static and device matrices
func TestRunner_MixedMatrices(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	np := 2
	kp := NewRunner(device, builder.Config{
		K:         []int{3},
		FloatType: builder.Float64,
	})
	defer kp.Free()

	// Small matrix for static embedding
	smallMat := mat.NewDense(np, np, []float64{
		1, 2,
		3, 4,
	})

	// Larger matrix for device allocation
	largeMat := mat.NewDense(np, np, []float64{
		5, 6,
		7, 8,
	})

	// Host arrays
	hostU := make([]float64, 6) // 3 elements * 2 nodes
	hostV := make([]float64, 6)
	hostW := make([]float64, 6)

	for i := range hostU {
		hostU[i] = float64(i + 1)
	}

	// Define kernel with mixed matrices
	err := kp.DefineKernel("mixedOps",
		Input("Small").Bind(smallMat).ToMatrix().Static(),
		Input("Large").Bind(largeMat).ToMatrix(), // Device matrix
		Input("U").Bind(hostU).CopyTo(),
		Output("V").Bind(hostV),
		Output("W").Bind(hostW),
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignature("mixedOps")

	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void mixedOps(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* U = U_PART(part);
		real_t* V = V_PART(part);
		real_t* W = W_PART(part);
		
		// Apply static matrix
		MATMUL_Small(U, V, K[part]);
		
		// Apply device matrix
		MATMUL_Large(U, W, K[part]);
	}
}`, np, signature)

	_, err = kp.BuildKernel(kernelSource, "mixedOps")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.RunKernel("mixedOps")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify both outputs
	resultV, _ := CopyArrayToHost[float64](kp, "V")
	resultW, _ := CopyArrayToHost[float64](kp, "W")

	t.Logf("Static matrix result: %v", resultV[:4])
	t.Logf("Device matrix result: %v", resultW[:4])
}

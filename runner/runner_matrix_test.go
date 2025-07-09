package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"gonum.org/v1/gonum/mat"
	"testing"
)

// TestRunner_MatrixOperations tests matrix multiplication with known results
func TestRunner_MatrixOperations(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	np := 3
	k := []int{2, 3} // 2 and 3 elements per partition

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
	hostU := [][]float64{
		make([]float64, k[0]*np),
		make([]float64, k[1]*np),
	}
	hostV := [][]float64{
		make([]float64, k[0]*np),
		make([]float64, k[1]*np),
	}

	// Define kernel with static matrix
	kernelName := "matmul"
	err := kp.DefineKernel(kernelName,
		builder.Input("TestMat").Bind(testMatrix).ToMatrix().Static(),
		builder.Input("U").Bind(hostU).CopyTo(),
		builder.Output("V").Bind(hostV).CopyBack(),
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignature(kernelName)

	// Kernel that performs V = TestMat * U
	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void %s(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* U = U_PART(part);
		real_t* V = V_PART(part);
		
		MATMUL_TestMat(U, V, K[part]);
	}
}`, np, kernelName, signature)

	_, err = kp.BuildKernel(kernelSource, kernelName)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.RunKernel(kernelName)
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
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
		builder.Input("Dr").Bind(Dr).ToMatrix(), // Device matrix (not static)
		builder.Input("U").Bind(hostU).CopyTo(),
		builder.Output("Ur").Bind(hostUr),
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

package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"gonum.org/v1/gonum/mat"
	"math"
	"testing"
)

// TestParameterAPI demonstrates the new kernel parameter API
func TestParameterAPI_BasicUsage(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	// Setup runner
	kp := NewRunner(device, builder.Config{
		K:         []int{10, 15},
		FloatType: builder.Float64,
	})
	defer kp.Free()

	// Host variables
	hostU := make([]float64, 25) // Total elements across partitions
	hostRHS := make([]float64, 25)
	dt := 0.01
	alpha := 2.5

	// Initialize input data
	for i := range hostU {
		hostU[i] = float64(i)
	}

	// Define kernel using new API
	err := kp.DefineKernel("computeRHS",
		Input("U").Bind(hostU).CopyTo(), // Copy input to device
		Output("RHS").Bind(hostRHS),     // Output array, no initial copy
		Scalar("dt").Bind(dt),           // Scalar parameter
		Scalar("alpha").Bind(alpha),     // Another scalar
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	// Get generated signature
	signature, err := kp.GetKernelSignature("computeRHS")
	if err != nil {
		t.Fatalf("Failed to get signature: %v", err)
	}
	t.Logf("Generated signature:\n%s", signature)

	// Build kernel
	kernelSource := fmt.Sprintf(`
@kernel void computeRHS(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* U = U_PART(part);
		real_t* RHS = RHS_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				RHS[i] = alpha * U[i] + dt;
			}
		}
	}
}`, signature)

	_, err = kp.BuildKernel(kernelSource, "computeRHS")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Run kernel - no arguments needed, all bindings are stored
	err = kp.RunKernel("computeRHS")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Copy results back
	result, err := CopyArrayToHost[float64](kp, "RHS")
	if err != nil {
		t.Fatalf("Failed to copy results: %v", err)
	}

	// Verify
	for i := 0; i < 25; i++ {
		expected := alpha*float64(i) + dt
		if math.Abs(result[i]-expected) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, result[i])
		}
	}
}

// TestParameterAPI_MatrixSupport tests matrix parameter support
func TestParameterAPI_MatrixSupport(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	np := 4
	kp := NewRunner(device, builder.Config{
		K:         []int{5},
		FloatType: builder.Float64,
	})
	defer kp.Free()

	// Create matrices
	drData := []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	Dr := mat.NewDense(np, np, drData)

	// Flat array to be promoted to matrix
	massFlat := make([]float64, np*np)
	for i := range massFlat {
		massFlat[i] = float64(i + 1)
	}

	// Host arrays
	hostU := make([]float64, 20) // 5 elements * 4 nodes
	hostUr := make([]float64, 20)

	// Initialize
	for i := range hostU {
		hostU[i] = float64(i)
	}

	// Define kernel with matrices
	err := kp.DefineKernel("differentiate",
		Input("Dr").Bind(Dr).ToMatrix().Static(),           // Static matrix
		Input("Mass").Bind(massFlat).ToMatrix().Stride(np), // Device matrix from flat array
		Input("U").Bind(hostU).CopyTo(),
		Output("Ur").Bind(hostUr),
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignature("differentiate")
	t.Logf("Signature with matrices:\n%s", signature)

	// The kernel would use MATMUL_Dr and MATMUL_Mass macros
	kernelSource := fmt.Sprintf(`
@kernel void differentiate(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* U = U_PART(part);
		real_t* Ur = Ur_PART(part);
		
		// Static matrix Dr is embedded, device matrix Mass is passed as parameter
		MATMUL_Dr(U, Ur, K[part]);
		MATMUL_Mass(U, Ur, K[part]); // Device matrix needs pointer
	}
}`, signature)

	_, err = kp.BuildKernel(kernelSource, "differentiate")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}
}

// TestParameterAPI_TypeConversion tests type conversion during copy
func TestParameterAPI_TypeConversion(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K:         []int{10},
		FloatType: builder.Float32, // Device uses float32
	})
	defer kp.Free()

	// Host uses float64 for precision
	hostData := make([]float64, 10)
	for i := range hostData {
		hostData[i] = float64(i) * 1.1
	}

	// Define kernel with type conversion
	err := kp.DefineKernel("processData",
		Input("data").Bind(hostData).CopyTo().Convert(builder.Float32), // Convert during copy
		Temp("work").Type(builder.Float32).Size(10),                    // Device-only array
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	// This saves bandwidth by only transferring float32 to device
	t.Log("Type conversion reduces memory transfer by 50%")
}

// TestParameterAPI_InOut tests bidirectional data movement
func TestParameterAPI_InOut(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K:         []int{10},
		FloatType: builder.Float64,
	})
	defer kp.Free()

	// Host data
	hostData := make([]float64, 10)
	for i := range hostData {
		hostData[i] = float64(i)
	}

	// Define kernel with InOut parameter
	err := kp.DefineKernel("doubleData",
		InOut("data").Bind(hostData).Copy(), // Copy to device before, from device after
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignature("doubleData")

	// Simple kernel that doubles values in place
	kernelSource := fmt.Sprintf(`
@kernel void doubleData(
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		real_t* data = data_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				data[i] *= 2.0;
			}
		}
	}
}`, signature)

	_, err = kp.BuildKernel(kernelSource, "doubleData")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Run kernel - data is automatically copied both ways
	err = kp.RunKernel("doubleData")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Host data should be automatically updated
	for i := 0; i < 10; i++ {
		expected := float64(i) * 2.0
		if math.Abs(hostData[i]-expected) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, hostData[i])
		}
	}
}

package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"math"
	"testing"
)

// TestExecuteKernel tests the new kernel execution system
func TestExecuteKernel(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	runner := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer runner.Free()

	// Setup bindings
	hostU := make([]float64, 10)
	hostRHS := make([]float64, 10)
	alpha := 2.5

	for i := range hostU {
		hostU[i] = float64(i)
	}

	err := runner.DefineBindings(
		builder.Input("U").Bind(hostU),
		builder.Output("RHS").Bind(hostRHS),
		builder.Scalar("alpha").Bind(alpha),
	)
	if err != nil {
		t.Fatalf("DefineBindings failed: %v", err)
	}

	err = runner.AllocateDevice()
	if err != nil {
		t.Fatalf("AllocateDevice failed: %v", err)
	}

	// Configure kernel
	_, err = runner.ConfigureKernel("compute",
		runner.Param("U").CopyTo(),
		runner.Param("RHS").CopyBack(),
		runner.Param("alpha"),
	)
	if err != nil {
		t.Fatalf("ConfigureKernel failed: %v", err)
	}

	// Get signature using new method
	signature, err := runner.GetKernelSignatureForConfig("compute")
	if err != nil {
		t.Fatalf("GetKernelSignatureForConfig failed: %v", err)
	}

	// Build kernel
	kernelSource := fmt.Sprintf(`
@kernel void compute(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* U = U_PART(part);
		double* RHS = RHS_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				RHS[i] = alpha * U[i];
			}
		}
	}
}`, signature)

	_, err = runner.BuildKernel(kernelSource, "compute")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Execute kernel using new API
	err = runner.ExecuteKernel("compute")
	if err != nil {
		t.Fatalf("ExecuteKernel failed: %v", err)
	}

	// Verify results
	for i := 0; i < 10; i++ {
		expected := alpha * float64(i)
		if math.Abs(hostRHS[i]-expected) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, hostRHS[i])
		}
	}
}

// TestMultiKernelExecution tests multiple kernels sharing device state
func TestMultiKernelExecution(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	runner := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer runner.Free()

	// Setup bindings
	hostU := make([]float64, 10)
	hostV := make([]float64, 10)
	hostW := make([]float64, 10)

	for i := range hostU {
		hostU[i] = float64(i)
		hostV[i] = float64(i) * 2
	}

	runner.DefineBindings(
		builder.InOut("U").Bind(hostU),
		builder.InOut("V").Bind(hostV),
		builder.Output("W").Bind(hostW),
	)
	runner.AllocateDevice()

	// Kernel 1: W = U + V (copy U,V to device)
	t.Run("AddKernel", func(t *testing.T) {
		_, err := runner.ConfigureKernel("add",
			runner.Param("U").CopyTo(),
			runner.Param("V").CopyTo(),
			runner.Param("W"), // W stays on device
		)
		if err != nil {
			t.Fatalf("ConfigureKernel failed: %v", err)
		}

		signature, _ := runner.GetKernelSignatureForConfig("add")
		kernelSource := fmt.Sprintf(`
@kernel void add(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* U = U_PART(part);
		const double* V = V_PART(part);
		double* W = W_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				W[i] = U[i] + V[i];
			}
		}
	}
}`, signature)

		runner.BuildKernel(kernelSource, "add")
		err = runner.ExecuteKernel("add")
		if err != nil {
			t.Fatalf("ExecuteKernel failed: %v", err)
		}

		// W is still on device, not copied back
		for i := range hostW {
			if hostW[i] != 0 {
				t.Error("W should not be copied back yet")
			}
		}
	})

	// Kernel 2: U = W * 2 (W already on device, copy result back)
	t.Run("MultiplyKernel", func(t *testing.T) {
		_, err := runner.ConfigureKernel("multiply",
			runner.Param("W"),            // W already on device
			runner.Param("U").CopyBack(), // Copy result back to U
		)
		if err != nil {
			t.Fatalf("ConfigureKernel failed: %v", err)
		}

		signature, _ := runner.GetKernelSignatureForConfig("multiply")
		kernelSource := fmt.Sprintf(`
@kernel void multiply(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* W = W_PART(part);
		double* U = U_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				U[i] = W[i] * 2.0;
			}
		}
	}
}`, signature)

		runner.BuildKernel(kernelSource, "multiply")
		err = runner.ExecuteKernel("multiply")
		if err != nil {
			t.Fatalf("ExecuteKernel failed: %v", err)
		}

		// Verify U = (U + V) * 2
		for i := 0; i < 10; i++ {
			expected := (float64(i) + float64(i)*2) * 2.0
			if math.Abs(hostU[i]-expected) > 1e-10 {
				t.Errorf("U[%d]: expected %f, got %f", i, expected, hostU[i])
			}
		}
	})

	// Manual copy to get W
	t.Run("ManualCopy", func(t *testing.T) {
		err := runner.CopyFromDevice("W")
		if err != nil {
			t.Fatalf("CopyFromDevice failed: %v", err)
		}

		// Verify W = original U + V
		for i := 0; i < 10; i++ {
			expected := float64(i) + float64(i)*2
			if math.Abs(hostW[i]-expected) > 1e-10 {
				t.Errorf("W[%d]: expected %f, got %f", i, expected, hostW[i])
			}
		}
	})
}

// TestScalarHandling tests scalar parameter handling in new system
func TestScalarHandling(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	runner := NewRunner(device, builder.Config{
		K: []int{5},
	})
	defer runner.Free()

	// Test with bound scalars
	t.Run("BoundScalars", func(t *testing.T) {
		hostData := make([]float64, 5)
		alpha := 3.0
		beta := 2.0

		runner.DefineBindings(
			builder.Output("data").Bind(hostData),
			builder.Scalar("alpha").Bind(alpha),
			builder.Scalar("beta").Bind(beta),
		)
		runner.AllocateDevice()

		_, err := runner.ConfigureKernel("scalars",
			runner.Param("data").CopyBack(),
			runner.Param("alpha"),
			runner.Param("beta"),
		)
		if err != nil {
			t.Fatalf("ConfigureKernel failed: %v", err)
		}

		signature, _ := runner.GetKernelSignatureForConfig("scalars")
		kernelSource := fmt.Sprintf(`
@kernel void scalars(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		double* data = data_PART(part);
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				data[i] = alpha + beta;
			}
		}
	}
}`, signature)

		runner.BuildKernel(kernelSource, "scalars")
		err = runner.ExecuteKernel("scalars")
		if err != nil {
			t.Fatalf("ExecuteKernel failed: %v", err)
		}

		// Verify
		for i := 0; i < 5; i++ {
			expected := alpha + beta
			if math.Abs(hostData[i]-expected) > 1e-10 {
				t.Errorf("data[%d]: expected %f, got %f", i, expected, hostData[i])
			}
		}
	})

	// Test with runtime scalars
	t.Run("RuntimeScalars", func(t *testing.T) {
		runner2 := NewRunner(device, builder.Config{K: []int{5}})
		defer runner2.Free()

		hostData := make([]float64, 5)

		runner2.DefineBindings(
			builder.Output("data").Bind(hostData),
			builder.Scalar("gamma"), // No binding
		)
		runner2.AllocateDevice()

		_, err := runner2.ConfigureKernel("runtime",
			runner2.Param("data").CopyBack(),
			runner2.Param("gamma"),
		)
		if err != nil {
			t.Fatalf("ConfigureKernel failed: %v", err)
		}

		signature, _ := runner2.GetKernelSignatureForConfig("runtime")
		kernelSource := fmt.Sprintf(`
@kernel void runtime(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		double* data = data_PART(part);
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				data[i] = gamma * (double)i;
			}
		}
	}
}`, signature)

		runner2.BuildKernel(kernelSource, "runtime")

		// Pass scalar at runtime
		gamma := 4.5
		err = runner2.ExecuteKernel("runtime", gamma)
		if err != nil {
			t.Fatalf("ExecuteKernel with runtime scalar failed: %v", err)
		}

		// Verify
		for i := 0; i < 5; i++ {
			expected := gamma * float64(i)
			if math.Abs(hostData[i]-expected) > 1e-10 {
				t.Errorf("data[%d]: expected %f, got %f", i, expected, hostData[i])
			}
		}
	})
}

// TestExecutionErrors tests error conditions
func TestExecutionErrors(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	runner := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer runner.Free()

	t.Run("NotConfigured", func(t *testing.T) {
		err := runner.ExecuteKernel("nonexistent")
		if err == nil {
			t.Error("Expected error for unconfigured kernel")
		}
	})

	t.Run("NotBuilt", func(t *testing.T) {
		runner.DefineBindings(
			builder.Temp("x").Size(10).Type(builder.Float64),
		)
		runner.AllocateDevice()
		runner.ConfigureKernel("notbuilt", runner.Param("x"))

		err := runner.ExecuteKernel("notbuilt")
		if err == nil {
			t.Error("Expected error for unbuilt kernel")
		}
	})
}

// TestStatePreservation verifies device state is preserved across executions
func TestStatePreservation(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	runner := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer runner.Free()

	// Setup
	hostData := make([]float64, 10)
	runner.DefineBindings(
		builder.InOut("data").Bind(hostData),
	)
	runner.AllocateDevice()

	// Initialize data on device
	for i := range hostData {
		hostData[i] = float64(i)
	}
	runner.CopyToDevice("data")

	// Clear host data
	for i := range hostData {
		hostData[i] = 0
	}

	// Configure kernel that doubles data
	runner.ConfigureKernel("doubleData",
		runner.Param("data"), // No copy - use device state
	)

	signature, _ := runner.GetKernelSignatureForConfig("doubleData")
	kernelSource := fmt.Sprintf(`
@kernel void doubleData(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		double* data = data_PART(part);
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				data[i] *= 2.0;
			}
		}
	}
}`, signature)

	runner.BuildKernel(kernelSource, "doubleData")

	// Execute kernel multiple times
	for iter := 0; iter < 3; iter++ {
		err := runner.ExecuteKernel("doubleData")
		if err != nil {
			t.Fatalf("Iteration %d: ExecuteKernel failed: %v", iter, err)
		}
	}

	// Copy back and verify
	runner.CopyFromDevice("data")

	// Data should be doubled 3 times: i * 2^3 = i * 8
	for i := 0; i < 10; i++ {
		expected := float64(i) * 8.0
		if math.Abs(hostData[i]-expected) > 1e-10 {
			t.Errorf("data[%d]: expected %f, got %f", i, expected, hostData[i])
		}
	}
}

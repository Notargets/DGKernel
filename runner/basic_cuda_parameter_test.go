package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"github.com/notargets/gocca"
	"math"
	"testing"
)

// TestScalarMinimal - absolute minimal test
func _TestScalarMinimal(t *testing.T) {
	device := utils.CreateTestDevice(true) // Force CUDA
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K:         []int{1},
		FloatType: builder.Float64,
	})
	defer kp.Free()

	// Test case 1: Verify memory allocation works
	t.Run("MemoryWorks", func(t *testing.T) {
		hostOutput := make([]float64, 1)
		hostOutput[0] = -999.0

		err := kp.DefineKernel("memtest",
			builder.Output("output").Bind(hostOutput).CopyBack(),
		)
		if err != nil {
			t.Fatalf("Failed to define kernel: %v", err)
		}

		signature, _ := kp.GetKernelSignature("memtest")
		kernelSource := fmt.Sprintf(`
@kernel void memtest(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		real_t* output = output_PART(part);
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				output[i] = 42.0;
			}
		}
	}
}`, signature)

		_, err = kp.BuildKernel(kernelSource, "memtest")
		if err != nil {
			t.Fatalf("Failed to build kernel: %v", err)
		}

		err = kp.RunKernel("memtest")
		if err != nil {
			t.Fatalf("Kernel execution failed: %v", err)
		}

		if math.Abs(hostOutput[0]-42.0) > 1e-10 {
			t.Errorf("Memory test failed: expected 42.0, got %f", hostOutput[0])
		} else {
			t.Logf("✓ Memory allocation and copy works")
		}
	})

	// Test case 2: Print what's in the kernel definition
	t.Run("PrintDefinition", func(t *testing.T) {
		hostOutput := make([]float64, 1)
		alpha := 2.5

		err := kp.DefineKernel("printtest",
			builder.Output("output").Bind(hostOutput),
			builder.Scalar("alpha").Bind(alpha),
		)
		if err != nil {
			t.Fatalf("Failed to define kernel: %v", err)
		}

		def := kp.kernelDefinitions["printtest"]
		t.Logf("\nKernel Definition for 'printtest':")
		t.Logf("  Name: %s", def.Name)
		t.Logf("  Signature:\n%s", def.Signature)
		t.Logf("  Parameters:")
		for i, p := range def.Parameters {
			t.Logf("    [%d] %s: Direction=%v, DataType=%v, Binding=%v",
				i, p.Name, p.Direction, p.DataType, p.HostBinding)
		}
	})

	// Test case 3: Direct comparison OpenMP vs CUDA
	t.Run("CompareBackends", func(t *testing.T) {
		testScalarOnDevice := func(dev *gocca.OCCADevice, mode string) (float64, error) {
			runner := NewRunner(dev, builder.Config{
				K:         []int{1},
				FloatType: builder.Float64,
			})
			defer runner.Free()

			hostOutput := make([]float64, 1)
			alpha := 2.5

			err := runner.DefineKernel("compare",
				builder.Output("output").Bind(hostOutput).CopyBack(),
				builder.Scalar("alpha").Bind(alpha),
			)
			if err != nil {
				return 0, fmt.Errorf("DefineKernel failed: %v", err)
			}

			signature, _ := runner.GetKernelSignature("compare")
			kernelSource := fmt.Sprintf(`
@kernel void compare(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		real_t* output = output_PART(part);
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				output[i] = alpha;
			}
		}
	}
}`, signature)

			_, err = runner.BuildKernel(kernelSource, "compare")
			if err != nil {
				return 0, fmt.Errorf("BuildKernel failed: %v", err)
			}

			err = runner.RunKernel("compare")
			if err != nil {
				return 0, fmt.Errorf("RunKernel failed: %v", err)
			}

			return hostOutput[0], nil
		}

		// Test on current device (CUDA)
		cudaResult, err := testScalarOnDevice(device, "CUDA")
		if err != nil {
			t.Fatalf("CUDA test failed: %v", err)
		}
		t.Logf("CUDA result: %f", cudaResult)

		// Test on OpenMP if available
		ompDevice := utils.CreateTestDevice()
		ompResult, err := testScalarOnDevice(ompDevice, "OpenMP")
		if err != nil {
			t.Logf("OpenMP test failed: %v", err)
		} else {
			t.Logf("OpenMP result: %f", ompResult)

			if cudaResult != ompResult {
				t.Errorf("Backend mismatch: CUDA=%f, OpenMP=%f", cudaResult, ompResult)
			}
		}
	})
}

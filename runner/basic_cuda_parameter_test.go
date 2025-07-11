package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"github.com/notargets/gocca"
	"math"
	"strings"
	"testing"
	"unsafe"
)

// TestScalarMinimal - absolute minimal test
func TestScalarMinimal(t *testing.T) {
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

// TestCudaScalarDebug - Debug CUDA scalar parameter issue
func TestCudaScalarDebug(t *testing.T) {
	device := utils.CreateTestDevice(true) // Force CUDA
	defer device.Free()

	if device.Mode() != "CUDA" {
		t.Skip("This test is CUDA-specific")
	}

	kp := NewRunner(device, builder.Config{
		K:         []int{1},
		FloatType: builder.Float64,
	})
	defer kp.Free()

	// Test 1: Hardcoded scalar in kernel
	t.Run("HardcodedScalar", func(t *testing.T) {
		hostOutput := make([]float64, 1)

		err := kp.DefineKernel("hardcoded",
			builder.Output("output").Bind(hostOutput).CopyBack(),
		)
		if err != nil {
			t.Fatalf("Failed to define kernel: %v", err)
		}

		signature, _ := kp.GetKernelSignature("hardcoded")
		// Use hardcoded value instead of parameter
		kernelSource := fmt.Sprintf(`
@kernel void hardcoded(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		real_t* output = output_PART(part);
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				output[i] = 2.5;  // Hardcoded value
			}
		}
	}
}`, signature)

		_, err = kp.BuildKernel(kernelSource, "hardcoded")
		if err != nil {
			t.Fatalf("Failed to build kernel: %v", err)
		}

		err = kp.RunKernel("hardcoded")
		if err != nil {
			t.Fatalf("Kernel execution failed: %v", err)
		}

		if math.Abs(hostOutput[0]-2.5) > 1e-10 {
			t.Errorf("Hardcoded test failed: expected 2.5, got %f", hostOutput[0])
		} else {
			t.Logf("✓ Hardcoded scalar works: got %f", hostOutput[0])
		}
	})

	// Test 2: Scalar parameter with different kernel structure
	if false {
		t.Run("ScalarWithDebug", func(t *testing.T) {
			hostOutput := make([]float64, 1)
			hostDebug := make([]float64, 3) // For debug values
			alpha := 2.5

			err := kp.DefineKernel("scalardebug",
				builder.Output("output").Bind(hostOutput).CopyBack(),
				builder.Output("debug").Bind(hostDebug).CopyBack(),
				builder.Scalar("alpha").Bind(alpha),
			)
			if err != nil {
				t.Fatalf("Failed to define kernel: %v", err)
			}

			signature, _ := kp.GetKernelSignature("scalardebug")
			kernelSource := fmt.Sprintf(`
@kernel void scalardebug(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		real_t* output = output_PART(part);
		real_t* debug = debug_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				// Debug info
				if (i == 0) {
					debug[0] = 1.0;        // Kernel executed
					debug[1] = alpha;      // Scalar value received
					debug[2] = (real_t)K[part]; // K value
				}
				
				output[i] = alpha;
			}
		}
	}
}`, signature)

			_, err = kp.BuildKernel(kernelSource, "scalardebug")
			if err != nil {
				t.Fatalf("Failed to build kernel: %v", err)
			}

			err = kp.RunKernel("scalardebug")
			if err != nil {
				t.Fatalf("Kernel execution failed: %v", err)
			}

			t.Logf("Debug values:")
			t.Logf("  Kernel executed: %f (should be 1.0)", hostDebug[0])
			t.Logf("  Alpha received:  %f (should be 2.5)", hostDebug[1])
			t.Logf("  K[part] value:   %f (should be 1.0)", hostDebug[2])
			t.Logf("  Output value:    %f (should be 2.5)", hostOutput[0])

			if math.Abs(hostOutput[0]-alpha) > 1e-10 {
				t.Errorf("Scalar parameter test failed: expected %f, got %f", alpha, hostOutput[0])
			}
		})
	}

	// Test 3: Multiple scalars
	t.Run("MultipleScalars", func(t *testing.T) {
		hostOutput := make([]float64, 1)
		alpha := 2.5
		beta := 3.0

		err := kp.DefineKernel("multiscalar",
			builder.Output("output").Bind(hostOutput).CopyBack(),
			builder.Scalar("alpha").Bind(alpha),
			builder.Scalar("beta").Bind(beta),
		)
		if err != nil {
			t.Fatalf("Failed to define kernel: %v", err)
		}

		signature, _ := kp.GetKernelSignature("multiscalar")
		kernelSource := fmt.Sprintf(`
@kernel void multiscalar(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		real_t* output = output_PART(part);
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				output[i] = alpha + beta;  // Should be 5.5
			}
		}
	}
}`, signature)

		_, err = kp.BuildKernel(kernelSource, "multiscalar")
		if err != nil {
			t.Fatalf("Failed to build kernel: %v", err)
		}

		err = kp.RunKernel("multiscalar")
		if err != nil {
			t.Fatalf("Kernel execution failed: %v", err)
		}

		expected := alpha + beta
		if math.Abs(hostOutput[0]-expected) > 1e-10 {
			t.Errorf("Multiple scalars test failed: expected %f, got %f", expected, hostOutput[0])
		} else {
			t.Logf("✓ Multiple scalars work: got %f", hostOutput[0])
		}
	})
}

// TestTraceAllocatedArrays - Trace when arrays are added to AllocatedArrays
func TestTraceAllocatedArrays(t *testing.T) {
	device := utils.CreateTestDevice(true) // Force CUDA
	defer device.Free()

	kp := NewRunner(device, builder.Config{
		K:         []int{1},
		FloatType: builder.Float64,
	})
	defer kp.Free()

	// Step 1: Initial state
	t.Logf("Initial AllocatedArrays: %v", kp.GetAllocatedArrays())

	// Step 2: Define kernel with two outputs
	hostOutput := make([]float64, 1)
	hostDebug := make([]float64, 3)
	alpha := 2.5

	t.Log("Defining kernel with two outputs...")
	err := kp.DefineKernel("test",
		builder.Output("output").Bind(hostOutput).CopyBack(),
		builder.Output("debug").Bind(hostDebug).CopyBack(),
		builder.Scalar("alpha").Bind(alpha),
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	// Step 3: Check AllocatedArrays after DefineKernel
	t.Logf("After DefineKernel, AllocatedArrays: %v", kp.GetAllocatedArrays())

	// Step 4: Check if memory was allocated
	for _, name := range []string{"output", "debug"} {
		if mem := kp.GetMemory(name); mem == nil {
			t.Errorf("Memory for %s not allocated", name)
		} else {
			t.Logf("Memory for %s: allocated", name)
		}
	}

	// Step 5: Generate preamble and check for macros
	preamble := kp.GeneratePreamble()
	t.Log("Generated preamble:")

	// Check for partition macros
	requiredMacros := []string{
		"#define output_PART(part)",
		"#define debug_PART(part)",
	}

	for _, macro := range requiredMacros {
		if strings.Contains(preamble, macro) {
			t.Logf("✓ Found macro: %s", macro)
		} else {
			t.Errorf("✗ Missing macro: %s", macro)
		}
	}

	// Print the partition macros section
	lines := strings.Split(preamble, "\n")
	inMacroSection := false
	t.Log("\nPartition macro section:")
	for _, line := range lines {
		if strings.Contains(line, "// Partition access macros") {
			inMacroSection = true
		}
		if inMacroSection && strings.Contains(line, "_PART(part)") {
			t.Logf("  %s", line)
		}
		if inMacroSection && line == "" {
			break
		}
	}
}

// TestSimplestCudaScalar - Absolute simplest test case
func TestSimplestCudaScalar(t *testing.T) {
	device := utils.CreateTestDevice(true) // Force CUDA
	defer device.Free()

	if device.Mode() != "CUDA" {
		t.Skip("This test is CUDA-specific")
	}

	kp := NewRunner(device, builder.Config{
		K:         []int{1},
		FloatType: builder.Float64,
	})
	defer kp.Free()

	// Single output, single scalar
	hostOutput := make([]float64, 1)
	alpha := 2.5

	err := kp.DefineKernel("simple",
		builder.Output("output").Bind(hostOutput).CopyBack(),
		builder.Scalar("alpha").Bind(alpha),
	)
	if err != nil {
		t.Fatalf("Failed to define kernel: %v", err)
	}

	signature, _ := kp.GetKernelSignature("simple")
	t.Logf("Signature:\n%s", signature)

	// Check AllocatedArrays
	t.Logf("AllocatedArrays: %v", kp.GetAllocatedArrays())

	// Generate and print preamble
	preamble := kp.GeneratePreamble()

	// Extract just the partition macros section
	lines := strings.Split(preamble, "\n")
	for i, line := range lines {
		if strings.Contains(line, "// Partition access macros") {
			t.Log("Partition macros section:")
			for j := i; j < len(lines) && j < i+10; j++ {
				t.Logf("  %s", lines[j])
			}
			break
		}
	}

	// Build kernel
	kernelSource := fmt.Sprintf(`
@kernel void simple(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		real_t* output = output_PART(part);
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				output[i] = alpha;
			}
		}
	}
}`, signature)

	_, err = kp.BuildKernel(kernelSource, "simple")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.RunKernel("simple")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	if hostOutput[0] != alpha {
		t.Errorf("Expected %f, got %f", alpha, hostOutput[0])
	} else {
		t.Logf("✓ CUDA scalar parameter works: got %f", hostOutput[0])
	}
}

// TestDirectOCCAScalar - Test scalar parameters directly with OCCA
func TestDirectOCCAScalar(t *testing.T) {
	// Test both CUDA and OpenMP
	testModes := []string{
		`{"mode": "CUDA", "device_id": 0}`,
		`{"mode": "OpenMP"}`,
	}

	for _, mode := range testModes {
		device, err := gocca.NewDevice(mode)
		if err != nil {
			t.Logf("Skipping %s: %v", mode, err)
			continue
		}
		defer device.Free()

		t.Run(device.Mode(), func(t *testing.T) {
			// Allocate memory for one double
			outputMem := device.Malloc(8, nil, nil)
			defer outputMem.Free()

			// Simple kernel with scalar parameter
			kernelSource := `
@kernel void testScalar(double *output, const double alpha) {
	for (int i = 0; i < 1; ++i; @outer) {
		for (int j = 0; j < 1; ++j; @inner) {
			output[0] = alpha;
		}
	}
}`

			kernel, err := device.BuildKernelFromString(kernelSource, "testScalar", nil)
			if err != nil {
				t.Fatalf("Failed to build kernel: %v", err)
			}
			defer kernel.Free()

			// Test 1: Direct scalar value
			alpha := 2.5
			err = kernel.RunWithArgs(outputMem, alpha)
			if err != nil {
				t.Fatalf("Failed to run kernel: %v", err)
			}
			device.Finish()

			// Check result
			result := make([]float64, 1)
			outputMem.CopyTo(unsafe.Pointer(&result[0]), 8)

			t.Logf("Direct scalar test result: %f (expected 2.5)", result[0])

			if result[0] != 2.5 {
				t.Errorf("Direct scalar test failed: expected 2.5, got %f", result[0])
			}

			// Test 2: Try with PushArg method
			outputMem.CopyFrom(unsafe.Pointer(&[]float64{0.0}[0]), 8) // Reset to 0

			kernel.ClearArgs()
			kernel.PushArg(outputMem)
			kernel.PushArg(alpha)
			kernel.RunFromArgs()
			device.Finish()

			outputMem.CopyTo(unsafe.Pointer(&result[0]), 8)
			t.Logf("PushArg test result: %f (expected 2.5)", result[0])

			if result[0] != 2.5 {
				t.Errorf("PushArg test failed: expected 2.5, got %f", result[0])
			}
		})
	}
}

// TestOCCAMemoryOperations - Test memory operations work correctly
func TestOCCAMemoryOperations(t *testing.T) {
	device, err := gocca.NewDevice(`{"mode": "CUDA", "device_id": 0}`)
	if err != nil {
		t.Skip("CUDA not available")
	}
	defer device.Free()

	// Test memory initialization and copy
	hostData := []float64{1.1, 2.2, 3.3, 4.4, 5.5}

	// Allocate and copy to device
	mem := device.Malloc(int64(len(hostData)*8), unsafe.Pointer(&hostData[0]), nil)
	defer mem.Free()

	// Copy back immediately
	result := make([]float64, len(hostData))
	mem.CopyTo(unsafe.Pointer(&result[0]), int64(len(result)*8))

	// Verify
	for i := range hostData {
		if result[i] != hostData[i] {
			t.Errorf("Memory copy failed at index %d: expected %f, got %f", i, hostData[i], result[i])
		}
	}
	t.Log("Memory operations work correctly")
}

package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"gonum.org/v1/gonum/mat"
	"math"
	"testing"
)

func TestSimpleScalarKernel(t *testing.T) {
	// device := utils.CreateTestDevice(true)
	device := utils.CreateTestDevice()
	defer device.Free()

	// Setup runner
	kp := NewRunner(device, builder.Config{
		K: []int{1},
	})
	defer kp.Free()

	// Host variables
	hostRHS := make([]float64, 1)
	alpha := 2.5

	// Phase 1: Define bindings (one-time setup)
	err := kp.DefineBindings(
		builder.Output("RHS").Bind(hostRHS),
		builder.Scalar("alpha").Bind(alpha),
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
	kernelName := "scalarRHS"
	_, err = kp.ConfigureKernel(kernelName,
		kp.Param("RHS").CopyBack(), // Output array
		kp.Param("alpha"),          // Scalar
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	// Get generated signature
	signature, err := kp.GetKernelSignatureForConfig(kernelName)
	if err != nil {
		t.Fatalf("Failed to get signature: %v", err)
	}
	t.Logf("Generated signature:\n%s", signature)

	// Build kernel
	kernelSource := fmt.Sprintf(`
@kernel void %s (
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		double* RHS = RHS_PART(part);
		for (int i = 0; i < KpartMax; ++i; @inner) {
			RHS[i] = alpha;
		}
	}
}`, kernelName, signature)

	_, err = kp.BuildKernel(kernelSource, kernelName)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Execute kernel - no arguments needed, all bindings are stored
	err = kp.ExecuteKernel(kernelName)
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	expected := alpha
	if math.Abs(hostRHS[0]-expected) > 1e-10 {
		t.Errorf("expected %f, got %f", expected, hostRHS[0])
	}
}

// TestParameterAPI demonstrates the new kernel parameter API
func TestParameterAPI_BasicUsage(t *testing.T) {
	// device := utils.CreateTestDevice(true)
	device := utils.CreateTestDevice()
	defer device.Free()

	// Setup runner
	kp := NewRunner(device, builder.Config{
		K: []int{25},
	})
	defer kp.Free()

	// Host variables
	hostU := make([]float64, 25)
	hostRHS := make([]float64, 25)
	dt := 0.01
	alpha := 2.5

	// Initialize input data
	for i := range hostU {
		hostU[i] = float64(i)
	}

	// Phase 1: Define bindings (one-time setup)
	err := kp.DefineBindings(
		builder.Input("U").Bind(hostU),
		builder.Output("RHS").Bind(hostRHS),
		builder.Scalar("dt").Bind(dt),
		builder.Scalar("alpha").Bind(alpha),
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
	kernelName := "computeRHS"
	_, err = kp.ConfigureKernel(kernelName,
		kp.Param("U").CopyTo(),     // Copy input to device
		kp.Param("RHS").CopyBack(), // Copy result back to host
		kp.Param("dt"),             // Scalar parameter
		kp.Param("alpha"),          // Scalar parameter
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	// Get generated signature
	signature, err := kp.GetKernelSignatureForConfig(kernelName)
	if err != nil {
		t.Fatalf("Failed to get signature: %v", err)
	}

	// Build kernel
	kernelSource := fmt.Sprintf(`
@kernel void %s (
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* U = U_PART(part);
		double* RHS = RHS_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				RHS[i] = alpha * U[i] + dt;
			}
		}
	}
}`, kernelName, signature)

	_, err = kp.BuildKernel(kernelSource, kernelName)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Execute kernel
	err = kp.ExecuteKernel(kernelName)
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify results
	for i := 0; i < 25; i++ {
		expected := alpha*float64(i) + dt
		if math.Abs(hostRHS[i]-expected) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, hostRHS[i])
		}
	}
}

// TestParameterAPI_MatrixOperations demonstrates matrix operations
func TestParameterAPI_MatrixOperations(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	// Matrix dimensions
	m, n := 4, 3
	numElements := 5

	kp := NewRunner(device, builder.Config{
		K: []int{numElements},
	})
	defer kp.Free()

	// Create test matrices
	A := mat.NewDense(m, n, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	})

	// Input and output vectors
	x := make([]float64, n*numElements)
	y := make([]float64, m*numElements)

	// Initialize input
	for k := 0; k < numElements; k++ {
		for i := 0; i < n; i++ {
			x[k*n+i] = float64(k + 1)
		}
	}

	// Phase 1: Define bindings
	err := kp.DefineBindings(
		builder.Input("A").Bind(A).ToMatrix(),
		builder.Input("x").Bind(x),
		builder.Output("y").Bind(y),
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
	kernelName := "matVecMul"
	_, err = kp.ConfigureKernel(kernelName,
		kp.Param("A"),            // Matrix already on device (ToMatrix)
		kp.Param("x").CopyTo(),   // Copy input vector
		kp.Param("y").CopyBack(), // Copy result back
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	// Get signature
	signature, err := kp.GetKernelSignatureForConfig(kernelName)
	if err != nil {
		t.Fatalf("Failed to get signature: %v", err)
	}

	// Build kernel with matrix multiplication
	kernelSource := fmt.Sprintf(`
@kernel void %s (
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* x = x_PART(part);
		double* y = y_PART(part);
		
		// Matrix dimensions
		const int m = %d;
		const int n = %d;
		
		for (int elem = 0; elem < K[part]; ++elem) {
			int ii = 0;
			for (int i = 0; i < m; ++i; @inner) {
				double sum = 0.0;
				for (int j = 0; j < n; ++j) {
					ii = i + m*j;
					sum += A[ii] * x[elem*n + j];
				}
				y[elem*m + i] = sum;
			}
		}
	}
}`, kernelName, signature, m, n)

	_, err = kp.BuildKernel(kernelSource, kernelName)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Execute kernel
	err = kp.ExecuteKernel(kernelName)
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify matrix multiplication results
	for k := 0; k < numElements; k++ {
		scale := float64(k + 1)
		expected := []float64{
			6 * scale,  // 1*k + 2*k + 3*k
			15 * scale, // 4*k + 5*k + 6*k
			24 * scale, // 7*k + 8*k + 9*k
			33 * scale, // 10*k + 11*k + 12*k
		}

		for i := 0; i < m; i++ {
			idx := k*m + i
			if math.Abs(y[idx]-expected[i]) > 1e-10 {
				t.Errorf("Element (%d,%d): expected %f, got %f",
					k, i, expected[i], y[idx])
			}
		}
	}
}

// TestParameterAPI_TemporaryArrays demonstrates temp array usage
func TestParameterAPI_TemporaryArrays(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	n := 20
	kp := NewRunner(device, builder.Config{
		K: []int{n},
	})
	defer kp.Free()

	// Host arrays
	input := make([]float64, n)
	output := make([]float64, n)

	for i := 0; i < n; i++ {
		input[i] = float64(i)
	}

	// Phase 1: Define bindings including temp arrays
	err := kp.DefineBindings(
		builder.Input("input").Bind(input),
		builder.Output("output").Bind(output),
		builder.Temp("work1").Type(builder.Float64).Size(n),
		builder.Temp("work2").Type(builder.Float64).Size(n),
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
	kernelName := "multiStageCompute"
	_, err = kp.ConfigureKernel(kernelName,
		kp.Param("input").CopyTo(),
		kp.Param("output").CopyBack(),
		kp.Param("work1"), // Temp arrays available on device
		kp.Param("work2"),
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	// Get signature
	signature, err := kp.GetKernelSignatureForConfig(kernelName)
	if err != nil {
		t.Fatalf("Failed to get signature: %v", err)
	}

	// Build kernel that uses temp arrays
	kernelSource := fmt.Sprintf(`
@kernel void %s (
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* input = input_PART(part);
		double* output = output_PART(part);
		double* work1 = work1_PART(part);
		double* work2 = work2_PART(part);
		
		// Stage 1: Square values into work1
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				work1[i] = input[i] * input[i];
			}
		}
		
		// Synchronization would happen here in real kernel
		
		// Stage 2: Add 10 to squared values
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				work2[i] = work1[i] + 10.0;
			}
		}
		
		// Stage 3: Final output
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				output[i] = work2[i];
			}
		}
	}
}`, kernelName, signature)

	_, err = kp.BuildKernel(kernelSource, kernelName)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Execute kernel
	err = kp.ExecuteKernel(kernelName)
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify: output = input^2 + 10
	for i := 0; i < n; i++ {
		expected := float64(i*i) + 10.0
		if math.Abs(output[i]-expected) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, output[i])
		}
	}
}

func splitSlice(splits []int, slice []float64) (splitSlice [][]float64) {
	var (
		total int
	)
	for _, k := range splits {
		total += k
	}
	if len(slice)%total != 0 {
		fmt.Printf("len(slice) = %d, total = %d\n", len(slice), total)
		panic("split slice is not a multiple of len(slice)")
	}
	stride := len(slice) / total
	splitSlice = make([][]float64, len(splits))
	var iii int
	for i, K := range splits {
		// This avoids a copy by just storing the addresses
		splitSlice[i] = slice[iii : iii+K*stride]
		iii += K * stride
	}
	return
}

// TestParameterAPI_MultiplePartitions tests with partitioned data
func TestParameterAPI_MultiplePartitions(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	// Multiple partitions
	k := []int{10, 15, 8}
	kp := NewRunner(device, builder.Config{
		K: k,
	})
	defer kp.Free()

	// Create partitioned data
	totalElements := 0
	for _, kval := range k {
		totalElements += kval
	}

	// Host arrays - flat arrays for simple binding
	u := make([]float64, totalElements)
	v := make([]float64, totalElements)

	// Initialize with partition-aware values
	offset := 0
	for p, kval := range k {
		for i := 0; i < kval; i++ {
			u[offset+i] = float64(p*100 + i)
		}
		offset += kval
	}

	// Phase 1: Define bindings
	err := kp.DefineBindings(
		builder.Input("u").Bind(splitSlice(k, u)),
		builder.Output("v").Bind(splitSlice(k, v)),
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
	kernelName := "partitionAware"
	_, err = kp.ConfigureKernel(kernelName,
		kp.Param("u").CopyTo(),
		kp.Param("v").CopyBack(),
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	// Get signature
	signature, err := kp.GetKernelSignatureForConfig(kernelName)
	if err != nil {
		t.Fatalf("Failed to get signature: %v", err)
	}

	// Build kernel
	kernelSource := fmt.Sprintf(`
@kernel void %s (
	%s
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* u = u_PART(part);
		double* v = v_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				// Double the value
				v[i] = 2.0 * u[i];
			}
		}
	}
}`, kernelName, signature)

	_, err = kp.BuildKernel(kernelSource, kernelName)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Execute kernel
	err = kp.ExecuteKernel(kernelName)
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify results
	offset = 0
	for p, kval := range k {
		for i := 0; i < kval; i++ {
			expected := 2.0 * float64(p*100+i)
			if math.Abs(v[offset+i]-expected) > 1e-10 {
				t.Errorf("Partition %d, element %d: expected %f, got %f",
					p, i, expected, v[offset+i])
			}
		}
		offset += kval
	}
}

// TestParameterAPI_DeviceDataPersistence tests data persistence across kernels
func TestParameterAPI_DeviceDataPersistence(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	n := 10
	kp := NewRunner(device, builder.Config{
		K: []int{n},
	})
	defer kp.Free()

	// Host arrays
	data := make([]float64, n)
	result := make([]float64, n)

	// Initialize
	for i := 0; i < n; i++ {
		data[i] = float64(i)
	}

	// Phase 1: Define bindings
	err := kp.DefineBindings(
		builder.InOut("data").Bind(data), // Can be used as input or output
		builder.Output("result").Bind(result),
		builder.Temp("temp").Type(builder.Float64).Size(n),
	)
	if err != nil {
		t.Fatalf("Failed to define bindings: %v", err)
	}

	// Phase 1: Allocate device memory
	err = kp.AllocateDevice()
	if err != nil {
		t.Fatalf("Failed to allocate device: %v", err)
	}

	// Kernel 1: Copy data to device and square it
	_, err = kp.ConfigureKernel("kernel1",
		kp.Param("data").CopyTo().CopyBack(), // Copy to device, then back after kernel
		kp.Param("temp"),
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel1: %v", err)
	}

	sig1, _ := kp.GetKernelSignatureForConfig("kernel1")
	kernel1Src := fmt.Sprintf(`
@kernel void kernel1(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		double* data = data_PART(part);
		double* temp = temp_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				temp[i] = data[i];
				data[i] = data[i] * data[i];
			}
		}
	}
}`, sig1)

	_, err = kp.BuildKernel(kernel1Src, "kernel1")
	if err != nil {
		t.Fatalf("Failed to build kernel1: %v", err)
	}

	err = kp.ExecuteKernel("kernel1")
	if err != nil {
		t.Fatalf("Failed to execute kernel1: %v", err)
	}

	// Verify data was squared
	for i := 0; i < n; i++ {
		expected := float64(i * i)
		if math.Abs(data[i]-expected) > 1e-10 {
			t.Errorf("After kernel1, element %d: expected %f, got %f",
				i, expected, data[i])
		}
	}

	// Kernel 2: Use device data (already squared) without copying from host
	_, err = kp.ConfigureKernel("kernel2",
		kp.Param("data"),              // No copy - use existing device data
		kp.Param("result").CopyBack(), // Copy result back
		kp.Param("temp"),              // Still has original values
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel2: %v", err)
	}

	sig2, _ := kp.GetKernelSignatureForConfig("kernel2")
	kernel2Src := fmt.Sprintf(`
@kernel void kernel2(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		double* data = data_PART(part);
		double* result = result_PART(part);
		double* temp = temp_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				// data has squared values, temp has original values
				result[i] = data[i] + temp[i];  // i^2 + i
			}
		}
	}
}`, sig2)

	_, err = kp.BuildKernel(kernel2Src, "kernel2")
	if err != nil {
		t.Fatalf("Failed to build kernel2: %v", err)
	}

	err = kp.ExecuteKernel("kernel2")
	if err != nil {
		t.Fatalf("Failed to execute kernel2: %v", err)
	}

	// Verify result = i^2 + i
	for i := 0; i < n; i++ {
		expected := float64(i*i + i)
		if math.Abs(result[i]-expected) > 1e-10 {
			t.Errorf("Final result element %d: expected %f, got %f",
				i, expected, result[i])
		}
	}
}

// TestParameterAPI_ManualCopyOperations tests manual memory operations
func TestParameterAPI_ManualCopyOperations(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	n := 15
	kp := NewRunner(device, builder.Config{
		K: []int{n},
	})
	defer kp.Free()

	// Host arrays
	a := make([]float64, n)
	b := make([]float64, n)
	c := make([]float64, n)

	// Initialize
	for i := 0; i < n; i++ {
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

	// Manual copy operations
	err = kp.CopyToDevice("a")
	if err != nil {
		t.Fatalf("Failed to copy a to device: %v", err)
	}

	err = kp.CopyToDevice("b")
	if err != nil {
		t.Fatalf("Failed to copy b to device: %v", err)
	}

	// Configure kernel without any copy actions
	_, err = kp.ConfigureKernel("add",
		kp.Param("a"), // No copy - already on device
		kp.Param("b"), // No copy - already on device
		kp.Param("c"), // No copy - will copy manually
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
		t.Fatalf("Failed to copy c from device: %v", err)
	}

	// Verify results
	for i := 0; i < n; i++ {
		expected := float64(i) + float64(i*2) // a[i] + b[i]
		if math.Abs(c[i]-expected) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, c[i])
		}
	}
}

// TestParameterAPI_ConfigureCopy tests batch copy operations
func TestParameterAPI_ConfigureCopy(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	n := 12
	kp := NewRunner(device, builder.Config{
		K: []int{n},
	})
	defer kp.Free()

	// Host arrays
	x := make([]float64, n)
	y := make([]float64, n)
	z := make([]float64, n)

	// Initialize
	for i := 0; i < n; i++ {
		x[i] = float64(i)
		y[i] = float64(i * i)
		z[i] = 0.0
	}

	// Phase 1: Define bindings
	err := kp.DefineBindings(
		builder.InOut("x").Bind(x),
		builder.InOut("y").Bind(y),
		builder.InOut("z").Bind(z),
	)
	if err != nil {
		t.Fatalf("Failed to define bindings: %v", err)
	}

	// Phase 1: Allocate device memory
	err = kp.AllocateDevice()
	if err != nil {
		t.Fatalf("Failed to allocate device: %v", err)
	}

	// Configure batch copy to device
	copyToOp, err := kp.ConfigureCopy(
		kp.Param("x").CopyTo(),
		kp.Param("y").CopyTo(),
	)
	if err != nil {
		t.Fatalf("Failed to configure copy to device: %v", err)
	}

	// Execute batch copy
	err = kp.ExecuteCopy(copyToOp)
	if err != nil {
		t.Fatalf("Failed to execute copy to device: %v", err)
	}

	// Run kernel to compute z = x + y
	_, err = kp.ConfigureKernel("compute",
		kp.Param("x"), // Already on device
		kp.Param("y"), // Already on device
		kp.Param("z"), // Will copy back later
	)
	if err != nil {
		t.Fatalf("Failed to configure kernel: %v", err)
	}

	sig, _ := kp.GetKernelSignatureForConfig("compute")
	kernelSrc := fmt.Sprintf(`
@kernel void compute(%s) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const double* x = x_PART(part);
		const double* y = y_PART(part);
		double* z = z_PART(part);
		
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K[part]) {
				z[i] = x[i] + y[i];
			}
		}
	}
}`, sig)

	_, err = kp.BuildKernel(kernelSrc, "compute")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.ExecuteKernel("compute")
	if err != nil {
		t.Fatalf("Failed to execute kernel: %v", err)
	}

	// Configure batch copy from device
	copyFromOp, err := kp.ConfigureCopy(
		kp.Param("z").CopyBack(),
	)
	if err != nil {
		t.Fatalf("Failed to configure copy from device: %v", err)
	}

	// Execute batch copy
	err = kp.ExecuteCopy(copyFromOp)
	if err != nil {
		t.Fatalf("Failed to execute copy from device: %v", err)
	}

	// Verify results
	for i := 0; i < n; i++ {
		expected := float64(i) + float64(i*i) // x[i] + y[i]
		if math.Abs(z[i]-expected) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, expected, z[i])
		}
	}
}

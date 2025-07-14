package runner

import (
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"gonum.org/v1/gonum/mat"
	"math"
	"testing"
	"unsafe"
)

// TestExecuteCopyActions tests the unified copy infrastructure
func TestExecuteCopyActions(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	runner := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer runner.Free()

	// Define bindings
	hostData := make([]float64, 10)
	for i := range hostData {
		hostData[i] = float64(i)
	}

	err := runner.DefineBindings(
		builder.Input("data").Bind(hostData),
	)
	if err != nil {
		t.Fatalf("DefineBindings failed: %v", err)
	}

	// Allocate device memory
	err = runner.AllocateDevice()
	if err != nil {
		t.Fatalf("AllocateDevice failed: %v", err)
	}

	// Test CopyTo action
	t.Run("CopyTo", func(t *testing.T) {
		binding := runner.GetBinding("data")
		actions := []ParameterUsage{
			{Binding: binding, Actions: CopyTo},
		}

		err := runner.executeCopyActions(actions)
		if err != nil {
			t.Fatalf("executeCopyActions failed: %v", err)
		}

		// Verify data is on device by reading it back
		deviceData := make([]float64, 10)
		mem := runner.GetMemory("data")
		mem.CopyTo(unsafe.Pointer(&deviceData[0]), int64(10*8))

		for i := range deviceData {
			if deviceData[i] != float64(i) {
				t.Errorf("Element %d: expected %f, got %f", i, float64(i), deviceData[i])
			}
		}
	})

	// Test CopyBack action
	t.Run("CopyBack", func(t *testing.T) {
		// Modify data on device
		modifiedData := make([]float64, 10)
		for i := range modifiedData {
			modifiedData[i] = float64(i) * 2.0
		}
		mem := runner.GetMemory("data")
		mem.CopyFrom(unsafe.Pointer(&modifiedData[0]), int64(10*8))

		// Clear host data
		for i := range hostData {
			hostData[i] = 0
		}

		// Copy back from device
		binding := runner.GetBinding("data")
		actions := []ParameterUsage{
			{Binding: binding, Actions: CopyBack},
		}

		err := runner.executeCopyActions(actions)
		if err != nil {
			t.Fatalf("executeCopyActions failed: %v", err)
		}

		// Verify host data was updated
		for i := range hostData {
			expected := float64(i) * 2.0
			if hostData[i] != expected {
				t.Errorf("Element %d: expected %f, got %f", i, expected, hostData[i])
			}
		}
	})

	// Test bidirectional copy
	t.Run("BidirectionalCopy", func(t *testing.T) {
		// Reset host data
		for i := range hostData {
			hostData[i] = float64(i) + 0.5
		}

		binding := runner.GetBinding("data")
		actions := []ParameterUsage{
			{Binding: binding, Actions: Copy}, // CopyTo | CopyBack
		}

		err := runner.executeCopyActions(actions)
		if err != nil {
			t.Fatalf("executeCopyActions failed: %v", err)
		}

		// Data should have made round trip unchanged
		for i := range hostData {
			expected := float64(i) + 0.5
			if math.Abs(hostData[i]-expected) > 1e-10 {
				t.Errorf("Element %d: expected %f, got %f", i, expected, hostData[i])
			}
		}
	})
}

// TestSimpleCopyMethods tests the simple copy methods
func TestSimpleCopyMethods(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	runner := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer runner.Free()

	// Define bindings
	hostData := make([]float64, 10)
	for i := range hostData {
		hostData[i] = float64(i)
	}

	runner.DefineBindings(
		builder.Input("data").Bind(hostData),
	)
	runner.AllocateDevice()

	t.Run("CopyToDevice", func(t *testing.T) {
		// Modify host data
		for i := range hostData {
			hostData[i] = float64(i) * 3.0
		}

		// Copy to device
		err := runner.CopyToDevice("data")
		if err != nil {
			t.Fatalf("CopyToDevice failed: %v", err)
		}

		// Verify by reading device memory
		deviceData := make([]float64, 10)
		mem := runner.GetMemory("data")
		mem.CopyTo(unsafe.Pointer(&deviceData[0]), int64(10*8))

		for i := range deviceData {
			expected := float64(i) * 3.0
			if deviceData[i] != expected {
				t.Errorf("Element %d: expected %f, got %f", i, expected, deviceData[i])
			}
		}
	})

	t.Run("CopyFromDevice", func(t *testing.T) {
		// Put test data on device
		testData := make([]float64, 10)
		for i := range testData {
			testData[i] = float64(i) * 4.0
		}
		mem := runner.GetMemory("data")
		mem.CopyFrom(unsafe.Pointer(&testData[0]), int64(10*8))

		// Clear host data
		for i := range hostData {
			hostData[i] = 0
		}

		// Copy from device
		err := runner.CopyFromDevice("data")
		if err != nil {
			t.Fatalf("CopyFromDevice failed: %v", err)
		}

		// Verify host data
		for i := range hostData {
			expected := float64(i) * 4.0
			if hostData[i] != expected {
				t.Errorf("Element %d: expected %f, got %f", i, expected, hostData[i])
			}
		}
	})
}

// TestCopyWithTypeConversion tests type conversion during copy
func TestCopyWithTypeConversion(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	runner := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer runner.Free()

	// Host data is float64, device will be float32
	hostData := make([]float64, 10)
	for i := range hostData {
		hostData[i] = float64(i) * 1.123456789
	}

	runner.DefineBindings(
		builder.Input("data").Bind(hostData).Convert(builder.Float32),
	)
	runner.AllocateDevice()

	t.Run("ConversionToDevice", func(t *testing.T) {
		// Copy to device (float64 → float32)
		err := runner.CopyToDevice("data")
		if err != nil {
			t.Fatalf("CopyToDevice failed: %v", err)
		}

		// Read as float32 from device
		deviceData32 := make([]float32, 10)
		mem := runner.GetMemory("data")
		mem.CopyTo(unsafe.Pointer(&deviceData32[0]), int64(10*4))

		// Verify conversion happened
		for i := range deviceData32 {
			expected := float32(float64(i) * 1.123456789)
			if math.Abs(float64(deviceData32[i]-expected)) > 1e-6 {
				t.Errorf("Element %d: expected %f, got %f", i, expected, deviceData32[i])
			}
		}
	})

	t.Run("ConversionFromDevice", func(t *testing.T) {
		// Clear host data
		for i := range hostData {
			hostData[i] = 0
		}

		// Copy from device (float32 → float64)
		err := runner.CopyFromDevice("data")
		if err != nil {
			t.Fatalf("CopyFromDevice failed: %v", err)
		}

		// Verify data (with precision loss from float32)
		for i := range hostData {
			expected := float32(float64(i) * 1.123456789) // Precision loss
			if math.Abs(hostData[i]-float64(expected)) > 1e-6 {
				t.Errorf("Element %d: expected %f, got %f", i, expected, hostData[i])
			}
		}
	})
}

// TestCopyMatrixData tests matrix copy operations
func TestCopyMatrixData(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	runner := NewRunner(device, builder.Config{
		K: []int{5},
	})
	defer runner.Free()

	// Create a test matrix
	matData := []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	hostMatrix := mat.NewDense(4, 4, matData)

	runner.DefineBindings(
		builder.Input("matrix").Bind(hostMatrix).ToMatrix(),
	)
	runner.AllocateDevice()

	t.Run("MatrixCopyRoundTrip", func(t *testing.T) {
		// Copy to device
		err := runner.CopyToDevice("matrix")
		if err != nil {
			t.Fatalf("CopyToDevice failed: %v", err)
		}

		// Clear host matrix
		for i := 0; i < 4; i++ {
			for j := 0; j < 4; j++ {
				hostMatrix.Set(i, j, 0)
			}
		}

		// Copy back from device
		err = runner.CopyFromDevice("matrix")
		if err != nil {
			t.Fatalf("CopyFromDevice failed: %v", err)
		}

		// Verify matrix data
		for i := 0; i < 4; i++ {
			for j := 0; j < 4; j++ {
				expected := matData[i*4+j]
				actual := hostMatrix.At(i, j)
				if math.Abs(actual-expected) > 1e-10 {
					t.Errorf("Matrix[%d,%d]: expected %f, got %f", i, j, expected, actual)
				}
			}
		}
	})
}

// TestCopyPartitionedData tests partitioned data copy
func TestCopyPartitionedData(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	runner := NewRunner(device, builder.Config{
		K: []int{10, 15, 20},
	})
	defer runner.Free()

	// Create partitioned data
	hostData := [][]float64{
		make([]float64, 10),
		make([]float64, 15),
		make([]float64, 20),
	}

	// Initialize with test data
	for p := 0; p < 3; p++ {
		for i := range hostData[p] {
			hostData[p][i] = float64(p*100 + i)
		}
	}

	runner.DefineBindings(
		builder.Input("pdata").Bind(hostData),
	)
	runner.AllocateDevice()

	t.Run("PartitionedCopyRoundTrip", func(t *testing.T) {
		// Copy to device
		err := runner.CopyToDevice("pdata")
		if err != nil {
			t.Fatalf("CopyToDevice failed: %v", err)
		}

		// Clear host data
		for p := range hostData {
			for i := range hostData[p] {
				hostData[p][i] = 0
			}
		}

		// Copy back from device
		err = runner.CopyFromDevice("pdata")
		if err != nil {
			t.Fatalf("CopyFromDevice failed: %v", err)
		}

		// Verify data
		for p := 0; p < 3; p++ {
			for i := range hostData[p] {
				expected := float64(p*100 + i)
				if hostData[p][i] != expected {
					t.Errorf("Partition %d, element %d: expected %f, got %f",
						p, i, expected, hostData[p][i])
				}
			}
		}
	})
}

// TestCopyErrors tests error conditions
func TestCopyErrors(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	runner := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer runner.Free()

	t.Run("NonexistentBinding", func(t *testing.T) {
		err := runner.CopyToDevice("nonexistent")
		if err == nil {
			t.Error("Expected error for nonexistent binding")
		}
	})

	t.Run("NoAllocation", func(t *testing.T) {
		// Define binding but don't allocate
		hostData := make([]float64, 10)
		runner.DefineBindings(
			builder.Input("data").Bind(hostData),
		)
		// Don't call AllocateDevice()

		err := runner.CopyToDevice("data")
		if err == nil {
			t.Error("Expected error for unallocated memory")
		}
	})
}

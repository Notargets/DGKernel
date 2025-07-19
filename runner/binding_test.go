package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"gonum.org/v1/gonum/mat"
	"testing"
)

// TestDefineBindings_Basic tests the basic DefineBindings functionality
func TestDefineBindings_Basic(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	runner := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer runner.Free()

	// Test data
	hostArray := make([]float64, 10)
	hostMatrix := mat.NewDense(4, 4, nil)
	alpha := 2.5

	// Define bindings
	err := runner.DefineBindings(
		builder.Input("array").Bind(hostArray),
		builder.Input("matrix").Bind(hostMatrix).ToMatrix(),
		builder.Scalar("alpha").Bind(alpha),
		builder.Temp("work").Type(builder.Float64).Size(10),
	)
	if err != nil {
		t.Fatalf("DefineBindings failed: %v", err)
	}

	// Verify bindings were created
	tests := []struct {
		name        string
		shouldExist bool
		checks      func(*DeviceBinding) error
	}{
		{
			name:        "array",
			shouldExist: true,
			checks: func(b *DeviceBinding) error {
				if b.Size != 10 {
					return fmt.Errorf("expected size 10, got %d", b.Size)
				}
				if b.IsMatrix {
					return fmt.Errorf("array should not be matrix")
				}
				if b.HostType != builder.Float64 {
					return fmt.Errorf("expected Float64 host type")
				}
				return nil
			},
		},
		{
			name:        "matrix",
			shouldExist: true,
			checks: func(b *DeviceBinding) error {
				if !b.IsMatrix {
					return fmt.Errorf("should be matrix")
				}
				if b.MatrixRows != 4 || b.MatrixCols != 4 {
					return fmt.Errorf("expected 4x4 matrix")
				}
				return nil
			},
		},
		{
			name:        "alpha",
			shouldExist: true,
			checks: func(b *DeviceBinding) error {
				if !b.IsScalar {
					return fmt.Errorf("should be scalar")
				}
				if b.Size != 1 {
					return fmt.Errorf("scalar should have size 1")
				}
				return nil
			},
		},
		{
			name:        "work",
			shouldExist: true,
			checks: func(b *DeviceBinding) error {
				if !b.IsTemp {
					return fmt.Errorf("should be temp array")
				}
				if b.Size != 10 {
					return fmt.Errorf("expected size 10")
				}
				if b.HostBinding != nil {
					return fmt.Errorf("temp array should not have host binding")
				}
				return nil
			},
		},
	}

	for _, tt := range tests {
		binding := runner.GetBinding(tt.name)
		if tt.shouldExist && binding == nil {
			t.Errorf("Binding %s not found", tt.name)
			continue
		}
		if !tt.shouldExist && binding != nil {
			t.Errorf("Binding %s should not exist", tt.name)
			continue
		}
		if tt.checks != nil && binding != nil {
			if err := tt.checks(binding); err != nil {
				t.Errorf("Binding %s: %v", tt.name, err)
			}
		}
	}
}

// TestDefineBindings_TypeConversion tests type conversion in bindings
func TestDefineBindings_TypeConversion(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	runner := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer runner.Free()

	hostData := make([]float64, 10)

	// Define binding with type conversion
	err := runner.DefineBindings(
		builder.Input("data").Bind(hostData).DeviceMemType(builder.Float32),
	)
	if err != nil {
		t.Fatalf("DefineBindings failed: %v", err)
	}

	binding := runner.GetBinding("data")
	if binding == nil {
		t.Fatal("Binding not found")
	}

	// Check types
	if binding.HostType != builder.Float64 {
		t.Errorf("Expected host type Float64, got %v", binding.HostType)
	}
	if binding.DeviceType != builder.Float32 {
		t.Errorf("Expected device type Float32, got %v", binding.DeviceType)
	}
	if binding.ElementSize != 4 {
		t.Errorf("Expected element size 4 (float32), got %d", binding.ElementSize)
	}
}

// TestDefineBindings_Partitioned tests partitioned data binding
func TestDefineBindings_Partitioned(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	runner := NewRunner(device, builder.Config{
		K: []int{10, 15, 20},
	})
	defer runner.Free()

	// Create partitioned data
	partitionedArray := [][]float64{
		make([]float64, 10),
		make([]float64, 15),
		make([]float64, 20),
	}

	err := runner.DefineBindings(
		builder.Input("parray").Bind(partitionedArray),
	)
	if err != nil {
		t.Fatalf("DefineBindings failed: %v", err)
	}

	binding := runner.GetBinding("parray")
	if binding == nil {
		t.Fatal("Binding not found")
	}

	// Verify partitioned data properties
	if !binding.IsPartitioned {
		t.Error("Should be partitioned")
	}
	if binding.PartitionCount != 3 {
		t.Errorf("Expected 3 partitions, got %d", binding.PartitionCount)
	}
	if binding.Size != 45 {
		t.Errorf("Expected total size 45, got %d", binding.Size)
	}

	// Check partition sizes
	expectedSizes := []int{10, 15, 20}
	for i, expected := range expectedSizes {
		if binding.PartitionSizes[i] != expected {
			t.Errorf("Partition %d: expected size %d, got %d",
				i, expected, binding.PartitionSizes[i])
		}
	}
}

// TestAllocateDevice_Basic tests basic device allocation
func TestAllocateDevice_Basic(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	runner := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer runner.Free()

	// Define bindings
	hostData := make([]float64, 10)
	err := runner.DefineBindings(
		builder.Input("data").Bind(hostData),
		builder.Temp("work").Type(builder.Float32).Size(10),
	)
	if err != nil {
		t.Fatalf("DefineBindings failed: %v", err)
	}

	// Allocate device memory
	err = runner.AllocateDevice()
	if err != nil {
		t.Fatalf("AllocateDevice failed: %v", err)
	}

	// Verify allocation state
	if !runner.IsAllocated {
		t.Error("IsAllocated should be true")
	}

	// Verify memory was allocated
	if runner.GetMemory("data") == nil {
		t.Error("Memory for 'data' not allocated")
	}
	if runner.GetMemory("work") == nil {
		t.Error("Memory for 'work' not allocated")
	}

	// Verify offsets were allocated
	if runner.GetOffsets("data") == nil {
		t.Error("Offsets for 'data' not allocated")
	}

	// Verify metadata
	meta, exists := runner.GetArrayMetadata("data")
	if !exists {
		t.Error("Metadata for 'data' not found")
	} else {
		if meta.dataType != builder.Float64 {
			t.Errorf("Expected Float64 type, got %v", meta.dataType)
		}
	}

	// Try to allocate again - should fail
	err = runner.AllocateDevice()
	if err == nil {
		t.Error("Second AllocateDevice should fail")
	}
}

// TestAllocateDevice_WithMatrices tests allocation with matrices
func TestAllocateDevice_WithMatrices(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	runner := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer runner.Free()

	// Create matrices
	staticMatrix := mat.NewDense(4, 4, nil)
	deviceMatrix := mat.NewDense(4, 4, nil)

	// Define bindings
	err := runner.DefineBindings(
		builder.Input("static").Bind(staticMatrix).ToMatrix().Static(),
		builder.Input("device").Bind(deviceMatrix).ToMatrix(),
	)
	if err != nil {
		t.Fatalf("DefineBindings failed: %v", err)
	}

	// Allocate
	err = runner.AllocateDevice()
	if err != nil {
		t.Fatalf("AllocateDevice failed: %v", err)
	}

	// Verify static matrix was added
	if _, exists := runner.StaticMatrices["static"]; !exists {
		t.Error("Static matrix not added")
	}

	// Verify device matrix was added
	if _, exists := runner.DeviceMatrices["device"]; !exists {
		t.Error("Device matrix not added")
	}

	// Device matrices should have allocated memory (stored without _global suffix)
	if _, exists := runner.PooledMemory["device"]; !exists {
		t.Error("Device matrix memory not allocated")
	}
}

// TestDefineBindings_Errors tests error conditions
func TestDefineBindings_Errors(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	t.Run("AfterAllocate", func(t *testing.T) {
		runner := NewRunner(device, builder.Config{K: []int{10}})
		defer runner.Free()

		// Define and allocate
		runner.DefineBindings(builder.Temp("x").Size(10).Type(builder.Float64))
		runner.AllocateDevice()

		// Try to define more bindings - should fail
		err := runner.DefineBindings(builder.Temp("y").Size(10).Type(builder.Float64))
		if err == nil {
			t.Error("DefineBindings after AllocateDevice should fail")
		}
	})

	t.Run("PartitionMismatch", func(t *testing.T) {
		runner := NewRunner(device, builder.Config{K: []int{10, 20}})
		defer runner.Free()

		// Non-partitioned data for partitioned kernel
		nonPartitioned := make([]float64, 30)
		err := runner.DefineBindings(
			builder.Input("data").Bind(nonPartitioned),
		)
		if err == nil {
			t.Error("Non-partitioned data for partitioned kernel should fail")
		}
	})

	t.Run("WrongPartitionCount", func(t *testing.T) {
		runner := NewRunner(device, builder.Config{K: []int{10, 20, 30}})
		defer runner.Free()

		// Wrong number of partitions
		wrongPartitions := [][]float64{
			make([]float64, 10),
			make([]float64, 20),
			// Missing third partition
		}
		err := runner.DefineBindings(
			builder.Input("data").Bind(wrongPartitions),
		)
		if err == nil {
			t.Error("Wrong partition count should fail")
		}
	})
}

// TestDefineBindings_PreservesExistingBehavior tests that the new API
// doesn't break existing allocation behavior
func TestDefineBindings_PreservesExistingBehavior(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	runner := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer runner.Free()

	// Use new API
	hostData := make([]float64, 10)
	runner.DefineBindings(
		builder.Input("data").Bind(hostData),
	)
	runner.AllocateDevice()

	// The allocated arrays should work with existing kernel infrastructure
	arrays := runner.GetAllocatedArrays()
	if len(arrays) != 1 || arrays[0] != "data" {
		t.Errorf("Expected allocated arrays [data], got %v", arrays)
	}

	// Memory should be accessible
	mem := runner.GetMemory("data")
	if mem == nil {
		t.Error("Memory not accessible through GetMemory")
	}

	// Offsets should be correct (offsets are in VALUES, not bytes)
	// For float64, 10 values = 80 bytes
	offsets, err := runner.readPartitionOffsets("data")
	if err != nil {
		t.Errorf("Failed to read offsets: %v", err)
	}
	// Expecting offsets in values: [0, 10]
	if len(offsets) != 2 || offsets[0] != 0 || offsets[1] != 10 {
		t.Errorf("Unexpected offsets: %v (expected [0, 10] values)", offsets)
	}
}

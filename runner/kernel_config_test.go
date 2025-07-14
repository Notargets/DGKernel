package runner

import (
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/DGKernel/utils"
	"testing"
	"unsafe"
)

// TestConfigureKernel tests kernel configuration
func TestConfigureKernel(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	runner := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer runner.Free()

	// Setup bindings
	hostU := make([]float64, 10)
	hostV := make([]float64, 10)
	hostRHS := make([]float64, 10)

	err := runner.DefineBindings(
		builder.Input("U").Bind(hostU),
		builder.Input("V").Bind(hostV),
		builder.Output("RHS").Bind(hostRHS),
		builder.Temp("work").Type(builder.Float64).Size(10),
	)
	if err != nil {
		t.Fatalf("DefineBindings failed: %v", err)
	}

	err = runner.AllocateDevice()
	if err != nil {
		t.Fatalf("AllocateDevice failed: %v", err)
	}

	// Test kernel configuration
	t.Run("BasicConfiguration", func(t *testing.T) {
		config, err := runner.ConfigureKernel("mykernel",
			runner.Param("U").CopyTo(),
			runner.Param("V").CopyTo(),
			runner.Param("RHS").CopyBack(),
			runner.Param("work"), // No copy for temp array
		)
		if err != nil {
			t.Fatalf("ConfigureKernel failed: %v", err)
		}

		// Verify configuration
		if config.Name != "mykernel" {
			t.Errorf("Expected kernel name 'mykernel', got '%s'", config.Name)
		}

		// Check parameter configurations
		tests := []struct {
			name           string
			expectCopyTo   bool
			expectCopyBack bool
		}{
			{"U", true, false},
			{"V", true, false},
			{"RHS", false, true},
			{"work", false, false},
		}

		for _, tt := range tests {
			param := config.GetParameter(tt.name)
			if param == nil {
				t.Errorf("Parameter %s not found in configuration", tt.name)
				continue
			}

			if param.HasAction(CopyTo) != tt.expectCopyTo {
				t.Errorf("Parameter %s: expected CopyTo=%v", tt.name, tt.expectCopyTo)
			}
			if param.HasAction(CopyBack) != tt.expectCopyBack {
				t.Errorf("Parameter %s: expected CopyBack=%v", tt.name, tt.expectCopyBack)
			}
		}
	})

	// Test configuration storage
	t.Run("ConfigurationStorage", func(t *testing.T) {
		// Configure kernel
		_, err := runner.ConfigureKernel("stored",
			runner.Param("U").Copy(),
		)
		if err != nil {
			t.Fatalf("ConfigureKernel failed: %v", err)
		}

		// Verify it was stored
		if runner.KernelConfigs["stored"] == nil {
			t.Error("Kernel configuration not stored")
		}
	})

	// Test parameter not found
	t.Run("NonexistentParameter", func(t *testing.T) {
		config, err := runner.ConfigureKernel("badkernel",
			runner.Param("nonexistent").CopyTo(),
		)

		// Should create config but with nil binding
		if err == nil {
			t.Error("Expected error for nonexistent parameter")
		}
		if config != nil {
			t.Error("Should not return config for invalid parameter")
		}
	})

	// Test before allocation
	t.Run("BeforeAllocation", func(t *testing.T) {
		runner2 := NewRunner(device, builder.Config{K: []int{10}})
		defer runner2.Free()

		// Define bindings but don't allocate
		runner2.DefineBindings(
			builder.Input("data").Bind(make([]float64, 10)),
		)

		// Try to configure kernel - should fail
		_, err := runner2.ConfigureKernel("test",
			runner2.Param("data").CopyTo(),
		)
		if err == nil {
			t.Error("ConfigureKernel should fail before AllocateDevice")
		}
	})
}

// TestConfigureCopy tests copy configuration
func TestConfigureCopy(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	runner := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer runner.Free()

	// Setup
	hostU := make([]float64, 10)
	hostV := make([]float64, 10)
	for i := range hostU {
		hostU[i] = float64(i)
		hostV[i] = float64(i) * 2
	}

	runner.DefineBindings(
		builder.Input("U").Bind(hostU),
		builder.Input("V").Bind(hostV),
	)
	runner.AllocateDevice()

	t.Run("BasicCopyConfig", func(t *testing.T) {
		// Configure a copy operation
		copyConfig, err := runner.ConfigureCopy(
			runner.Param("U").CopyTo(),
			runner.Param("V").CopyTo(),
		)
		if err != nil {
			t.Fatalf("ConfigureCopy failed: %v", err)
		}

		// Execute the copy
		err = runner.ExecuteCopy(copyConfig)
		if err != nil {
			t.Fatalf("ExecuteCopy failed: %v", err)
		}

		// Verify data is on device by reading back
		verifyU := make([]float64, 10)
		verifyV := make([]float64, 10)

		memU := runner.GetMemory("U")
		memV := runner.GetMemory("V")

		memU.CopyTo(unsafe.Pointer(&verifyU[0]), int64(10*8))
		memV.CopyTo(unsafe.Pointer(&verifyV[0]), int64(10*8))

		for i := 0; i < 10; i++ {
			if verifyU[i] != float64(i) {
				t.Errorf("U[%d]: expected %f, got %f", i, float64(i), verifyU[i])
			}
			if verifyV[i] != float64(i)*2 {
				t.Errorf("V[%d]: expected %f, got %f", i, float64(i)*2, verifyV[i])
			}
		}
	})

	t.Run("CopyBackConfig", func(t *testing.T) {
		// Modify data on device
		modU := make([]float64, 10)
		modV := make([]float64, 10)
		for i := range modU {
			modU[i] = float64(i) * 10
			modV[i] = float64(i) * 20
		}

		memU := runner.GetMemory("U")
		memV := runner.GetMemory("V")
		memU.CopyFrom(unsafe.Pointer(&modU[0]), int64(10*8))
		memV.CopyFrom(unsafe.Pointer(&modV[0]), int64(10*8))

		// Clear host arrays
		for i := range hostU {
			hostU[i] = 0
			hostV[i] = 0
		}

		// Configure and execute copy back
		copyConfig, err := runner.ConfigureCopy(
			runner.Param("U").CopyBack(),
			runner.Param("V").CopyBack(),
		)
		if err != nil {
			t.Fatalf("ConfigureCopy failed: %v", err)
		}

		err = runner.ExecuteCopy(copyConfig)
		if err != nil {
			t.Fatalf("ExecuteCopy failed: %v", err)
		}

		// Verify host data was updated
		for i := 0; i < 10; i++ {
			if hostU[i] != float64(i)*10 {
				t.Errorf("U[%d]: expected %f, got %f", i, float64(i)*10, hostU[i])
			}
			if hostV[i] != float64(i)*20 {
				t.Errorf("V[%d]: expected %f, got %f", i, float64(i)*20, hostV[i])
			}
		}
	})
}

// TestParamConfig tests parameter configuration builder
func TestParamConfig(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	runner := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer runner.Free()

	runner.DefineBindings(
		builder.Input("data").Bind(make([]float64, 10)),
	)
	runner.AllocateDevice()

	t.Run("ActionFlags", func(t *testing.T) {
		// Test CopyTo
		param := runner.Param("data").CopyTo()
		if param.actions != CopyTo {
			t.Errorf("Expected CopyTo action, got %v", param.actions)
		}

		// Test CopyBack
		param = runner.Param("data").CopyBack()
		if param.actions != CopyBack {
			t.Errorf("Expected CopyBack action, got %v", param.actions)
		}

		// Test Copy (bidirectional)
		param = runner.Param("data").Copy()
		if param.actions != Copy {
			t.Errorf("Expected Copy action, got %v", param.actions)
		}

		// Test NoCopy
		param = runner.Param("data").Copy().NoCopy()
		if param.actions != NoAction {
			t.Errorf("Expected NoAction, got %v", param.actions)
		}
	})

	t.Run("ChainedActions", func(t *testing.T) {
		// Test combining actions
		param := runner.Param("data").CopyTo().CopyBack()
		if param.actions != (CopyTo | CopyBack) {
			t.Errorf("Expected CopyTo|CopyBack, got %v", param.actions)
		}
	})

	t.Run("NonexistentParam", func(t *testing.T) {
		param := runner.Param("nonexistent")
		if param.binding != nil {
			t.Error("Expected nil binding for nonexistent parameter")
		}
	})
}

// TestMultipleKernelConfigs tests multiple kernel configurations
func TestMultipleKernelConfigs(t *testing.T) {
	device := utils.CreateTestDevice()
	defer device.Free()

	runner := NewRunner(device, builder.Config{
		K: []int{10},
	})
	defer runner.Free()

	// Setup bindings
	runner.DefineBindings(
		builder.Input("U").Bind(make([]float64, 10)),
		builder.Input("V").Bind(make([]float64, 10)),
		builder.Output("R1").Bind(make([]float64, 10)),
		builder.Output("R2").Bind(make([]float64, 10)),
	)
	runner.AllocateDevice()

	// Configure multiple kernels with different parameter usage
	t.Run("IndependentConfigs", func(t *testing.T) {
		// Kernel 1: Uses U and R1
		config1, err := runner.ConfigureKernel("kernel1",
			runner.Param("U").CopyTo(),
			runner.Param("R1").CopyBack(),
		)
		if err != nil {
			t.Fatalf("ConfigureKernel failed: %v", err)
		}

		// Kernel 2: Uses V and R2
		config2, err := runner.ConfigureKernel("kernel2",
			runner.Param("V").CopyTo(),
			runner.Param("R2").CopyBack(),
		)
		if err != nil {
			t.Fatalf("ConfigureKernel failed: %v", err)
		}

		// Verify configurations are independent
		if config1.HasParameter("V") {
			t.Error("kernel1 should not have parameter V")
		}
		if config2.HasParameter("U") {
			t.Error("kernel2 should not have parameter U")
		}

		// Verify both are stored
		if runner.KernelConfigs["kernel1"] != config1 {
			t.Error("kernel1 configuration not stored correctly")
		}
		if runner.KernelConfigs["kernel2"] != config2 {
			t.Error("kernel2 configuration not stored correctly")
		}
	})

	// Test reusing device data between kernels
	t.Run("ReusingDeviceData", func(t *testing.T) {
		// Kernel that modifies data in place
		_, err := runner.ConfigureKernel("inplace",
			runner.Param("U"), // No copy - data already on device
			runner.Param("V"), // No copy - data already on device
		)
		if err != nil {
			t.Fatalf("ConfigureKernel failed: %v", err)
		}

		config := runner.KernelConfigs["inplace"]

		// Verify no copy actions
		uParam := config.GetParameter("U")
		vParam := config.GetParameter("V")

		if uParam.Actions != NoAction {
			t.Error("Parameter U should have no copy actions")
		}
		if vParam.Actions != NoAction {
			t.Error("Parameter V should have no copy actions")
		}
	})
}

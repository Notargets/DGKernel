// File: runner/kernel_execution.go
// Phase 6: Execution Refactor - Execute kernels using configurations

package runner

import (
	"fmt"
	"sort"
)

// ExecuteKernel executes a kernel using its configuration
func (kr *Runner) ExecuteKernel(name string, scalarValues ...interface{}) error {
	// Get kernel configuration
	config, exists := kr.KernelConfigs[name]
	if !exists {
		return fmt.Errorf("kernel %s not configured - use ConfigureKernel first", name)
	}

	// Get compiled kernel
	kernel, exists := kr.Kernels[name]
	if !exists {
		return fmt.Errorf("kernel %s not compiled - use BuildKernel first", name)
	}

	// Validate offsets before kernel execution
	for _, arrayName := range kr.GetAllocatedArrays() {
		// Skip validation if this is a device matrix (stored without _global suffix)
		if _, isDeviceMatrix := kr.DeviceMatrices[arrayName]; isDeviceMatrix {
			continue
		}
		if err := kr.validateOffsets(arrayName, "before kernel "+name); err != nil {
			fmt.Printf("Pre-kernel validation failed: %v\n", err)
		}
	}

	// Perform pre-kernel memory operations (CopyTo only)
	preCopyParams := make([]ParameterUsage, 0)
	for _, param := range config.Parameters {
		if param.HasAction(CopyTo) {
			preCopyParams = append(preCopyParams, ParameterUsage{
				Binding: param.Binding,
				Actions: CopyTo,
			})
		}
	}
	if err := kr.executeCopyActions(preCopyParams); err != nil {
		return fmt.Errorf("pre-kernel copy failed: %w", err)
	}

	// Build kernel arguments
	args, err := kr.buildKernelArgumentsFromConfig(config, scalarValues)
	if err != nil {
		return fmt.Errorf("failed to build arguments: %w", err)
	}

	// Execute kernel
	if err := kernel.RunWithArgs(args...); err != nil {
		return fmt.Errorf("kernel execution failed: %w", err)
	}

	kr.Device.Finish()

	// Validate offsets after kernel execution
	for _, arrayName := range kr.GetAllocatedArrays() {
		// Skip validation if this is a device matrix (stored without _global suffix)
		if _, isDeviceMatrix := kr.DeviceMatrices[arrayName]; isDeviceMatrix {
			continue
		}
		if err := kr.validateOffsets(arrayName, "after kernel "+name); err != nil {
			fmt.Printf("Post-kernel validation failed: %v\n", err)
		}
	}

	// Perform post-kernel memory operations (CopyBack only)
	postCopyParams := make([]ParameterUsage, 0)
	for _, param := range config.Parameters {
		if param.HasAction(CopyBack) {
			postCopyParams = append(postCopyParams, ParameterUsage{
				Binding: param.Binding,
				Actions: CopyBack,
			})
		}
	}
	if err := kr.executeCopyActions(postCopyParams); err != nil {
		return fmt.Errorf("post-kernel copy failed: %w", err)
	}

	return nil
}

// buildKernelArgumentsFromConfig builds kernel arguments using KernelConfig
func (kr *Runner) buildKernelArgumentsFromConfig(config *KernelConfig, scalarValues []interface{}) ([]interface{}, error) {
	// Get ordered argument list
	kernelArgs := kr.GetKernelArgumentsForConfig(config)
	args := make([]interface{}, 0, len(kernelArgs))

	// Build arguments in order
	for _, karg := range kernelArgs {
		switch karg.Category {
		case "system":
			mem, exists := kr.PooledMemory[karg.MemoryKey]
			if !exists {
				return nil, fmt.Errorf("system memory %s not found", karg.MemoryKey)
			}
			args = append(args, mem)

		case "matrix", "array_data", "array_offset":
			mem, exists := kr.PooledMemory[karg.MemoryKey]
			if !exists {
				return nil, fmt.Errorf("memory for %s not found", karg.MemoryKey)
			}
			args = append(args, mem)

		case "scalar":
			// Find the scalar binding
			var found bool
			binding := kr.GetBinding(karg.Name)
			if binding != nil && binding.IsScalar && binding.HostBinding != nil {
				args = append(args, binding.HostBinding)
				found = true
			}

			// If not found in bindings, check passed values
			if !found {
				scalarIdx := 0
				for _, usage := range config.Parameters {
					if usage.Binding.IsScalar && usage.Binding.Name == karg.Name {
						if scalarIdx < len(scalarValues) {
							args = append(args, scalarValues[scalarIdx])
							found = true
						}
						break
					}
					if usage.Binding.IsScalar {
						scalarIdx++
					}
				}
			}

			if !found {
				return nil, fmt.Errorf("scalar %s not provided", karg.Name)
			}
		}
	}

	return args, nil
}

// GetKernelArgumentsForConfig returns kernel arguments based on configuration
func (kr *Runner) GetKernelArgumentsForConfig(config *KernelConfig) []KernelArgument {
	var args []KernelArgument

	// 1. K array is always first
	args = append(args, KernelArgument{
		Name:      "K",
		Type:      "int_t*",
		MemoryKey: "K",
		IsConst:   true,
		Category:  "system",
	})

	// 2. Device matrices in sorted order
	var deviceMatrixBindings []*DeviceBinding
	for _, usage := range config.Parameters {
		if usage.Binding.IsMatrix && !usage.Binding.IsStatic {
			deviceMatrixBindings = append(deviceMatrixBindings, usage.Binding)
		}
	}
	// Sort by name for consistent ordering
	sort.Slice(deviceMatrixBindings, func(i, j int) bool {
		return deviceMatrixBindings[i].Name < deviceMatrixBindings[j].Name
	})

	for _, binding := range deviceMatrixBindings {
		args = append(args, KernelArgument{
			Name:      binding.Name,
			Type:      "real_t*",
			MemoryKey: binding.Name,
			IsConst:   true,
			Category:  "matrix",
		})
	}

	// 3. Arrays (non-matrix, non-scalar parameters)
	for _, usage := range config.Parameters {
		binding := usage.Binding
		if binding.IsScalar || binding.IsMatrix {
			continue // Skip scalars and matrices
		}

		// Global data pointer
		args = append(args, KernelArgument{
			Name:        binding.Name + "_global",
			Type:        "real_t*",
			MemoryKey:   binding.Name + "_global",
			IsConst:     !binding.IsOutput,
			Category:    "array_data",
			UserArgName: binding.Name,
		})

		// Offset array
		args = append(args, KernelArgument{
			Name:        binding.Name + "_offsets",
			Type:        "int_t*",
			MemoryKey:   binding.Name + "_offsets",
			IsConst:     true,
			Category:    "array_offset",
			UserArgName: binding.Name,
		})
	}

	// 4. Scalars last
	for _, usage := range config.Parameters {
		if usage.Binding.IsScalar {
			typeStr := GetScalarTypeName(usage.Binding.DeviceType)
			args = append(args, KernelArgument{
				Name:      usage.Binding.Name,
				Type:      typeStr,
				MemoryKey: "", // Scalars don't have memory keys
				IsConst:   true,
				Category:  "scalar",
			})
		}
	}

	return args
}

// GetKernelSignatureForConfig generates kernel signature for a named kernel configuration
func (kr *Runner) GetKernelSignatureForConfig(kernelName string) (string, error) {
	config, exists := kr.KernelConfigs[kernelName]
	if !exists {
		return "", fmt.Errorf("kernel %s not configured", kernelName)
	}

	return config.GetSignature(kr)
}

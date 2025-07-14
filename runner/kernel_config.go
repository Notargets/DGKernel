// File: runner/kernel_config.go
// Phase 1: Foundation - Define KernelConfig and CopyConfig structures
// Phase 5: Configuration System - Implement kernel and copy configuration

package runner

import (
	"fmt"
	"strings"
)

// KernelConfig represents the configuration for a specific kernel execution
// It references bindings and specifies which memory operations to perform
type KernelConfig struct {
	Name       string
	Parameters []ParameterUsage
}

// CopyConfig represents a standalone memory copy operation configuration
// Used for manual memory operations without kernel execution
type CopyConfig struct {
	Parameters []ParameterUsage
}

// GetParameter finds a parameter usage by name
func (kc *KernelConfig) GetParameter(name string) *ParameterUsage {
	for i := range kc.Parameters {
		if kc.Parameters[i].Binding.Name == name {
			return &kc.Parameters[i]
		}
	}
	return nil
}

// HasParameter checks if a parameter is configured
func (kc *KernelConfig) HasParameter(name string) bool {
	return kc.GetParameter(name) != nil
}

// GetParameter finds a parameter usage by name in CopyConfig
func (cc *CopyConfig) GetParameter(name string) *ParameterUsage {
	for i := range cc.Parameters {
		if cc.Parameters[i].Binding.Name == name {
			return &cc.Parameters[i]
		}
	}
	return nil
}

// ConfigureKernel creates a kernel-specific parameter configuration
func (kr *Runner) ConfigureKernel(name string, params ...*ParamConfig) (*KernelConfig, error) {
	if !kr.IsAllocated {
		return nil, fmt.Errorf("device memory not allocated - call AllocateDevice first")
	}

	config := &KernelConfig{
		Name:       name,
		Parameters: make([]ParameterUsage, 0, len(params)),
	}

	// Process each parameter configuration
	for _, param := range params {
		if param == nil {
			continue
		}

		// Ensure the binding exists
		if param.binding == nil {
			return nil, fmt.Errorf("parameter configuration has no binding")
		}

		// Create parameter usage with configured actions
		usage := ParameterUsage{
			Binding: param.binding,
			Actions: param.actions,
		}

		config.Parameters = append(config.Parameters, usage)
	}

	// Store the configuration
	if kr.KernelConfigs == nil {
		kr.KernelConfigs = make(map[string]*KernelConfig)
	}
	kr.KernelConfigs[name] = config

	return config, nil
}

// ConfigureCopy creates a configuration for standalone memory operations
func (kr *Runner) ConfigureCopy(params ...*ParamConfig) (*CopyConfig, error) {
	if !kr.IsAllocated {
		return nil, fmt.Errorf("device memory not allocated - call AllocateDevice first")
	}

	config := &CopyConfig{
		Parameters: make([]ParameterUsage, 0, len(params)),
	}

	// Process each parameter configuration
	for _, param := range params {
		if param == nil {
			continue
		}

		// Ensure the binding exists
		if param.binding == nil {
			return nil, fmt.Errorf("parameter configuration has no binding")
		}

		// Create parameter usage with configured actions
		usage := ParameterUsage{
			Binding: param.binding,
			Actions: param.actions,
		}

		config.Parameters = append(config.Parameters, usage)
	}

	return config, nil
}

// ExecuteCopy executes a copy configuration
func (kr *Runner) ExecuteCopy(config *CopyConfig) error {
	if config == nil {
		return fmt.Errorf("copy configuration is nil")
	}

	return kr.executeCopyActions(config.Parameters)
}

// Param creates a parameter configuration for a named binding
func (kr *Runner) Param(name string) *ParamConfig {
	binding := kr.GetBinding(name)
	if binding == nil {
		// Return a config that will cause an error when used
		return &ParamConfig{
			runner:  kr,
			name:    name,
			binding: nil,
		}
	}

	return &ParamConfig{
		runner:  kr,
		name:    name,
		binding: binding,
		actions: NoAction, // Start with no actions
	}
}

// ParamConfig is a lightweight builder for configuring parameter actions
// Used during kernel/copy configuration phase
type ParamConfig struct {
	runner  *Runner
	name    string
	binding *DeviceBinding
	actions ActionFlags
}

// CopyTo sets the parameter to copy from host to device
func (pc *ParamConfig) CopyTo() *ParamConfig {
	pc.actions |= CopyTo
	return pc
}

// CopyBack sets the parameter to copy from device to host
func (pc *ParamConfig) CopyBack() *ParamConfig {
	pc.actions |= CopyBack
	return pc
}

// Copy sets the parameter for bidirectional copy
func (pc *ParamConfig) Copy() *ParamConfig {
	pc.actions |= Copy
	return pc
}

// NoCopy explicitly disables all copy operations for this parameter
func (pc *ParamConfig) NoCopy() *ParamConfig {
	pc.actions = NoAction
	return pc
}

// GetSignature generates the kernel signature for a configuration
func (kc *KernelConfig) GetSignature(runner *Runner) (string, error) {
	return GetSignatureForConfig(kc, runner)
}

// GetSignatureForConfig is a helper that can be called from kernel_execution.go
func GetSignatureForConfig(config *KernelConfig, runner *Runner) (string, error) {
	args := runner.GetKernelArgumentsForConfig(config)
	params := make([]string, 0, len(args))

	for _, arg := range args {
		if arg.Category == "scalar" {
			// Scalars are passed by value
			params = append(params, fmt.Sprintf("%s %s", arg.Type, arg.Name))
		} else if arg.Category == "array_data" || arg.Category == "array_offset" {
			// Find the binding to get its type
			var typeStr string
			if arg.UserArgName != "" {
				// Find the parameter that corresponds to this array
				for _, usage := range config.Parameters {
					if usage.Binding.Name == arg.UserArgName {
						if arg.Category == "array_data" {
							typeStr = runner.dataTypeToCType(usage.Binding.DeviceType)
						} else {
							typeStr = "int_t" // offsets are always int_t
						}
						break
					}
				}
			}

			constStr := ""
			if arg.IsConst {
				constStr = "const "
			}
			params = append(params, fmt.Sprintf("%s%s* %s", constStr, typeStr, arg.Name))
		} else {
			// System arrays (K) or matrices
			constStr := ""
			if arg.IsConst {
				constStr = "const "
			}

			// For matrices, find their type from bindings
			if arg.Category == "matrix" {
				typeStr := "real_t" // default
				for _, usage := range config.Parameters {
					if usage.Binding.Name == arg.Name && usage.Binding.IsMatrix {
						typeStr = runner.dataTypeToCType(usage.Binding.DeviceType)
						break
					}
				}
				params = append(params, fmt.Sprintf("%s%s* %s", constStr, typeStr, arg.Name))
			} else {
				// System arrays already have the complete type including pointer
				params = append(params, fmt.Sprintf("%s%s %s", constStr, arg.Type, arg.Name))
			}
		}
	}

	return strings.Join(params, ",\n\t"), nil
}

// File: runner/kernel_config.go
// Phase 1: Foundation - Define KernelConfig and CopyConfig structures

package runner

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

// ParamConfig is a lightweight builder for configuring parameter actions
// Used during kernel/copy configuration phase
type ParamConfig struct {
	runner  *Runner
	name    string
	binding *DeviceBinding
}

// CopyTo sets the parameter to copy from host to device
func (pc *ParamConfig) CopyTo() *ParamConfig {
	// Actions will be set when building the final configuration
	return pc
}

// CopyBack sets the parameter to copy from device to host
func (pc *ParamConfig) CopyBack() *ParamConfig {
	// Actions will be set when building the final configuration
	return pc
}

// Copy sets the parameter for bidirectional copy
func (pc *ParamConfig) Copy() *ParamConfig {
	// Actions will be set when building the final configuration
	return pc
}

// NoCopy explicitly disables all copy operations for this parameter
func (pc *ParamConfig) NoCopy() *ParamConfig {
	// Actions will be set when building the final configuration
	return pc
}

package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/gocca"
	"reflect"
)

// RunKernelEnhanced executes a kernel defined with DefineKernel
// It handles all data movement automatically based on parameter specifications
func (kr *Runner) RunKernelEnhanced(kernelName string, scalarValues ...interface{}) error {
	// Get kernel definition
	def, exists := kr.kernelDefinitions[kernelName]
	if !exists {
		// Fall back to old RunKernel for backward compatibility
		return kr.RunKernel(kernelName, scalarValues...)
	}

	// Get compiled kernel
	kernel, exists := kr.Kernels[kernelName]
	if !exists {
		return fmt.Errorf("kernel %s not compiled", kernelName)
	}

	// Perform pre-kernel data copies (host→device)
	if err := kr.performPreKernelCopies(def); err != nil {
		return fmt.Errorf("pre-kernel copy failed: %w", err)
	}

	// Build kernel arguments
	args, err := kr.buildKernelArguments(def, scalarValues)
	if err != nil {
		return fmt.Errorf("failed to build arguments: %w", err)
	}

	// Execute kernel
	if err := kernel.RunWithArgs(args...); err != nil {
		return fmt.Errorf("kernel execution failed: %w", err)
	}

	// Perform post-kernel data copies (device→host)
	if err := kr.performPostKernelCopies(def); err != nil {
		return fmt.Errorf("post-kernel copy failed: %w", err)
	}

	return nil
}

// performPreKernelCopies handles all host→device transfers before kernel execution
func (kr *Runner) performPreKernelCopies(def *KernelDefinition) error {
	for _, param := range def.Parameters {
		if param.NeedsCopyTo() {
			if err := kr.copyToDeviceWithConversion(&param); err != nil {
				return fmt.Errorf("failed to copy %s to device: %w", param.Name, err)
			}
		}
	}
	return nil
}

// performPostKernelCopies handles all device→host transfers after kernel execution
func (kr *Runner) performPostKernelCopies(def *KernelDefinition) error {
	for _, param := range def.Parameters {
		if param.NeedsCopyBack() {
			if err := kr.copyFromDeviceWithConversion(&param); err != nil {
				return fmt.Errorf("failed to copy %s from device: %w", param.Name, err)
			}
		}
	}
	return nil
}

// buildKernelArguments constructs the argument list for kernel execution
func (kr *Runner) buildKernelArguments(def *KernelDefinition, scalarValues []interface{}) ([]interface{}, error) {
	var args []interface{}

	// K array is always first
	args = append(args, kr.PooledMemory["K"])

	// Add device matrices in sorted order
	deviceMatrixNames := make([]string, 0)
	for _, p := range def.Parameters {
		if p.IsMatrix && !p.IsStatic {
			deviceMatrixNames = append(deviceMatrixNames, p.Name)
		}
	}
	sortStrings(deviceMatrixNames)

	for _, name := range deviceMatrixNames {
		if mem, exists := kr.PooledMemory[name]; exists {
			args = append(args, mem)
		} else {
			return nil, fmt.Errorf("device matrix %s not found", name)
		}
	}

	// Add arrays
	for _, p := range def.Parameters {
		if p.Direction == builder.DirectionScalar || (p.IsMatrix && !p.IsStatic) {
			continue
		}

		if !p.IsMatrix {
			// Regular array
			globalMem, hasGlobal := kr.PooledMemory[p.Name+"_global"]
			offsetMem, hasOffset := kr.PooledMemory[p.Name+"_offsets"]

			if !hasGlobal || !hasOffset {
				return nil, fmt.Errorf("array %s not properly allocated", p.Name)
			}

			args = append(args, globalMem, offsetMem)
		}
	}

	// Add scalars - match them with parameter definitions
	scalarIndex := 0
	for _, p := range def.Parameters {
		if p.Direction == builder.DirectionScalar {
			if scalarIndex >= len(scalarValues) {
				// Try to get value from binding
				if p.HostBinding != nil {
					args = append(args, kr.getScalarValue(p.HostBinding))
				} else {
					return nil, fmt.Errorf("missing value for scalar %s", p.Name)
				}
			} else {
				args = append(args, scalarValues[scalarIndex])
				scalarIndex++
			}
		}
	}

	return args, nil
}

// getScalarValue extracts the scalar value from a binding
func (kr *Runner) getScalarValue(binding interface{}) interface{} {
	v := reflect.ValueOf(binding)

	// Dereference pointers
	if v.Kind() == reflect.Ptr {
		v = v.Elem()
	}

	// Return the actual value
	return v.Interface()
}

// BuildKernelFromDefinition builds a kernel using stored parameter definitions
func (kr *Runner) BuildKernelFromDefinition(kernelSource, kernelName string) (*gocca.OCCAKernel, error) {
	def, exists := kr.kernelDefinitions[kernelName]
	if !exists {
		// Fall back to regular BuildKernel
		return kr.BuildKernel(kernelSource, kernelName)
	}

	// Use the signature from the definition
	signatureWithName := fmt.Sprintf("@kernel void %s(\n\t%s\n)", kernelName, def.Signature)
	_ = signatureWithName

	// Ensure the kernel source uses the correct signature
	// In practice, you might want to parse and replace the signature in kernelSource
	// For now, we'll just build as-is and let the user ensure consistency

	return kr.BuildKernel(kernelSource, kernelName)
}

// GetKernelSignature returns the generated signature for a defined kernel
func (kr *Runner) GetKernelSignature(kernelName string) (string, error) {
	def, exists := kr.kernelDefinitions[kernelName]
	if !exists {
		return "", fmt.Errorf("kernel %s not defined", kernelName)
	}
	return def.Signature, nil
}

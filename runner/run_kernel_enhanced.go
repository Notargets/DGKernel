package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/gocca"
	"gonum.org/v1/gonum/mat"
	"reflect"
	"unsafe"
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
		if p.Direction == DirectionScalar || (p.IsMatrix && !p.IsStatic) {
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
		if p.Direction == DirectionScalar {
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

// copyFromDeviceWithConversion handles device→host copy with optional type conversion
func (kr *Runner) copyFromDeviceWithConversion(spec *ParamSpec) error {
	mem := kr.GetMemory(spec.Name)
	if mem == nil {
		return fmt.Errorf("memory for %s not found", spec.Name)
	}

	// For now, use the existing CopyArrayToHost for full arrays
	// This would be enhanced to handle partial copies and conversions

	hostBinding := spec.HostBinding
	if hostBinding == nil {
		// Try to get from stored bindings
		if binding, exists := kr.hostBindings[spec.Name]; exists {
			hostBinding = binding
		} else {
			return fmt.Errorf("no host binding for %s", spec.Name)
		}
	}

	// Get total size
	totalSize := spec.Size * sizeOfType(spec.GetEffectiveType())

	// Handle type conversion if needed
	if spec.ConvertType != 0 && spec.ConvertType != spec.DataType {
		// Perform conversion during copy back
		return kr.copyFromDeviceWithTypeConversion(mem, hostBinding, spec.ConvertType, spec.DataType, totalSize)
	}

	// Direct copy based on type
	switch data := hostBinding.(type) {
	case []float32:
		mem.CopyTo(unsafe.Pointer(&data[0]), totalSize)
	case []float64:
		mem.CopyTo(unsafe.Pointer(&data[0]), totalSize)
	case []int32:
		mem.CopyTo(unsafe.Pointer(&data[0]), totalSize)
	case []int64:
		mem.CopyTo(unsafe.Pointer(&data[0]), totalSize)
	case mat.Matrix:
		// Handle mat.Matrix for copy back
		rows, cols := data.Dims()
		size := rows * cols

		// Create temporary buffer to receive device data
		flatData := make([]float64, size)
		mem.CopyTo(unsafe.Pointer(&flatData[0]), totalSize)

		// Type assert to a mutable matrix type
		switch m := data.(type) {
		case *mat.Dense:
			// Unpack column-major data back into the Dense matrix
			for i := 0; i < rows; i++ {
				for j := 0; j < cols; j++ {
					// Column-major: position j*rows + i contains element (i,j)
					val := flatData[j*rows+i]
					m.Set(i, j, val)
				}
			}
		}
	default:
		return fmt.Errorf("unsupported type for copy: %T", data)
	}

	return nil
}

// copyFromDeviceWithTypeConversion performs type conversion during device→host copy
func (kr *Runner) copyFromDeviceWithTypeConversion(deviceMem *gocca.OCCAMemory, hostData interface{},
	fromType, toType builder.DataType, totalSize int64) error {
	// This is a simplified example - full implementation would handle all conversions
	switch fromType {
	case builder.Float32:
		if toType == builder.Float64 {
			// Device has float32, host wants float64
			temp := make([]float32, totalSize/4)
			deviceMem.CopyTo(unsafe.Pointer(&temp[0]), totalSize)

			dst := hostData.([]float64)
			for i, v := range temp {
				dst[i] = float64(v)
			}
			return nil
		}
	case builder.Float64:
		if toType == builder.Float32 {
			// Device has float64, host wants float32
			temp := make([]float64, totalSize/8)
			deviceMem.CopyTo(unsafe.Pointer(&temp[0]), totalSize)

			dst := hostData.([]float32)
			for i, v := range temp {
				dst[i] = float32(v)
			}
			return nil
		}
	}

	return fmt.Errorf("unsupported conversion from %v to %v", fromType, toType)
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

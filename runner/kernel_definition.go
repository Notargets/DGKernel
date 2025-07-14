// File: runner/kernel_definition.go
// Phase 7: Migration and Compatibility - Reimplement DefineKernel using new infrastructure

package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"gonum.org/v1/gonum/mat"
	"sort"
	"strings"
)

// KernelDefinition holds all information about a defined kernel
// DEPRECATED: This struct will be removed in the next major version
type KernelDefinition struct {
	Name       string
	Parameters []builder.ParamSpec
}

// DefineKernel defines a kernel with its parameters using the old API
// DEPRECATED: Use DefineBindings + AllocateDevice + ConfigureKernel instead
func (kr *Runner) DefineKernel(kernelName string, params ...*builder.ParamBuilder) error {
	// Print deprecation warning
	fmt.Printf("WARNING: DefineKernel is deprecated and will be removed. Use DefineBindings + AllocateDevice + ConfigureKernel instead.\n")

	// Extract and validate parameter specifications
	paramSpecs := make([]builder.ParamSpec, len(params))
	for i, p := range params {
		paramSpecs[i] = p.Spec
		if err := paramSpecs[i].Validate(); err != nil {
			return fmt.Errorf("parameter %d: %w", i, err)
		}
	}

	// Phase 1: Define bindings using new infrastructure
	if !kr.IsAllocated {
		// Only define bindings if not already allocated
		err := kr.DefineBindings(params...)
		if err != nil {
			return fmt.Errorf("failed to define bindings: %w", err)
		}

		// Phase 2: Allocate device memory
		err = kr.AllocateDevice()
		if err != nil {
			return fmt.Errorf("failed to allocate device: %w", err)
		}
	} else {
		// If already allocated, just update bindings for any new parameters
		for _, p := range params {
			spec := p.Spec
			if kr.GetBinding(spec.Name) == nil {
				// This is a new parameter, create binding
				binding, err := kr.createBindingFromParam(&spec)
				if err != nil {
					return fmt.Errorf("failed to create binding for %s: %w", spec.Name, err)
				}
				kr.Bindings[spec.Name] = binding

				// Process the parameter (allocate if needed)
				if err := kr.processParameterFromBinding(binding); err != nil {
					return fmt.Errorf("failed to process parameter %s: %w", spec.Name, err)
				}
			}
		}

		// Allocate any new device matrices
		if err := kr.AllocateDeviceMatrices(); err != nil {
			return fmt.Errorf("failed to allocate device matrices: %w", err)
		}
	}

	// Phase 3: Configure kernel using new infrastructure
	paramConfigs := make([]*ParamConfig, 0, len(params))
	for _, p := range params {
		pc := kr.Param(p.Spec.Name)

		// Apply actions based on old API flags
		if p.Spec.NeedsCopyTo() {
			pc = pc.CopyTo()
		}
		if p.Spec.NeedsCopyBack() {
			pc = pc.CopyBack()
		}

		paramConfigs = append(paramConfigs, pc)
	}

	_, err := kr.ConfigureKernel(kernelName, paramConfigs...)
	if err != nil {
		return fmt.Errorf("failed to configure kernel: %w", err)
	}

	// Store kernel definition for backward compatibility
	if kr.kernelDefinitions == nil {
		kr.kernelDefinitions = make(map[string]*KernelDefinition)
	}

	kr.kernelDefinitions[kernelName] = &KernelDefinition{
		Name:       kernelName,
		Parameters: paramSpecs,
	}

	return nil
}

// processParameterFromBinding processes a parameter using its binding
// This is extracted from the old processParameter method
func (kr *Runner) processParameterFromBinding(binding *DeviceBinding) error {
	// Skip scalars - they don't need allocation
	if binding.IsScalar {
		return nil
	}

	// Skip if already allocated
	if _, exists := kr.PooledMemory[binding.Name+"_global"]; exists {
		return nil
	}
	if _, exists := kr.PooledMemory[binding.Name]; exists && binding.IsMatrix {
		return nil
	}

	// Allocate based on binding type
	if binding.IsMatrix && binding.IsStatic {
		return kr.addStaticMatrixFromBinding(binding)
	} else if binding.IsMatrix && !binding.IsStatic {
		return kr.addDeviceMatrixFromBinding(binding)
	} else {
		// Regular array or temp array
		return kr.allocateArrayFromBinding(binding)
	}
}

// GetKernelSignature returns the kernel signature for backward compatibility
// DEPRECATED: Use GetKernelSignatureForConfig instead
func (kr *Runner) GetKernelSignature(kernelName string) (string, error) {
	// Try new API first
	if config, exists := kr.KernelConfigs[kernelName]; exists {
		return config.GetSignature(kr)
	}

	// Fall back to old API
	def, exists := kr.kernelDefinitions[kernelName]
	if !exists {
		return "", fmt.Errorf("kernel %s not defined", kernelName)
	}

	args := kr.GetKernelArgumentsForDefinition(def)
	params := make([]string, 0, len(args))

	for _, arg := range args {
		if arg.Category == "scalar" {
			// Scalars are passed by value
			params = append(params, fmt.Sprintf("%s %s", arg.Type, arg.Name))
		} else if arg.Category == "array_data" || arg.Category == "array_offset" {
			// Find the parameter spec to get its type
			var typeStr string
			if arg.UserArgName != "" {
				// Find the parameter that corresponds to this array
				for _, p := range def.Parameters {
					if p.Name == arg.UserArgName {
						if arg.Category == "array_data" {
							typeStr = kr.getParamTypeString(&p)
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

			// For matrices, find their type from parameters
			if arg.Category == "matrix" {
				typeStr := "real_t" // default
				for _, p := range def.Parameters {
					if p.Name == arg.Name && p.IsMatrix {
						typeStr = kr.getParamTypeString(&p)
						break
					}
				}
				params = append(params, fmt.Sprintf("%s%s* %s", constStr, typeStr, arg.Name))
			} else {
				// System arrays already have the complete type including pointer
				// arg.Type is already "int_t*" for K array, don't add another *
				params = append(params, fmt.Sprintf("%s%s %s", constStr, arg.Type, arg.Name))
			}
		}
	}

	return strings.Join(params, ",\n\t"), nil
}

// GetKernelArgumentsForDefinition is kept for backward compatibility
// DEPRECATED: The new API uses GetKernelArgumentsForConfig
func (kr *Runner) GetKernelArgumentsForDefinition(def *KernelDefinition) []KernelArgument {
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
	var deviceMatrixParams []builder.ParamSpec
	for _, p := range def.Parameters {
		if p.IsMatrix && !p.IsStatic {
			deviceMatrixParams = append(deviceMatrixParams, p)
		}
	}
	// Sort by name for consistent ordering
	sort.Slice(deviceMatrixParams, func(i, j int) bool {
		return deviceMatrixParams[i].Name < deviceMatrixParams[j].Name
	})

	for _, p := range deviceMatrixParams {
		args = append(args, KernelArgument{
			Name:      p.Name,
			Type:      "real_t*",
			MemoryKey: p.Name,
			IsConst:   true,
			Category:  "matrix",
		})
	}

	// 3. Arrays (non-matrix, non-scalar parameters)
	for _, p := range def.Parameters {
		if p.Direction == builder.DirectionScalar || p.IsMatrix {
			continue // Skip scalars and matrices
		}

		// Global data pointer
		args = append(args, KernelArgument{
			Name:        p.Name + "_global",
			Type:        "real_t*",
			MemoryKey:   p.Name + "_global",
			IsConst:     p.IsConst(),
			Category:    "array_data",
			UserArgName: p.Name,
		})

		// Offset array
		args = append(args, KernelArgument{
			Name:        p.Name + "_offsets",
			Type:        "int_t*",
			MemoryKey:   p.Name + "_offsets",
			IsConst:     true,
			Category:    "array_offset",
			UserArgName: p.Name,
		})
	}

	// 4. Scalars last
	for _, p := range def.Parameters {
		if p.Direction == builder.DirectionScalar {
			typeStr := GetScalarTypeName(p.DataType)
			args = append(args, KernelArgument{
				Name:      p.Name,
				Type:      typeStr,
				MemoryKey: "", // Scalars don't have memory keys
				IsConst:   true,
				Category:  "scalar",
			})
		}
	}

	return args
}

// RunKernel executes a kernel using the old API
// DEPRECATED: Use ExecuteKernel instead
func (kr *Runner) RunKernel(kernelName string, scalarValues ...interface{}) error {
	fmt.Printf("WARNING: RunKernel is deprecated and will be removed. Use ExecuteKernel instead.\n")

	// Use the new ExecuteKernel method
	return kr.ExecuteKernel(kernelName, scalarValues...)
}

// The following methods are kept for backward compatibility but are now internal helpers

// performPreKernelCopies handles all host→device transfers before kernel execution
// DEPRECATED: The new API uses executeCopyActions
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
// DEPRECATED: The new API uses executeCopyActions
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

// processParameter handles allocation and setup for a single parameter
// DEPRECATED: The new API uses processParameterFromBinding
func (kr *Runner) processParameter(spec *builder.ParamSpec) error {
	// Validate partitioned data matches kernel configuration
	if spec.IsPartitioned && !kr.IsPartitioned {
		return fmt.Errorf("partitioned data %s provided to non-partitioned kernel", spec.Name)
	}
	if !spec.IsPartitioned && kr.IsPartitioned &&
		spec.Direction != builder.DirectionScalar &&
		spec.Direction != builder.DirectionTemp &&
		!spec.IsMatrix {
		return fmt.Errorf("non-partitioned array %s provided to partitioned kernel", spec.Name)
	}
	if spec.IsPartitioned && spec.PartitionCount != kr.NumPartitions {
		return fmt.Errorf("partition count mismatch for %s: expected %d, got %d",
			spec.Name, kr.NumPartitions, spec.PartitionCount)
	}

	switch spec.Direction {
	case builder.DirectionScalar:
		// Scalars don't need allocation, just type checking
		return nil

	case builder.DirectionTemp:
		// Allocate device-only array
		return kr.allocateTempArray(spec)

	default:
		// Regular arrays - check if already allocated
		if _, exists := kr.PooledMemory[spec.Name+"_global"]; exists {
			// Verify compatibility
			return kr.verifyExistingAllocation(spec)
		}

		// Need to allocate
		if spec.IsMatrix && spec.IsStatic {
			// Static matrix - add to static matrices
			return kr.addStaticMatrixFromSpec(spec)
		} else if spec.IsMatrix {
			// Device matrix
			return kr.addDeviceMatrixFromSpec(spec)
		} else {
			// Regular array
			return kr.allocateArrayFromSpec(spec)
		}
	}
}

// Helper methods that are still used by the compatibility layer

func (kr *Runner) allocateTempArray(spec *builder.ParamSpec) error {
	// Create array spec with effective type
	effectiveType := spec.GetEffectiveType()
	arraySpec := builder.ArraySpec{
		Name:      spec.Name,
		Size:      spec.Size * SizeOfType(effectiveType),
		DataType:  effectiveType,
		Alignment: spec.Alignment,
		IsOutput:  true, // Temp arrays are writable
	}

	return kr.allocateSingleArray(arraySpec, spec)
}

func (kr *Runner) allocateArrayFromSpec(spec *builder.ParamSpec) error {
	// Determine effective type (considering conversion)
	effectiveType := spec.GetEffectiveType()

	// Create array spec
	arraySpec := builder.ArraySpec{
		Name:      spec.Name,
		Size:      spec.Size * SizeOfType(effectiveType),
		DataType:  effectiveType,
		Alignment: spec.Alignment,
		IsOutput:  !spec.IsConst(),
	}

	// Allocate
	if err := kr.allocateSingleArray(arraySpec, spec); err != nil {
		return err
	}

	// Store binding if present
	if spec.HostBinding != nil {
		if kr.hostBindings == nil {
			kr.hostBindings = make(map[string]interface{})
		}
		kr.hostBindings[spec.Name] = spec.HostBinding
	}

	// Perform initial copy if requested
	if spec.NeedsCopyTo() {
		return kr.copyToDeviceWithConversion(spec)
	}

	return nil
}

func (kr *Runner) verifyExistingAllocation(spec *builder.ParamSpec) error {
	meta, exists := kr.arrayMetadata[spec.Name]
	if !exists {
		return fmt.Errorf("array %s allocated but metadata missing", spec.Name)
	}

	// Check type compatibility
	effectiveType := spec.GetEffectiveType()
	if meta.dataType != effectiveType {
		return fmt.Errorf("array %s type mismatch: allocated as %v, requested %v",
			spec.Name, meta.dataType, effectiveType)
	}

	// Check size
	expectedSize := spec.Size * SizeOfType(effectiveType)
	if meta.spec.Size != expectedSize {
		return fmt.Errorf("array %s size mismatch: allocated %d, requested %d",
			spec.Name, meta.spec.Size, expectedSize)
	}

	// Update binding if provided
	if spec.HostBinding != nil {
		if kr.hostBindings == nil {
			kr.hostBindings = make(map[string]interface{})
		}
		kr.hostBindings[spec.Name] = spec.HostBinding
	}

	// Store updated param spec
	meta.paramSpec = spec
	kr.arrayMetadata[spec.Name] = meta

	return nil
}

func (kr *Runner) addStaticMatrixFromSpec(spec *builder.ParamSpec) error {
	var matrix mat.Matrix

	if m, ok := spec.HostBinding.(mat.Matrix); ok {
		matrix = m
	} else if data, ok := spec.HostBinding.([]float64); ok && spec.Stride > 0 {
		// Create matrix from flat array using stride
		matrix = mat.NewDense(len(data)/spec.Stride, spec.Stride, data)
	} else {
		return fmt.Errorf("invalid matrix binding for %s", spec.Name)
	}

	kr.Builder.AddStaticMatrix(spec.Name, matrix)

	// Store metadata
	rows, cols := matrix.Dims()
	meta := ArrayMetadata{
		spec: builder.ArraySpec{
			Name:     spec.Name,
			DataType: spec.GetEffectiveType(),
			Size:     int64(rows * cols * int(SizeOfType(spec.GetEffectiveType()))),
		},
		dataType:  spec.GetEffectiveType(),
		paramSpec: spec,
	}
	kr.arrayMetadata[spec.Name] = meta

	return nil
}

func (kr *Runner) addDeviceMatrixFromSpec(spec *builder.ParamSpec) error {
	var matrix mat.Matrix

	if m, ok := spec.HostBinding.(mat.Matrix); ok {
		matrix = m
	} else if data, ok := spec.HostBinding.([]float64); ok && spec.Stride > 0 {
		// Create matrix from flat array using stride
		matrix = mat.NewDense(len(data)/spec.Stride, spec.Stride, data)
	} else {
		return fmt.Errorf("invalid matrix binding for %s", spec.Name)
	}

	kr.Builder.AddDeviceMatrix(spec.Name, matrix)

	// Store metadata
	rows, cols := matrix.Dims()
	meta := ArrayMetadata{
		spec: builder.ArraySpec{
			Name:     spec.Name,
			DataType: spec.GetEffectiveType(),
			Size:     int64(rows * cols * int(SizeOfType(spec.GetEffectiveType()))),
		},
		dataType:  spec.GetEffectiveType(),
		paramSpec: spec,
	}
	kr.arrayMetadata[spec.Name] = meta

	return nil
}

// getParamTypeString returns the C type string for a parameter based on its effective type
func (kr *Runner) getParamTypeString(param *builder.ParamSpec) string {
	effectiveType := param.GetEffectiveType()
	return kr.dataTypeToCType(effectiveType)
}

// dataTypeToCType converts a builder.DataType to its C type string
func (kr *Runner) dataTypeToCType(dt builder.DataType) string {
	switch dt {
	case builder.Float32:
		return "float"
	case builder.Float64:
		return "double"
	case builder.INT32:
		return "int"
	case builder.INT64:
		return "long"
	default:
		return "double" // fallback
	}
}

// These methods just forward to the Builder for backward compatibility
func (kr *Runner) AddStaticMatrix(name string, m mat.Matrix) {
	kr.Builder.AddStaticMatrix(name, m)
}

func (kr *Runner) AddDeviceMatrix(name string, m mat.Matrix) {
	kr.Builder.AddDeviceMatrix(name, m)
}

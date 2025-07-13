// File: runner/kernel_definition.go
// Complete replacement implementing Phase 1: Array-specific type support

package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"gonum.org/v1/gonum/mat"
	"strings"
)

// KernelDefinition holds all information about a defined kernel
type KernelDefinition struct {
	Name       string
	Parameters []builder.ParamSpec
}

// DefineKernel defines a kernel with its parameters using the new API
func (kr *Runner) DefineKernel(kernelName string, params ...*builder.ParamBuilder) error {
	// Extract and validate parameter specifications
	paramSpecs := make([]builder.ParamSpec, len(params))
	for i, p := range params {
		paramSpecs[i] = p.Spec
		if err := paramSpecs[i].Validate(); err != nil {
			return fmt.Errorf("parameter %d: %w", i, err)
		}
	}

	// Process each parameter
	for _, spec := range paramSpecs {
		if err := kr.processParameter(&spec); err != nil {
			return fmt.Errorf("failed to process parameter %s: %w", spec.Name, err)
		}
	}

	// Allocate all device matrices that were added
	if err := kr.AllocateDeviceMatrices(); err != nil {
		return fmt.Errorf("failed to allocate device matrices: %w", err)
	}

	// Store kernel definition
	if kr.kernelDefinitions == nil {
		kr.kernelDefinitions = make(map[string]*KernelDefinition)
	}

	kr.kernelDefinitions[kernelName] = &KernelDefinition{
		Name:       kernelName,
		Parameters: paramSpecs,
	}

	return nil
}

func (kr *Runner) GetKernelSignature(kernelName string) (string, error) {
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

// processParameter handles allocation and setup for a single parameter
func (kr *Runner) processParameter(spec *builder.ParamSpec) error {
	// NEW: Validate partitioned data matches kernel configuration
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

// allocateTempArray allocates a device-only temporary array
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

// allocateArrayFromSpec allocates an array from parameter specification
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

// verifyExistingAllocation checks if existing allocation is compatible
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

// addStaticMatrixFromSpec adds a static matrix from parameter specification
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

// addDeviceMatrixFromSpec adds a device matrix from parameter specification
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

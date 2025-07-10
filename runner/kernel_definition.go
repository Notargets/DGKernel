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
	Signature  string
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

	signature := kr.generateSignatureFromParams(paramSpecs)
	kr.kernelDefinitions[kernelName] = &KernelDefinition{
		Name:       kernelName,
		Parameters: paramSpecs,
		Signature:  signature,
	}

	return nil
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
	// Create array spec
	arraySpec := builder.ArraySpec{
		Name:      spec.Name,
		Size:      spec.Size * sizeOfType(spec.DataType),
		DataType:  spec.DataType,
		Alignment: spec.Alignment,
		IsOutput:  true, // Temp arrays are writable
	}

	return kr.allocateSingleArray(arraySpec)
}

// allocateArrayFromSpec allocates an array from parameter specification
func (kr *Runner) allocateArrayFromSpec(spec *builder.ParamSpec) error {
	// Determine effective type (considering conversion)
	effectiveType := spec.GetEffectiveType()

	// Create array spec
	arraySpec := builder.ArraySpec{
		Name:      spec.Name,
		Size:      spec.Size * sizeOfType(effectiveType),
		DataType:  effectiveType,
		Alignment: spec.Alignment,
		IsOutput:  !spec.IsConst(),
	}

	// Allocate
	if err := kr.allocateSingleArray(arraySpec); err != nil {
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

// addStaticMatrixFromSpec adds a static matrix from parameter specification
func (kr *Runner) addStaticMatrixFromSpec(spec *builder.ParamSpec) error {
	var matrix mat.Matrix

	if m, ok := spec.HostBinding.(mat.Matrix); ok {
		matrix = m
	} else if spec.HostBinding != nil && spec.Stride > 0 {
		// Convert flat array to matrix
		matrix = kr.flatArrayToMatrix(spec)
	} else {
		return fmt.Errorf("static matrix %s needs valid binding", spec.Name)
	}

	kr.AddStaticMatrix(spec.Name, matrix)
	return nil
}

// addDeviceMatrixFromSpec adds a device matrix from parameter specification
func (kr *Runner) addDeviceMatrixFromSpec(spec *builder.ParamSpec) error {
	var matrix mat.Matrix

	if m, ok := spec.HostBinding.(mat.Matrix); ok {
		matrix = m
	} else if spec.HostBinding != nil && spec.Stride > 0 {
		// Convert flat array to matrix
		matrix = kr.flatArrayToMatrix(spec)
	} else {
		// Create zero matrix if no binding
		matrix = mat.NewDense(spec.MatrixRows, spec.MatrixCols, nil)
	}

	// Add to device matrices collection
	kr.AddDeviceMatrix(spec.Name, matrix)

	// Store binding if present
	if spec.HostBinding != nil {
		if kr.hostBindings == nil {
			kr.hostBindings = make(map[string]interface{})
		}
		kr.hostBindings[spec.Name] = spec.HostBinding
	}

	return nil
}

// flatArrayToMatrix converts a flat array to a matrix using stride
func (kr *Runner) flatArrayToMatrix(spec *builder.ParamSpec) mat.Matrix {
	// This is a simplified version - real implementation would handle type conversion
	if data, ok := spec.HostBinding.([]float64); ok {
		return mat.NewDense(spec.MatrixRows, spec.MatrixCols, data)
	}
	// Handle other types...
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
	expectedSize := spec.Size * sizeOfType(effectiveType)
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

	return nil
}

// generateSignatureFromParams creates kernel signature from parameters
func (kr *Runner) generateSignatureFromParams(params []builder.ParamSpec) string {
	var parts []string

	// K array is always first
	parts = append(parts, "const int_t* K")

	// Add device matrices in sorted order
	deviceMatrixNames := make([]string, 0)
	for _, p := range params {
		if p.IsMatrix && !p.IsStatic {
			deviceMatrixNames = append(deviceMatrixNames, p.Name)
		}
	}
	sortStrings(deviceMatrixNames)

	for _, name := range deviceMatrixNames {
		parts = append(parts, fmt.Sprintf("const real_t* %s", name))
	}

	// Add arrays
	for _, p := range params {
		if p.Direction == builder.DirectionScalar || (p.IsMatrix && !p.IsStatic) {
			continue // Skip scalars and already-added matrices
		}

		if !p.IsMatrix {
			// Regular array
			constStr := ""
			if p.IsConst() {
				constStr = "const "
			}
			parts = append(parts,
				fmt.Sprintf("%sreal_t* %s_global", constStr, p.Name),
				fmt.Sprintf("const int_t* %s_offsets", p.Name))
		}
	}

	// Add scalars last
	for _, p := range params {
		if p.Direction == builder.DirectionScalar {
			typeStr := "real_t"
			if p.DataType == builder.INT32 || p.DataType == builder.INT64 {
				typeStr = "int_t"
			}
			parts = append(parts, fmt.Sprintf("const %s %s", typeStr, p.Name))
		}
	}

	return strings.Join(parts, ",\n\t")
}

// sizeOfType returns the size in bytes of a data type
func sizeOfType(dt builder.DataType) int64 {
	switch dt {
	case builder.Float32, builder.INT32:
		return 4
	case builder.Float64, builder.INT64:
		return 8
	default:
		return 8
	}
}

// sortStrings sorts a slice of strings (simple bubble sort for small slices)
func sortStrings(s []string) {
	for i := 0; i < len(s); i++ {
		for j := i + 1; j < len(s); j++ {
			if s[i] > s[j] {
				s[i], s[j] = s[j], s[i]
			}
		}
	}
}

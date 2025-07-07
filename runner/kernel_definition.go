package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/gocca"
	"gonum.org/v1/gonum/mat"
	"strings"
	"unsafe"
)

// KernelDefinition holds all information about a defined kernel
type KernelDefinition struct {
	Name       string
	Parameters []ParamSpec
	Signature  string
}

// DefineKernel defines a kernel with its parameters using the new API
func (kr *Runner) DefineKernel(kernelName string, params ...*ParamBuilder) error {
	// Extract and validate parameter specifications
	paramSpecs := make([]ParamSpec, len(params))
	for i, p := range params {
		paramSpecs[i] = p.spec
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
func (kr *Runner) processParameter(spec *ParamSpec) error {
	switch spec.Direction {
	case DirectionScalar:
		// Scalars don't need allocation, just type checking
		return nil

	case DirectionTemp:
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
func (kr *Runner) allocateTempArray(spec *ParamSpec) error {
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
func (kr *Runner) allocateArrayFromSpec(spec *ParamSpec) error {
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
func (kr *Runner) addStaticMatrixFromSpec(spec *ParamSpec) error {
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
func (kr *Runner) addDeviceMatrixFromSpec(spec *ParamSpec) error {
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
func (kr *Runner) flatArrayToMatrix(spec *ParamSpec) mat.Matrix {
	// This is a simplified version - real implementation would handle type conversion
	if data, ok := spec.HostBinding.([]float64); ok {
		return mat.NewDense(spec.MatrixRows, spec.MatrixCols, data)
	}
	// Handle other types...
	return nil
}

// verifyExistingAllocation checks if existing allocation is compatible
func (kr *Runner) verifyExistingAllocation(spec *ParamSpec) error {
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
func (kr *Runner) generateSignatureFromParams(params []ParamSpec) string {
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
		if p.Direction == DirectionScalar || (p.IsMatrix && !p.IsStatic) {
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
		if p.Direction == DirectionScalar {
			typeStr := "real_t"
			if p.DataType == builder.INT32 || p.DataType == builder.INT64 {
				typeStr = "int_t"
			}
			parts = append(parts, fmt.Sprintf("const %s %s", typeStr, p.Name))
		}
	}

	return strings.Join(parts, ",\n\t")
}

// copyToDeviceWithConversion handles host→device copy with optional type conversion
func (kr *Runner) copyToDeviceWithConversion(spec *ParamSpec) error {
	mem := kr.GetMemory(spec.Name)
	if mem == nil {
		return fmt.Errorf("memory for %s not found", spec.Name)
	}

	// Handle type conversion if needed
	if spec.ConvertType != 0 && spec.ConvertType != spec.DataType {
		// Perform conversion during copy
		return kr.copyWithTypeConversion(spec.HostBinding, mem, spec.DataType, spec.ConvertType)
	}

	// Direct copy based on type
	switch data := spec.HostBinding.(type) {
	case []float32:
		mem.CopyFrom(unsafe.Pointer(&data[0]), int64(len(data)*4))
	case []float64:
		mem.CopyFrom(unsafe.Pointer(&data[0]), int64(len(data)*8))
	case []int32:
		mem.CopyFrom(unsafe.Pointer(&data[0]), int64(len(data)*4))
	case []int64:
		mem.CopyFrom(unsafe.Pointer(&data[0]), int64(len(data)*8))
		// Add this case to the type switch in copyToDeviceWithConversion method in runner/kernel_definition.go
	case mat.Matrix:
		// Handle mat.Matrix as a flat array
		rows, cols := data.Dims()
		size := rows * cols

		// Extract matrix data as a flat array (column-major)
		flatData := make([]float64, size)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				// Column-major: column j, row i goes to position j*rows + i
				flatData[j*rows+i] = data.At(i, j)
			}
		}
		// Copy the flattened data to device
		mem.CopyFrom(unsafe.Pointer(&flatData[0]), int64(size*8))
	default:
		return fmt.Errorf("unsupported type for copy: %T", data)
	}

	return nil
}

// copyWithTypeConversion performs type conversion during copy
func (kr *Runner) copyWithTypeConversion(hostData interface{}, deviceMem *gocca.OCCAMemory,
	fromType, toType builder.DataType) error {
	// This is a simplified example - full implementation would handle all conversions
	switch fromType {
	case builder.Float64:
		if toType == builder.Float32 {
			// Convert float64 → float32
			src := hostData.([]float64)
			dst := make([]float32, len(src))
			for i, v := range src {
				dst[i] = float32(v)
			}
			deviceMem.CopyFrom(unsafe.Pointer(&dst[0]), int64(len(dst)*4))
			return nil
		}
	case builder.Float32:
		if toType == builder.Float64 {
			// Convert float32 → float64
			src := hostData.([]float32)
			dst := make([]float64, len(src))
			for i, v := range src {
				dst[i] = float64(v)
			}
			deviceMem.CopyFrom(unsafe.Pointer(&dst[0]), int64(len(dst)*8))
			return nil
		}
	}

	return fmt.Errorf("unsupported conversion from %v to %v", fromType, toType)
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

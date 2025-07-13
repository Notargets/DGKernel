// File: runner/builder/builder.go
// Complete replacement implementing Phases 2 and 4

package builder

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"strings"
)

// DataType represents the precision of numerical data
type DataType int

const (
	Float32 DataType = iota + 1
	Float64
	INT32
	INT64
)

// AlignmentType specifies memory alignment requirements
type AlignmentType int

const (
	NoAlignment    AlignmentType = 1
	CacheLineAlign AlignmentType = 64
	WarpAlign      AlignmentType = 128
	PageAlign      AlignmentType = 4096
)

// ArraySpec defines user requirements for array allocation
type ArraySpec struct {
	Name      string
	Size      int64
	Alignment AlignmentType
	DataType  DataType
	IsOutput  bool
}

// Builder manages code generation and execution for partition-parallel Kernels
type Builder struct {
	// Partition configuration
	NumPartitions int
	K             []int
	KpartMax      int // Maximum K value across all partitions

	// Type configuration
	// REMOVED: FloatType - each array determines its own type
	IntType DataType

	// Reference Elements

	// Static data to embed
	StaticMatrices map[string]mat.Matrix

	// Device matrices to allocate in global memory
	DeviceMatrices map[string]mat.Matrix

	// Generated code
	KernelPreamble string
}

// Config holds configuration for creating a Builder
// MODIFIED: Removed FloatType
type Config struct {
	K       []int
	IntType DataType
}

// NewBuilder creates a new Builder instance
// MODIFIED: Removed FloatType handling
func NewBuilder(cfg Config) *Builder {
	if len(cfg.K) == 0 {
		panic("K array cannot be empty")
	}
	// Compute KpartMax
	kpartMax := 0
	for _, k := range cfg.K {
		if k > kpartMax {
			kpartMax = k
		}
	}
	// Set defaults
	intType := cfg.IntType
	if intType == 0 {
		intType = INT64
	}
	kb := &Builder{
		NumPartitions:  len(cfg.K),
		K:              make([]int, len(cfg.K)),
		KpartMax:       kpartMax,
		IntType:        intType,
		StaticMatrices: make(map[string]mat.Matrix),
		DeviceMatrices: make(map[string]mat.Matrix),
	}
	// Copy K values
	copy(kb.K, cfg.K)
	return kb
}

// AddStaticMatrix adds a matrix to be embedded as static const in Kernels
func (kb *Builder) AddStaticMatrix(name string, m mat.Matrix) {
	kb.StaticMatrices[name] = m
}

// AddDeviceMatrix adds a matrix to be allocated in device global memory
func (kb *Builder) AddDeviceMatrix(name string, m mat.Matrix) {
	kb.DeviceMatrices[name] = m
}

// CalculateAlignedOffsetsAndSize computes partition offsets with alignment
func (kb *Builder) CalculateAlignedOffsetsAndSize(spec ArraySpec) ([]int64, int64) {
	offsets := make([]int64, kb.NumPartitions+1)
	totalElements := kb.GetTotalElements()
	bytesPerElement := spec.Size / int64(totalElements)

	// Determine the size of individual values
	var valueSize int64
	switch spec.DataType {
	case Float32, INT32:
		valueSize = 4
	case Float64, INT64:
		valueSize = 8
	default:
		// Default to 8 bytes if not specified
		valueSize = 8
	}

	valuesPerElement := bytesPerElement / valueSize

	alignment := int64(spec.Alignment)
	if alignment == 0 {
		alignment = int64(NoAlignment) // Default to no alignment (1)
	}
	currentByteOffset := int64(0)

	for i := 0; i < kb.NumPartitions; i++ {
		// Align current offset
		if currentByteOffset%alignment != 0 {
			currentByteOffset = ((currentByteOffset + alignment - 1) / alignment) * alignment
		}

		// Store offset in units of VALUES, not elements
		offsets[i] = currentByteOffset / valueSize

		// Advance by partition data size
		partitionValues := int64(kb.K[i]) * valuesPerElement
		currentByteOffset += partitionValues * valueSize
	}

	// Final offset for bounds checking
	if currentByteOffset%alignment != 0 {
		currentByteOffset = ((currentByteOffset + alignment - 1) / alignment) * alignment
	}
	offsets[kb.NumPartitions] = currentByteOffset / valueSize

	return offsets, offsets[kb.NumPartitions] * valueSize
}

// GetTotalElements returns sum of all K values
func (kb *Builder) GetTotalElements() int {
	total := 0
	for _, k := range kb.K {
		total += k
	}
	return total
}

// GeneratePreamble generates the kernel preamble with static data and utilities
// MODIFIED: Now requires array type information
func (kb *Builder) GeneratePreamble(allocatedArrays []string, arrayTypes map[string]DataType) string {
	var sb strings.Builder

	// 1. Type definitions and constants (WITHOUT real_t)
	sb.WriteString(kb.generateTypeDefinitions())

	// 2. Static matrix declarations
	sb.WriteString(kb.generateStaticMatrices(arrayTypes))

	// 3. Partition access macros with type information
	sb.WriteString(kb.generatePartitionMacros(allocatedArrays, arrayTypes))

	// 4. Matrix operation macros with @inner
	sb.WriteString(kb.generateMatrixMacros(arrayTypes))

	kb.KernelPreamble = sb.String()
	return kb.KernelPreamble
}

// generateTypeDefinitions creates type definitions based on precision settings
// MODIFIED: Removed real_t typedef
func (kb *Builder) generateTypeDefinitions() string {
	var sb strings.Builder

	// Only int_t typedef remains
	intTypeStr := "long"
	if kb.IntType == INT32 {
		intTypeStr = "int"
	}

	sb.WriteString(fmt.Sprintf("typedef %s int_t;\n", intTypeStr))
	sb.WriteString("\n")

	// Constants
	sb.WriteString(fmt.Sprintf("#define NPART %d\n", kb.NumPartitions))
	sb.WriteString(fmt.Sprintf("#define KpartMax %d\n", kb.KpartMax))
	sb.WriteString("\n")

	return sb.String()
}

// generateStaticMatrices converts matrices to static array initializations
// MODIFIED: Uses specific type for each matrix
func (kb *Builder) generateStaticMatrices(arrayTypes map[string]DataType) string {
	var sb strings.Builder

	if len(kb.StaticMatrices) > 0 {
		sb.WriteString("// Static matrices\n")
		for name, matrix := range kb.StaticMatrices {
			// Only generate if not also in DeviceMatrices
			if _, isDevice := kb.DeviceMatrices[name]; !isDevice {
				// Get type for this matrix
				matrixType := Float64 // default
				if t, exists := arrayTypes[name]; exists {
					matrixType = t
				}
				sb.WriteString(kb.formatStaticMatrix(name, matrix, matrixType))
			}
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

// formatStaticMatrix formats a single matrix as a static C array
// MODIFIED: Takes explicit type parameter
func (kb *Builder) formatStaticMatrix(name string, m mat.Matrix, dataType DataType) string {
	rows, cols := m.Dims()
	var sb strings.Builder

	typeStr := "double"
	suffix := ""
	if dataType == Float32 {
		typeStr = "float"
		suffix = "f"
	}

	// COLUMN-MAJOR STORAGE: Static matrices are generated in column-major format
	sb.WriteString(fmt.Sprintf("const %s %s[%d][%d] = {\n",
		typeStr, name, cols, rows))

	// Generate column-major data
	for j := 0; j < cols; j++ {
		sb.WriteString("  {")
		for i := 0; i < rows; i++ {
			val := m.At(i, j)
			if dataType == Float32 {
				sb.WriteString(fmt.Sprintf("%.8f%s", val, suffix))
			} else {
				sb.WriteString(fmt.Sprintf("%.16e", val))
			}
			if i < rows-1 {
				sb.WriteString(", ")
			}
		}
		sb.WriteString("}")
		if j < cols-1 {
			sb.WriteString(",")
		}
		sb.WriteString("\n")
	}
	sb.WriteString("};\n")

	return sb.String()
}

// generatePartitionMacros creates macros for accessing partitioned data
// MODIFIED: Now casts to specific type for each array
func (kb *Builder) generatePartitionMacros(allocatedArrays []string, arrayTypes map[string]DataType) string {
	var sb strings.Builder

	sb.WriteString("// Partition access macros\n")

	// Generate macro for each allocated array with proper type cast
	for _, arrayName := range allocatedArrays {
		// Get the type for this array
		arrayType := Float64 // default
		if t, exists := arrayTypes[arrayName]; exists {
			arrayType = t
		}

		typeStr := "double"
		if arrayType == Float32 {
			typeStr = "float"
		} else if arrayType == INT32 {
			typeStr = "int"
		} else if arrayType == INT64 {
			typeStr = "long"
		}

		// Cast to specific type instead of using real_t*
		sb.WriteString(fmt.Sprintf("#define %s_PART(part) ((%s*)(%s_global + %s_offsets[part]))\n",
			arrayName, typeStr, arrayName, arrayName))
	}

	sb.WriteString("\n")
	return sb.String()
}

// generateMatrixMacros generates MATMUL macros with @inner loops
// MODIFIED: Uses specific types for matrix operations
func (kb *Builder) generateMatrixMacros(arrayTypes map[string]DataType) string {
	var sb strings.Builder

	// Generate macros for static matrices
	for name, matrix := range kb.StaticMatrices {
		// Skip if also in DeviceMatrices
		if _, isDevice := kb.DeviceMatrices[name]; !isDevice {
			matrixType := Float64 // default
			if t, exists := arrayTypes[name]; exists {
				matrixType = t
			}
			sb.WriteString(kb.generateStaticMatrixMacro(name, matrix, matrixType))
		}
	}

	// Generate macros for device matrices
	for name, matrix := range kb.DeviceMatrices {
		matrixType := Float64 // default
		if t, exists := arrayTypes[name]; exists {
			matrixType = t
		}
		sb.WriteString(kb.generateDeviceMatrixMacro(name, matrix, matrixType))
	}

	return sb.String()
}

// generateStaticMatrixMacro generates a MATMUL macro for a static matrix
// MODIFIED: Uses specific type instead of real_t
func (kb *Builder) generateStaticMatrixMacro(name string, matrix mat.Matrix, dataType DataType) string {
	return kb.generateSingleMatrixMacro(name, matrix, true, dataType)
}

// generateDeviceMatrixMacro generates a MATMUL macro for a device matrix
// MODIFIED: Uses specific type instead of real_t
func (kb *Builder) generateDeviceMatrixMacro(name string, matrix mat.Matrix, dataType DataType) string {
	return kb.generateSingleMatrixMacro(name, matrix, false, dataType)
}

// generateSingleMatrixMacro generates both MATMUL and MATMUL_ADD macros
// MODIFIED: Uses specific type for accumulator and zero constant
func (kb *Builder) generateSingleMatrixMacro(name string, matrix mat.Matrix, isStatic bool, dataType DataType) string {
	rows, cols := matrix.Dims()
	var sb strings.Builder

	// Determine type and zero constant
	typeStr := "double"
	zeroStr := "0.0"
	if dataType == Float32 {
		typeStr = "float"
		zeroStr = "0.0f"
	}

	// Standard multiply macro: OUT = Matrix × IN
	sb.WriteString(fmt.Sprintf("#define MATMUL_%s(IN, OUT, K_VAL) {\\\n", name))
	sb.WriteString(fmt.Sprintf("  for (int ii = 0; ii < %d; ++ii) {\\\n", rows))
	sb.WriteString("    for (int elem = 0; elem < KpartMax; ++elem; @inner) {\\\n")
	sb.WriteString("      if (elem < (K_VAL)) {\\\n")
	sb.WriteString(fmt.Sprintf("        %s sum = %s;\\\n", typeStr, zeroStr))
	sb.WriteString(fmt.Sprintf("        for (int jj = 0; jj < %d; ++jj) {\\\n", cols))

	if isStatic {
		sb.WriteString(fmt.Sprintf("          sum += %s[jj][ii] * (IN)[elem * %d + jj];\\\n", name, cols))
	} else {
		sb.WriteString(fmt.Sprintf("          sum += %s[jj * %d + ii] * (IN)[elem * %d + jj];\\\n", name, rows, cols))
	}

	sb.WriteString("        }\\\n")
	sb.WriteString(fmt.Sprintf("        (OUT)[elem * %d + ii] = sum;\\\n", rows))
	sb.WriteString("      }\\\n")
	sb.WriteString("    }\\\n")
	sb.WriteString("  }\\\n")
	sb.WriteString("}\n\n")

	// Accumulating multiply macro: OUT += Matrix × IN
	sb.WriteString(fmt.Sprintf("#define MATMUL_ADD_%s(IN, OUT, K_VAL) {\\\n", name))
	sb.WriteString(fmt.Sprintf("  for (int ii = 0; ii < %d; ++ii) {\\\n", rows))
	sb.WriteString("    for (int elem = 0; elem < KpartMax; ++elem; @inner) {\\\n")
	sb.WriteString("      if (elem < (K_VAL)) {\\\n")
	sb.WriteString(fmt.Sprintf("        %s sum = %s;\\\n", typeStr, zeroStr))
	sb.WriteString(fmt.Sprintf("        for (int jj = 0; jj < %d; ++jj) {\\\n", cols))

	if isStatic {
		sb.WriteString(fmt.Sprintf("          sum += %s[jj][ii] * (IN)[elem * %d + jj];\\\n", name, cols))
	} else {
		sb.WriteString(fmt.Sprintf("          sum += %s[jj * %d + ii] * (IN)[elem * %d + jj];\\\n", name, rows, cols))
	}

	sb.WriteString("        }\\\n")
	sb.WriteString(fmt.Sprintf("        (OUT)[elem * %d + ii] += sum;\\\n", rows))
	sb.WriteString("      }\\\n")
	sb.WriteString("    }\\\n")
	sb.WriteString("  }\\\n")
	sb.WriteString("}\n\n")

	return sb.String()
}

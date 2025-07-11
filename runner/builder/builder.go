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
	IsOutput  bool // NEW: Explicitly declare if this is an output array
}

// Builder manages code generation and execution for partition-parallel Kernels
type Builder struct {
	// Partition configuration
	NumPartitions int
	K             []int
	KpartMax      int // Maximum K value across all partitions

	// Type configuration
	FloatType DataType
	IntType   DataType

	// Reference Elements

	// Static data to embed
	StaticMatrices map[string]mat.Matrix

	// Device matrices to allocate in global memory (NEW)
	DeviceMatrices map[string]mat.Matrix

	// Array tracking for macro generation
	AllocatedArrays []string

	// Generated code
	KernelPreamble string
}

// Config holds configuration for creating a Builder
type Config struct {
	K         []int
	FloatType DataType
	IntType   DataType
}

// NewBuilder creates a new Builder instance
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
	floatType := cfg.FloatType
	if floatType == 0 {
		floatType = Float64
	}
	intType := cfg.IntType
	if intType == 0 {
		intType = INT64
	}
	kb := &Builder{
		NumPartitions:   len(cfg.K),
		K:               make([]int, len(cfg.K)),
		KpartMax:        kpartMax,
		FloatType:       floatType,
		IntType:         intType,
		StaticMatrices:  make(map[string]mat.Matrix),
		DeviceMatrices:  make(map[string]mat.Matrix), // NEW: Initialize DeviceMatrices
		AllocatedArrays: []string{},
	}
	// Copy K values
	copy(kb.K, cfg.K)
	return kb
}

// AddStaticMatrix adds a matrix to be embedded as static const in Kernels
func (kb *Builder) AddStaticMatrix(name string, m mat.Matrix) {
	kb.StaticMatrices[name] = m
}

// AddDeviceMatrix adds a matrix to be allocated in device global memory (NEW)
func (kb *Builder) AddDeviceMatrix(name string, m mat.Matrix) {
	kb.DeviceMatrices[name] = m
}

func (kb *Builder) GetAllocatedArrays() []string {
	return kb.AllocatedArrays
}

// CalculateAlignedOffsetsAndSize computes partition offsets with alignment
func (kb *Builder) CalculateAlignedOffsetsAndSize(spec ArraySpec) (
	[]int64, int64) {
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
		// This makes pointer arithmetic work correctly: ptr + offset
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

	// return offsets, currentByteOffset
	return offsets, offsets[kb.NumPartitions] * valueSize
}

// CalculateAlignedOffsetsAndSize computes partition offsets with alignment
// DEBUG VERSION - Add this to builder/builder.go temporarily
func (kb *Builder) CalculateAlignedOffsetsAndSize_debug(spec ArraySpec) (
	[]int64, int64) {
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

	// DEBUG: Print initial setup
	fmt.Printf("\n=== CalculateAlignedOffsetsAndSize DEBUG ===\n")
	fmt.Printf("Array: %s\n", spec.Name)
	fmt.Printf("Total elements: %d\n", totalElements)
	fmt.Printf("Spec.Size: %d bytes\n", spec.Size)
	fmt.Printf("Bytes per element: %d\n", bytesPerElement)
	fmt.Printf("Value size: %d bytes\n", valueSize)
	fmt.Printf("Values per element: %d\n", valuesPerElement)
	fmt.Printf("Alignment: %d bytes\n", alignment)
	fmt.Printf("NumPartitions: %d\n", kb.NumPartitions)
	fmt.Printf("K values: %v\n", kb.K)
	fmt.Printf("\n")

	for i := 0; i < kb.NumPartitions; i++ {
		// DEBUG: Print state before alignment
		fmt.Printf("Partition %d:\n", i)
		fmt.Printf("  Current byte offset (before align): %d\n", currentByteOffset)

		// Align current offset
		oldByteOffset := currentByteOffset
		if currentByteOffset%alignment != 0 {
			currentByteOffset = ((currentByteOffset + alignment - 1) / alignment) * alignment
		}

		// DEBUG: Print alignment adjustment
		if oldByteOffset != currentByteOffset {
			fmt.Printf("  Alignment adjusted: %d -> %d (added %d bytes)\n",
				oldByteOffset, currentByteOffset, currentByteOffset-oldByteOffset)
		}

		// Store offset in units of VALUES, not elements
		// This makes pointer arithmetic work correctly: ptr + offset
		offsets[i] = currentByteOffset / valueSize

		// DEBUG: Print offset storage
		fmt.Printf("  Stored offset[%d]: %d values (= %d bytes)\n",
			i, offsets[i], offsets[i]*valueSize)

		// Advance by partition data size
		partitionValues := int64(kb.K[i]) * valuesPerElement
		partitionBytes := partitionValues * valueSize

		// DEBUG: Print partition size calculation
		fmt.Printf("  Partition size: %d elements × %d values/elem = %d values\n",
			kb.K[i], valuesPerElement, partitionValues)
		fmt.Printf("  Partition bytes: %d values × %d bytes/value = %d bytes\n",
			partitionValues, valueSize, partitionBytes)

		currentByteOffset += partitionBytes

		// DEBUG: Print new offset
		fmt.Printf("  New byte offset (after data): %d\n", currentByteOffset)
		fmt.Printf("\n")
	}

	// Final offset for bounds checking
	fmt.Printf("Final alignment check:\n")
	fmt.Printf("  Current byte offset: %d\n", currentByteOffset)

	oldByteOffset := currentByteOffset
	if currentByteOffset%alignment != 0 {
		currentByteOffset = ((currentByteOffset + alignment - 1) / alignment) * alignment
	}

	if oldByteOffset != currentByteOffset {
		fmt.Printf("  Final alignment: %d -> %d (added %d bytes)\n",
			oldByteOffset, currentByteOffset, currentByteOffset-oldByteOffset)
	}

	offsets[kb.NumPartitions] = currentByteOffset / valueSize

	// DEBUG: Print final results
	fmt.Printf("\nFinal Results:\n")
	fmt.Printf("  offsets array: %v (in values)\n", offsets)
	fmt.Printf("  offsets in bytes: [")
	for i, off := range offsets {
		if i > 0 {
			fmt.Printf(", ")
		}
		fmt.Printf("%d", off*valueSize)
	}
	fmt.Printf("]\n")
	fmt.Printf("  Returned total size: %d bytes\n", currentByteOffset)
	fmt.Printf("  Expected total based on final offset: %d bytes\n", offsets[kb.NumPartitions]*valueSize)

	// DEBUG: Verify each partition can be accessed
	fmt.Printf("\nVerification - Can each partition be accessed?\n")
	for i := 0; i < kb.NumPartitions; i++ {
		startByte := offsets[i] * valueSize
		partitionValues := int64(kb.K[i]) * valuesPerElement
		partitionBytes := partitionValues * valueSize
		endByte := startByte + partitionBytes

		fmt.Printf("  Partition %d: bytes [%d, %d), size %d",
			i, startByte, endByte, partitionBytes)

		if endByte > currentByteOffset {
			fmt.Printf(" *** EXCEEDS ALLOCATION! ***")
		}
		fmt.Printf("\n")
	}
	fmt.Printf("=== END DEBUG ===\n\n")

	return offsets, currentByteOffset
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
func (kb *Builder) GeneratePreamble() string {
	var sb strings.Builder

	// 1. Type definitions and constants
	sb.WriteString(kb.generateTypeDefinitions())

	// 2. Static matrix declarations
	sb.WriteString(kb.generateStaticMatrices())

	// 3. Partition access macros
	sb.WriteString(kb.generatePartitionMacros())

	// 4. Matrix operation macros with @inner
	sb.WriteString(kb.generateMatrixMacros())

	kb.KernelPreamble = sb.String()
	return kb.KernelPreamble
}

// generateTypeDefinitions creates type definitions based on precision settings
func (kb *Builder) generateTypeDefinitions() string {
	var sb strings.Builder

	// Type definitions
	floatTypeStr := "double"
	floatSuffix := ""
	if kb.FloatType == Float32 {
		floatTypeStr = "float"
		floatSuffix = "f"
	}

	intTypeStr := "long"
	if kb.IntType == INT32 {
		intTypeStr = "int"
	}

	sb.WriteString(fmt.Sprintf("typedef %s real_t;\n", floatTypeStr))
	sb.WriteString(fmt.Sprintf("typedef %s int_t;\n", intTypeStr))
	sb.WriteString(fmt.Sprintf("#define REAL_ZERO 0.0%s\n", floatSuffix))
	sb.WriteString(fmt.Sprintf("#define REAL_ONE 1.0%s\n", floatSuffix))
	sb.WriteString("\n")

	// Constants
	sb.WriteString(fmt.Sprintf("#define NPART %d\n", kb.NumPartitions))
	sb.WriteString(fmt.Sprintf("#define KpartMax %d\n", kb.KpartMax))
	sb.WriteString("\n")

	return sb.String()
}

// generateStaticMatrices converts matrices to static array initializations
func (kb *Builder) generateStaticMatrices() string {
	var sb strings.Builder

	if len(kb.StaticMatrices) > 0 {
		sb.WriteString("// Static matrices\n")
		for name, matrix := range kb.StaticMatrices {
			// Only generate if not also in DeviceMatrices (MODIFIED)
			if _, isDevice := kb.DeviceMatrices[name]; !isDevice {
				sb.WriteString(kb.formatStaticMatrix(name, matrix))
			}
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

// formatStaticMatrix formats a single matrix as a static C array
// IMPORTANT: This function transposes matrices during generation to convert from
// Go's row-major format to the column-major format expected by numerical
// libraries and GPU kernels.
func (kb *Builder) formatStaticMatrix(name string, m mat.Matrix) string {
	rows, cols := m.Dims()
	var sb strings.Builder

	typeStr := "double"
	if kb.FloatType == Float32 {
		typeStr = "float"
	}

	// COLUMN-MAJOR STORAGE: Static matrices are generated in column-major format
	// for consistency with device matrices and numerical libraries.
	//
	// The matrix is declared with [cols][rows] to maintain column-major layout
	// in C, where the first index varies fastest in memory.
	sb.WriteString(fmt.Sprintf("// Matrix %s stored in column-major format\n", name))
	sb.WriteString(fmt.Sprintf("const %s %s[%d][%d] = {\n", typeStr, name, cols, rows))

	// Write columns (not rows) - this transposes the matrix
	for j := 0; j < cols; j++ {
		sb.WriteString("    {")
		for i := 0; i < rows; i++ {
			if i > 0 {
				sb.WriteString(", ")
			}
			val := m.At(i, j) // Note: reading row i, col j but writing as column j
			if kb.FloatType == Float32 {
				sb.WriteString(fmt.Sprintf("%.7ef", val))
			} else {
				sb.WriteString(fmt.Sprintf("%.15e", val))
			}
		}
		sb.WriteString("}")
		if j < cols-1 {
			sb.WriteString(",")
		}
		sb.WriteString("\n")
	}
	sb.WriteString("};\n\n")

	return sb.String()
}

// generatePartitionMacros creates macros for partition data access
func (kb *Builder) generatePartitionMacros() string {
	var sb strings.Builder

	sb.WriteString("// Partition access macros\n")

	for _, arrayName := range kb.AllocatedArrays {
		sb.WriteString(fmt.Sprintf("#define %s_PART(part) (%s_global + %s_offsets[part])\n",
			arrayName, arrayName, arrayName))
	}

	if len(kb.AllocatedArrays) > 0 {
		sb.WriteString("\n")
	}

	return sb.String()
}

// generateMatrixMacros creates matrix multiplication macros with @inner loop (MODIFIED)
func (kb *Builder) generateMatrixMacros() string {
	var sb strings.Builder

	sb.WriteString("// Matrix multiplication macros\n")
	sb.WriteString("// Automatically infer strides from matrix dimensions\n\n")

	// Generate static matrix macros (existing logic)
	for name, matrix := range kb.StaticMatrices {
		sb.WriteString(kb.generateStaticMatrixMacro(name, matrix))
	}

	// Generate device matrix macros (NEW logic)
	for name, matrix := range kb.DeviceMatrices {
		sb.WriteString(kb.generateDeviceMatrixMacro(name, matrix))
	}

	return sb.String()
}

// generateStaticMatrixMacro generates macro for static matrix with column-major access
func (kb *Builder) generateStaticMatrixMacro(name string, matrix mat.Matrix) string {
	rows, cols := matrix.Dims()
	var sb strings.Builder

	// IMPORTANT: Matrix is stored in column-major format
	// Access pattern: matrix[col][row] for column-major storage
	sb.WriteString(fmt.Sprintf("// MATMUL macro for %s (column-major storage)\n", name))

	// Standard multiply: OUT = Matrix × IN
	sb.WriteString(fmt.Sprintf("#define MATMUL_%s(IN, OUT, K_VAL) \\\n", name))
	sb.WriteString("    do { \\\n")
	sb.WriteString(fmt.Sprintf("        for (int i = 0; i < %d; ++i) { \\\n", rows))
	sb.WriteString("            for (int elem = 0; elem < KpartMax; ++elem; @inner) { \\\n")
	sb.WriteString("                if (elem < (K_VAL)) { \\\n")
	sb.WriteString("                    real_t sum = REAL_ZERO; \\\n")
	sb.WriteString(fmt.Sprintf("                    for (int j = 0; j < %d; ++j) { \\\n", cols))
	// Column-major access: matrix[j][i] instead of matrix[i][j]
	sb.WriteString(fmt.Sprintf("                        sum += %s[j][i] * (IN)[elem * %d + j]; \\\n", name, cols))
	sb.WriteString("                    } \\\n")
	sb.WriteString(fmt.Sprintf("                    (OUT)[elem * %d + i] = sum; \\\n", rows))
	sb.WriteString("                } \\\n")
	sb.WriteString("            } \\\n")
	sb.WriteString("        } \\\n")
	sb.WriteString("    } while(0)\n\n")

	// Accumulating multiply: OUT += Matrix × IN
	sb.WriteString(fmt.Sprintf("#define MATMUL_ADD_%s(IN, OUT, K_VAL) \\\n", name))
	sb.WriteString("    do { \\\n")
	sb.WriteString(fmt.Sprintf("        for (int i = 0; i < %d; ++i) { \\\n", rows))
	sb.WriteString("            for (int elem = 0; elem < KpartMax; ++elem; @inner) { \\\n")
	sb.WriteString("                if (elem < (K_VAL)) { \\\n")
	sb.WriteString("                    real_t sum = REAL_ZERO; \\\n")
	sb.WriteString(fmt.Sprintf("                    for (int j = 0; j < %d; ++j) { \\\n", cols))
	// Column-major access: matrix[j][i] instead of matrix[i][j]
	sb.WriteString(fmt.Sprintf("                        sum += %s[j][i] * (IN)[elem * %d + j]; \\\n", name, cols))
	sb.WriteString("                    } \\\n")
	sb.WriteString(fmt.Sprintf("                    (OUT)[elem * %d + i] += sum; \\\n", rows))
	sb.WriteString("                } \\\n")
	sb.WriteString("            } \\\n")
	sb.WriteString("        } \\\n")
	sb.WriteString("    } while(0)\n\n")

	return sb.String()
}

// generateDeviceMatrixMacro generates macro for device matrix with column-major access
func (kb *Builder) generateDeviceMatrixMacro(name string, matrix mat.Matrix) string {
	rows, cols := matrix.Dims()
	var sb strings.Builder

	// IMPORTANT: Device matrix pointer points to column-major data
	sb.WriteString(fmt.Sprintf("// MATMUL macro for device matrix %s (column-major storage)\n", name))

	// Device matrix macro references the matrix pointer by name directly
	sb.WriteString(fmt.Sprintf("#define MATMUL_%s(IN, OUT, K_VAL) \\\n", name))
	sb.WriteString("    do { \\\n")
	sb.WriteString(fmt.Sprintf("        for (int i = 0; i < %d; ++i) { \\\n", rows))
	sb.WriteString("            for (int elem = 0; elem < KpartMax; ++elem; @inner) { \\\n")
	sb.WriteString("                if (elem < (K_VAL)) { \\\n")
	sb.WriteString("                    real_t sum = REAL_ZERO; \\\n")
	sb.WriteString(fmt.Sprintf("                    for (int j = 0; j < %d; ++j) { \\\n", cols))
	// Column-major access for device matrix: [j*rows + i]
	sb.WriteString(fmt.Sprintf("                        sum += %s[j * %d + i] * (IN)[elem * %d + j]; \\\n", name, rows, cols))
	sb.WriteString("                    } \\\n")
	sb.WriteString(fmt.Sprintf("                    (OUT)[elem * %d + i] = sum; \\\n", rows))
	sb.WriteString("                } \\\n")
	sb.WriteString("            } \\\n")
	sb.WriteString("        } \\\n")
	sb.WriteString("    } while(0)\n\n")

	// Similar for MATMUL_ADD...

	return sb.String()
}

// GetIntSize returns the size of the integer type in bytes
func (kb *Builder) GetIntSize() int {
	if kb.IntType == INT32 {
		return 4
	}
	return 8
}

// Note: GenerateKernelSignature is now implemented in the runner package using GetKernelArguments()

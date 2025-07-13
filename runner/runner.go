package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/gocca"
	"gonum.org/v1/gonum/mat"
	"unsafe"
)

// ArrayMetadata stores information about allocated arrays
type ArrayMetadata struct {
	spec      builder.ArraySpec
	dataType  builder.DataType
	isOutput  bool
	paramSpec *builder.ParamSpec
}

// Runner orchestrates kernel compilation and execution with flexible parameter handling
type Runner struct {
	*builder.Builder
	IsPartitioned     bool // NEW: Track if this runner uses partitioned data
	Device            *gocca.OCCADevice
	Kernels           map[string]*gocca.OCCAKernel
	PooledMemory      map[string]*gocca.OCCAMemory
	arrayMetadata     map[string]ArrayMetadata
	kernelDefinitions map[string]*KernelDefinition
	hostBindings      map[string]interface{} // Store host data bindings
	hostOffsets       map[string][]int64     // NEW: Store host-side copy of offsets
}

// NewRunner creates a new Runner instance
func NewRunner(device *gocca.OCCADevice, Config builder.Config) (kr *Runner) {
	bld := builder.NewBuilder(Config)

	// Note: Removed GetMemoryAllocated check as it doesn't exist in gocca API
	// Users should ensure clean device state before creating a new runner

	if bld.KpartMax > 1048576 { // 2^20 elements
		panic(fmt.Sprintf("KpartMax exceeds 2^20 (1048576), usually caused by unbalanced workloads.\n"+
			"Found KpartMax=%d. Please balance K values or increase partition count.\n"+
			"Current K values: %v\n"+
			"Reduce partition sizes.", bld.KpartMax, bld.K))
	}

	// Allocate K array on Device
	intSize := 8 // INT64
	if bld.IntType == builder.INT32 {
		intSize = 4
	}

	kr = &Runner{
		Builder:           bld,
		arrayMetadata:     make(map[string]ArrayMetadata),
		Device:            device,
		Kernels:           make(map[string]*gocca.OCCAKernel),
		PooledMemory:      make(map[string]*gocca.OCCAMemory),
		kernelDefinitions: make(map[string]*KernelDefinition),
		hostBindings:      make(map[string]interface{}),
		IsPartitioned:     len(Config.K) > 1,
		hostOffsets:       make(map[string][]int64), // NEW: Initialize host offset storage
	}

	kMem := device.Malloc(int64(len(bld.K)*intSize), unsafe.Pointer(&bld.K[0]), nil)
	kr.PooledMemory["K"] = kMem
	return
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

// AddStaticMatrix adds a matrix that will be embedded as static const in kernels
func (kr *Runner) AddStaticMatrix(name string, m mat.Matrix) {
	kr.Builder.AddStaticMatrix(name, m)
}

// AddDeviceMatrix adds a matrix that will be allocated in device global memory
func (kr *Runner) AddDeviceMatrix(name string, m mat.Matrix) {
	kr.Builder.AddDeviceMatrix(name, m)
}

// GetAllocatedArrays returns a sorted list of allocated array names
func (kr *Runner) GetAllocatedArrays() []string {
	arrays := make([]string, 0, len(kr.arrayMetadata))
	for name := range kr.arrayMetadata {
		arrays = append(arrays, name)
	}
	SortStrings(arrays)
	return arrays
}

// GetMemory returns the device memory for a named array
func (kr *Runner) GetMemory(arrayName string) *gocca.OCCAMemory {
	if mem, exists := kr.PooledMemory[arrayName+"_global"]; exists {
		return mem
	}
	return nil
}

// GetOffsets returns the offset memory for a named array
func (kr *Runner) GetOffsets(arrayName string) *gocca.OCCAMemory {
	if mem, exists := kr.PooledMemory[arrayName+"_offsets"]; exists {
		return mem
	}
	return nil
}

// GetArrayMetadata returns metadata for a named array
func (kr *Runner) GetArrayMetadata(arrayName string) (ArrayMetadata, bool) {
	meta, exists := kr.arrayMetadata[arrayName]
	return meta, exists
}

// GetArrayLogicalSize returns the number of logical elements in an array
func (kr *Runner) GetArrayLogicalSize(name string) (int, error) {
	metadata, exists := kr.arrayMetadata[name]
	if !exists {
		return 0, fmt.Errorf("array %s not found", name)
	}

	// Return the total size divided by the size of the data type
	elementSize := SizeOfType(metadata.dataType)
	return int(metadata.spec.Size / elementSize), nil
}

// GetIntSize returns the size of int type in bytes
func (kr *Runner) GetIntSize() int {
	if kr.IntType == builder.INT32 {
		return 4
	}
	return 8
}

// Free releases all resources
func (kr *Runner) Free() {
	// Free Kernels
	for _, kernel := range kr.Kernels {
		kernel.Free()
	}

	// Free memory
	for _, mem := range kr.PooledMemory {
		mem.Free()
	}
}

// Modified allocateSingleArray in runner/runner.go
func (kr *Runner) allocateSingleArray(spec builder.ArraySpec, paramSpec *builder.ParamSpec) error {

	// Calculate aligned offsets and total size
	offsets, totalSize := kr.CalculateAlignedOffsetsAndSize(spec)

	// Allocate global memory
	globalMem := kr.Device.Malloc(totalSize, nil, nil)
	kr.PooledMemory[spec.Name+"_global"] = globalMem

	// Allocate and populate offset array
	intSize := kr.GetIntSize()
	offsetsSize := int64(len(offsets) * intSize)

	var offsetMem *gocca.OCCAMemory
	if intSize == 4 {
		// Convert to int32 for Device
		offsets32 := make([]int32, len(offsets))
		for i, v := range offsets {
			offsets32[i] = int32(v)
		}
		offsetMem = kr.Device.Malloc(offsetsSize, unsafe.Pointer(&offsets32[0]), nil)
	} else {
		offsetMem = kr.Device.Malloc(offsetsSize, unsafe.Pointer(&offsets[0]), nil)
	}
	kr.PooledMemory[spec.Name+"_offsets"] = offsetMem

	// NEW: Store offsets on host for validation
	kr.hostOffsets[spec.Name] = make([]int64, len(offsets))
	copy(kr.hostOffsets[spec.Name], offsets)

	// NEW: Immediate validation
	if err := kr.validateOffsets(spec.Name, "after allocation"); err != nil {
		return fmt.Errorf("offset corruption detected immediately after allocation: %w", err)
	}

	// Track allocation
	kr.arrayMetadata[spec.Name] = ArrayMetadata{
		spec:      spec,
		dataType:  spec.DataType,
		isOutput:  spec.IsOutput,
		paramSpec: paramSpec,
	}

	return nil
}

func (kr *Runner) validateOffsets(name string, context string) error {
	// Get expected offsets from host storage
	expectedOffsets, exists := kr.hostOffsets[name]
	if !exists {
		return fmt.Errorf("no host offsets found for %s", name)
	}

	// Read offsets from device
	offsetsMem := kr.PooledMemory[name+"_offsets"]
	if offsetsMem == nil {
		return fmt.Errorf("no device offsets found for %s", name)
	}

	var actualOffsets []int64
	if kr.GetIntSize() == 4 {
		offsets32 := make([]int32, len(expectedOffsets))
		offsetsMem.CopyTo(unsafe.Pointer(&offsets32[0]), int64(len(offsets32)*4))
		actualOffsets = make([]int64, len(offsets32))
		for i, v := range offsets32 {
			actualOffsets[i] = int64(v)
		}
	} else {
		actualOffsets = make([]int64, len(expectedOffsets))
		offsetsMem.CopyTo(unsafe.Pointer(&actualOffsets[0]), int64(len(actualOffsets)*8))
	}

	// Compare offsets
	for i := range expectedOffsets {
		if expectedOffsets[i] != actualOffsets[i] {
			fmt.Printf("\n!!! OFFSET CORRUPTION DETECTED for %s %s !!!\n", name, context)
			fmt.Printf("Offset[%d]: expected %d, got %d\n", i, expectedOffsets[i], actualOffsets[i])
			fmt.Printf("Expected offsets: %v\n", expectedOffsets)
			fmt.Printf("Actual offsets: %v\n", actualOffsets)
			return fmt.Errorf("offset[%d] corrupted: expected %d, got %d",
				i, expectedOffsets[i], actualOffsets[i])
		}
	}

	return nil
}

// Key changes to runner.go:

// Remove AllocateAndSetArrays method entirely

// Remove old RunKernel variant that takes array names
// Keep only this version:
func (kr *Runner) RunKernel(kernelName string, scalarValues ...interface{}) error {
	// Get kernel definition
	def, exists := kr.kernelDefinitions[kernelName]
	if !exists {
		return fmt.Errorf("kernel %s not defined - use DefineKernel first", kernelName)
	}

	// Get compiled kernel
	kernel, exists := kr.Kernels[kernelName]
	if !exists {
		return fmt.Errorf("kernel %s not compiled", kernelName)
	}

	// Validate offsets before kernel execution
	for _, arrayName := range kr.GetAllocatedArrays() {
		if err := kr.validateOffsets(arrayName, "before kernel "+kernelName); err != nil {
			fmt.Printf("Pre-kernel validation failed: %v\n", err)
		}
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

	kr.Device.Finish()

	// Validate offsets after kernel execution
	for _, arrayName := range kr.GetAllocatedArrays() {
		if err := kr.validateOffsets(arrayName, "after kernel "+kernelName); err != nil {
			fmt.Printf("Post-kernel validation failed: %v\n", err)
		}
	}

	// Perform post-kernel data copies (device→host)
	if err := kr.performPostKernelCopies(def); err != nil {
		return fmt.Errorf("post-kernel copy failed: %w", err)
	}

	return nil
}

// Remove buildKernelArgs method entirely

// Make matrix methods internal (lowercase first letter)
func (kr *Runner) addStaticMatrix(name string, m mat.Matrix) {
	kr.Builder.AddStaticMatrix(name, m)
}

func (kr *Runner) addDeviceMatrix(name string, m mat.Matrix) {
	kr.Builder.AddDeviceMatrix(name, m)
}

// Update buildKernelArguments to only handle DefineKernel path
func (kr *Runner) buildKernelArguments(def *KernelDefinition, scalarValues []interface{}) ([]interface{}, error) {
	// Get ordered argument list for this definition
	kernelArgs := kr.GetKernelArgumentsForDefinition(def)
	args := make([]interface{}, 0, len(kernelArgs))

	// Build arguments in order
	for _, karg := range kernelArgs {
		switch karg.Category {
		case "system":
			mem, exists := kr.PooledMemory[karg.MemoryKey]
			if !exists {
				return nil, fmt.Errorf("system memory %s not found", karg.MemoryKey)
			}
			args = append(args, mem)

		case "matrix", "array_data", "array_offset":
			mem, exists := kr.PooledMemory[karg.MemoryKey]
			if !exists {
				return nil, fmt.Errorf("memory for %s not found", karg.MemoryKey)
			}
			args = append(args, mem)

		case "scalar":
			// Find the scalar parameter
			var found bool
			for _, p := range def.Parameters {
				if p.Direction == builder.DirectionScalar && p.Name == karg.Name {
					if p.HostBinding != nil {
						args = append(args, p.HostBinding)
						found = true
						break
					}
				}
			}

			// If not found in bindings, check passed values
			if !found {
				scalarIdx := 0
				for _, p := range def.Parameters {
					if p.Direction == builder.DirectionScalar {
						if p.Name == karg.Name {
							if scalarIdx < len(scalarValues) {
								args = append(args, scalarValues[scalarIdx])
								found = true
							}
							break
						}
						scalarIdx++
					}
				}
			}

			if !found {
				return nil, fmt.Errorf("scalar %s not provided", karg.Name)
			}
		}
	}

	return args, nil
}

func (kr *Runner) BuildKernel(kernelSource, kernelName string) (*gocca.OCCAKernel, error) {
	// NEW: Collect array types for preamble generation
	arrayTypes := kr.collectArrayTypes()

	// MODIFIED: Pass array types to GeneratePreamble
	kr.Builder.GeneratePreamble(kr.GetAllocatedArrays(), arrayTypes)

	// Combine preamble with kernel source
	fullSource := kr.KernelPreamble + "\n" + kernelSource

	// Build kernel with OpenMP optimization fix
	var kernel *gocca.OCCAKernel
	var err error

	if kr.Device.Mode() == "OpenMP" {
		// Workaround for OCCA bug: OpenMP doesn't get default -O3 flag
		props := gocca.JsonParse(`{"compiler_flags": "-O3"}`)
		defer props.Free()
		kernel, err = kr.Device.BuildKernelFromString(fullSource, kernelName, props)
	} else {
		// Other devices work correctly with default flags
		kernel, err = kr.Device.BuildKernelFromString(fullSource, kernelName, nil)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to build kernel %s: %w", kernelName, err)
	}

	// Register kernel
	if kernel != nil {
		kr.Kernels[kernelName] = kernel
		return kernel, nil
	}

	return nil, fmt.Errorf("kernel build returned nil for %s", kernelName)
}

// generatePreambleWithTypes is a helper method for testing/debugging
// NEW method to support Phase 2
func (kr *Runner) generatePreambleWithTypes() string {
	arrayTypes := kr.collectArrayTypes()
	return kr.Builder.GeneratePreamble(kr.GetAllocatedArrays(), arrayTypes)
}

// GetArrayType returns the data type of an allocated array
// Existing method - no changes needed
func (kr *Runner) GetArrayType(name string) (builder.DataType, error) {
	metadata, exists := kr.arrayMetadata[name]
	if !exists {
		return 0, fmt.Errorf("array %s not found", name)
	}
	return metadata.dataType, nil
}

// File: runner/runner.go
// Update to AllocateDeviceMatrices method

// AllocateDeviceMatrices allocates all device matrices that were added
// MODIFIED: Gets type from array metadata instead of using kr.FloatType
func (kr *Runner) AllocateDeviceMatrices() error {
	// Allocate each device matrix
	for name, matrix := range kr.DeviceMatrices {
		if _, exists := kr.PooledMemory[name]; exists {
			continue // Already allocated
		}

		rows, cols := matrix.Dims()
		totalElements := rows * cols

		// Get the type for this matrix from metadata
		// Default to Float64 if not specified
		matrixType := builder.Float64
		if meta, exists := kr.arrayMetadata[name]; exists {
			matrixType = meta.dataType
		}

		var mem *gocca.OCCAMemory
		if matrixType == builder.Float64 {
			// Copy and transpose for column-major storage
			transposed := make([]float64, totalElements)
			for i := 0; i < rows; i++ {
				for j := 0; j < cols; j++ {
					transposed[j*rows+i] = matrix.At(i, j)
				}
			}
			mem = kr.Device.Malloc(int64(totalElements*8), unsafe.Pointer(&transposed[0]), nil)
		} else {
			// Convert to float32 and transpose
			transposed := make([]float32, totalElements)
			for i := 0; i < rows; i++ {
				for j := 0; j < cols; j++ {
					transposed[j*rows+i] = float32(matrix.At(i, j))
				}
			}
			mem = kr.Device.Malloc(int64(totalElements*4), unsafe.Pointer(&transposed[0]), nil)
		}

		kr.PooledMemory[name] = mem
	}

	return nil
}

// File: runner/runner.go
// Fix collectArrayTypes method to remove FloatType references

// collectArrayTypes creates a map of array names to their effective data types
// MODIFIED: Defaults to Float64 instead of kr.FloatType
func (kr *Runner) collectArrayTypes() map[string]builder.DataType {
	arrayTypes := make(map[string]builder.DataType)

	// Collect types from arrayMetadata
	for name, meta := range kr.arrayMetadata {
		arrayTypes[name] = meta.dataType
	}

	// Collect types from static matrices
	for name := range kr.StaticMatrices {
		// Check if there's metadata with a specific type
		if meta, exists := kr.arrayMetadata[name]; exists {
			arrayTypes[name] = meta.dataType
		} else {
			// Default to Float64 for static matrices without metadata
			// This is the sensible default since gonum matrices are float64
			arrayTypes[name] = builder.Float64
		}
	}

	// Collect types from device matrices
	for name := range kr.DeviceMatrices {
		// Check if there's metadata with a specific type
		if meta, exists := kr.arrayMetadata[name]; exists {
			arrayTypes[name] = meta.dataType
		} else {
			// Default to Float64 for device matrices without metadata
			// This is the sensible default since gonum matrices are float64
			arrayTypes[name] = builder.Float64
		}
	}

	return arrayTypes
}

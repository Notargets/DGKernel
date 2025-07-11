package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/gocca"
	"gonum.org/v1/gonum/mat"
	"reflect"
	"unsafe"
)

type Runner struct {
	*builder.Builder
	arrayMetadata map[string]ArrayMetadata
	// Runtime resources
	Device       *gocca.OCCADevice
	Kernels      map[string]*gocca.OCCAKernel
	PooledMemory map[string]*gocca.OCCAMemory

	// Parameter API fields
	kernelDefinitions map[string]*KernelDefinition
	hostBindings      map[string]interface{}
	IsPartitioned     bool
	// Host-side offset storage for validation
	hostOffsets map[string][]int64 // Maps array name to its offsets
}

// ArrayMetadata tracks information about allocated arrays
type ArrayMetadata struct {
	spec     builder.ArraySpec
	dataType builder.DataType
	isOutput bool
}

// Modified NewRunner to initialize hostOffsets
func NewRunner(device *gocca.OCCADevice, Config builder.Config) (kr *Runner) {
	bld := builder.NewBuilder(Config)

	if bld.KpartMax > 1024 {
		panic(fmt.Sprintf("DGKernel doesn't support KpartMax > 1024, have %d. "+
			"Reduce partition sizes.", bld.KpartMax))
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

	// NEW: Validate offsets before kernel execution
	for _, arrayName := range kr.AllocatedArrays {
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

	// NEW: Validate offsets after kernel execution
	for _, arrayName := range kr.AllocatedArrays {
		if err := kr.validateOffsets(arrayName, "after kernel "+kernelName); err != nil {
			fmt.Printf("Post-kernel validation failed: %v\n", err)
			// This tells us the kernel is corrupting offset memory
		}
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
func (kr *Runner) buildKernelArguments(def *KernelDefinition,
	scalarValues []interface{}) ([]interface{}, error) {
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

// GetKernelSignature returns the generated signature for a defined kernel
func (kr *Runner) GetKernelSignature(kernelName string) (string, error) {
	def, exists := kr.kernelDefinitions[kernelName]
	if !exists {
		return "", fmt.Errorf("kernel %s not defined", kernelName)
	}
	return def.Signature, nil
}

// GetMemory returns the device memory handle for an array
func (kr *Runner) GetMemory(arrayName string) *gocca.OCCAMemory {
	if mem, exists := kr.PooledMemory[arrayName+"_global"]; exists {
		return mem
	}
	// Also check for non-array allocations (like device matrices)
	if mem, exists := kr.PooledMemory[arrayName]; exists {
		return mem
	}
	return nil
}

// GetOffsets returns the offset memory handle for an array
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

// GetArrayType returns the data type of an allocated array
func (kr *Runner) GetArrayType(name string) (builder.DataType, error) {
	metadata, exists := kr.arrayMetadata[name]
	if !exists {
		return 0, fmt.Errorf("array %s not found", name)
	}
	return metadata.dataType, nil
}

// GetArrayLogicalSize returns the number of logical elements in an array
func (kr *Runner) GetArrayLogicalSize(name string) (int, error) {
	metadata, exists := kr.arrayMetadata[name]
	if !exists {
		return 0, fmt.Errorf("array %s not found", name)
	}

	// Return the total size divided by the size of the data type
	var elementSize int64
	switch metadata.dataType {
	case builder.Float32, builder.INT32:
		elementSize = 4
	case builder.Float64, builder.INT64:
		elementSize = 8
	default:
		return 0, fmt.Errorf("unknown data type")
	}

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
func (kr *Runner) allocateSingleArray(spec builder.ArraySpec) error {
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
	kr.AllocatedArrays = append(kr.AllocatedArrays, spec.Name)
	kr.arrayMetadata[spec.Name] = ArrayMetadata{
		spec:     spec,
		dataType: spec.DataType,
		isOutput: spec.IsOutput,
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
			fmt.Printf("Position %d: expected %d, got %d (diff: %d)\n",
				i, expectedOffsets[i], actualOffsets[i], actualOffsets[i]-expectedOffsets[i])
			fmt.Printf("Full arrays:\n")
			fmt.Printf("  Expected: %v\n", expectedOffsets)
			fmt.Printf("  Actual:   %v\n", actualOffsets)

			return fmt.Errorf("offset corruption at position %d: expected %d, got %d",
				i, expectedOffsets[i], actualOffsets[i])
		}
	}

	return nil
}

// AllocateDeviceMatrices allocates device memory for all device matrices
func (kr *Runner) AllocateDeviceMatrices() error {
	for name, matrix := range kr.Builder.DeviceMatrices {
		err := kr.allocateDeviceMatrix(name, matrix)
		if err != nil {
			return fmt.Errorf("failed to allocate device matrix %s: %w", name, err)
		}
	}
	return nil
}

// allocateDeviceMatrix handles allocation of a single device matrix
func (kr *Runner) allocateDeviceMatrix(name string, matrix mat.Matrix) error {
	rows, cols := matrix.Dims()

	// Convert matrix to column-major flat array
	data := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			// Transpose during copy: write to column-major position
			data[j*rows+i] = matrix.At(i, j)
		}
	}

	// Handle float32 conversion if needed
	var mem *gocca.OCCAMemory
	if kr.FloatType == builder.Float32 {
		data32 := make([]float32, len(data))
		for i, v := range data {
			data32[i] = float32(v)
		}
		size := int64(len(data32) * 4) // 4 bytes per float32
		mem = kr.Device.Malloc(size, unsafe.Pointer(&data32[0]), nil)
	} else {
		size := int64(len(data) * 8) // 8 bytes per float64
		mem = kr.Device.Malloc(size, unsafe.Pointer(&data[0]), nil)
	}

	// Store in pooled memory for automatic cleanup
	kr.PooledMemory[name] = mem

	return nil
}

// BuildKernel compiles and registers a kernel with the program
func (kr *Runner) BuildKernel(kernelSource, kernelName string) (*gocca.OCCAKernel, error) {
	kr.GeneratePreamble()

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

// CopyArrayToHost copies array data from device to host, removing alignment padding
func CopyArrayToHost[T any](kr *Runner, name string) ([]T, error) {
	// Check if array exists
	metadata, exists := kr.arrayMetadata[name]
	if !exists {
		return nil, fmt.Errorf("array %s not found", name)
	}

	// Verify type matches
	var sample T
	requestedType := getDataTypeFromSample(sample)
	if requestedType != metadata.dataType {
		return nil, fmt.Errorf("type mismatch: array is %v, requested %v",
			metadata.dataType, requestedType)
	}

	// Get memory and offsets
	memory := kr.GetMemory(name)
	if memory == nil {
		return nil, fmt.Errorf("memory for %s not found", name)
	}

	offsetsMem := kr.PooledMemory[name+"_offsets"]
	if offsetsMem == nil {
		return nil, fmt.Errorf("offsets for %s not found", name)
	}

	// Read offsets to determine actual data locations
	numOffsets := kr.NumPartitions + 1
	var offsets []int64

	if kr.GetIntSize() == 4 {
		offsets32 := make([]int32, numOffsets)
		offsetsMem.CopyTo(unsafe.Pointer(&offsets32[0]), int64(numOffsets*4))
		offsets = make([]int64, numOffsets)
		for i, v := range offsets32 {
			offsets[i] = int64(v)
		}
	} else {
		offsets = make([]int64, numOffsets)
		offsetsMem.CopyTo(unsafe.Pointer(&offsets[0]), int64(numOffsets*8))
	}

	// Calculate total elements (sum of K values)
	totalElements := kr.GetTotalElements()
	result := make([]T, totalElements)

	// Get element size
	elementSize := int64(unsafe.Sizeof(sample))

	// Copy each partition contiguously
	destIndex := 0
	for i := 0; i < kr.NumPartitions; i++ {
		partitionElements := kr.K[i]
		partitionBytes := int64(partitionElements) * elementSize
		sourceOffset := offsets[i] * elementSize

		// Copy this partition's data
		memory.CopyToWithOffset(
			unsafe.Pointer(&result[destIndex]),
			partitionBytes,
			sourceOffset,
		)

		destIndex += partitionElements
	}

	return result, nil
}

// CopyPartitionToHost copies a specific partition's data from device to host
func CopyPartitionToHost[T any](kr *Runner, name string, partitionID int) ([]T, error) {
	if partitionID < 0 || partitionID >= kr.NumPartitions {
		return nil, fmt.Errorf("invalid partition ID: %d", partitionID)
	}

	// Check if array exists
	metadata, exists := kr.arrayMetadata[name]
	if !exists {
		return nil, fmt.Errorf("array %s not found", name)
	}

	// Verify type matches
	var sample T
	requestedType := getDataTypeFromSample(sample)
	if requestedType != metadata.dataType {
		return nil, fmt.Errorf("type mismatch: array is %v, requested %v",
			metadata.dataType, requestedType)
	}

	// Get memory and offsets
	memory := kr.GetMemory(name)
	if memory == nil {
		return nil, fmt.Errorf("memory for %s not found", name)
	}

	offsetsMem := kr.PooledMemory[name+"_offsets"]
	if offsetsMem == nil {
		return nil, fmt.Errorf("offsets for %s not found", name)
	}

	// Read just the offsets we need
	var partitionOffset int64
	if kr.GetIntSize() == 4 {
		offsets32 := make([]int32, 2)
		offsetsBytes := int64(8) // 2 * 4 bytes
		sourceOffset := int64(partitionID * 4)
		offsetsMem.CopyToWithOffset(unsafe.Pointer(&offsets32[0]), offsetsBytes, sourceOffset)
		partitionOffset = int64(offsets32[0])
	} else {
		offsets := make([]int64, 2)
		offsetsBytes := int64(16) // 2 * 8 bytes
		sourceOffset := int64(partitionID * 8)
		offsetsMem.CopyToWithOffset(unsafe.Pointer(&offsets[0]), offsetsBytes, sourceOffset)
		partitionOffset = offsets[0]
	}

	// Create result array for this partition
	partitionElements := kr.K[partitionID]
	result := make([]T, partitionElements)

	// Copy partition data
	elementSize := int64(unsafe.Sizeof(sample))
	partitionBytes := int64(partitionElements) * elementSize
	sourceOffset := partitionOffset * elementSize

	memory.CopyToWithOffset(
		unsafe.Pointer(&result[0]),
		partitionBytes,
		sourceOffset,
	)

	return result, nil
}

// Helper function to get DataType from sample
func getDataTypeFromSample(sample interface{}) builder.DataType {
	switch sample.(type) {
	case float32:
		return builder.Float32
	case float64:
		return builder.Float64
	case int32:
		return builder.INT32
	case int64:
		return builder.INT64
	default:
		return 0
	}
}

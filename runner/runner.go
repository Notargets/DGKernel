package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/builder"
	"github.com/notargets/gocca"
	"unsafe"
)

type Runner struct {
	*builder.Builder
	arrayMetadata map[string]ArrayMetadata
	// Runtime resources
	Device       *gocca.OCCADevice
	Kernels      map[string]*gocca.OCCAKernel
	PooledMemory map[string]*gocca.OCCAMemory
}

// ArrayMetadata tracks information about allocated arrays
type ArrayMetadata struct {
	spec     builder.ArraySpec
	dataType builder.DataType
}

// func NewRunner(device *gocca.OCCADevice, bld *builder.Builder) (kr *Runner) {
func NewRunner(device *gocca.OCCADevice, Config builder.Config) (kr *Runner) {
	if device == nil {
		panic("Device cannot be nil")
	}
	bld := builder.NewBuilder(Config)
	// Check CUDA @inner limit
	if device.Mode() == "CUDA" && bld.KpartMax > 1024 {
		panic(fmt.Sprintf("CUDA @inner limit exceeded: KpartMax=%d but CUDA"+
			" is limited to 1024 threads per @inner loop. "+
			"Reduce partition sizes.", bld.KpartMax))
	}
	// Allocate K array on Device
	intSize := 8 // INT64
	if bld.IntType == builder.INT32 {
		intSize = 4
	}
	kr = &Runner{
		Builder:       bld,
		arrayMetadata: make(map[string]ArrayMetadata),
		Device:        device,
		Kernels:       make(map[string]*gocca.OCCAKernel),
		PooledMemory:  make(map[string]*gocca.OCCAMemory),
	}
	kMem := device.Malloc(int64(len(bld.K)*intSize), unsafe.Pointer(&bld.K[0]),
		nil)
	kr.PooledMemory["K"] = kMem
	return
}

// RunKernel executes a registered kernel with the given arguments
func (kr *Runner) RunKernel(name string, args ...interface{}) error {
	kernel, exists := kr.Kernels[name]
	if !exists {
		return fmt.Errorf("kernel %s not found", name)
	}

	// Expand args to include renamed arrays
	expandedArgs := kr.expandKernelArgs(args)

	return kernel.RunWithArgs(expandedArgs...)
}

// expandKernelArgs transforms user array names to kernel parameter names
func (kr *Runner) expandKernelArgs(args []interface{}) []interface{} {
	expanded := []interface{}{}

	// Always pass K array first
	expanded = append(expanded, kr.PooledMemory["K"])

	// Process remaining arguments
	for _, arg := range args {
		switch v := arg.(type) {
		case string:
			// Array name - expand to global and offsets
			globalMem, hasGlobal := kr.PooledMemory[v+"_global"]
			offsetMem, hasOffset := kr.PooledMemory[v+"_offsets"]

			if hasGlobal && hasOffset {
				expanded = append(expanded, globalMem, offsetMem)
			} else {
				// Pass through if not a recognized array
				expanded = append(expanded, arg)
			}
		default:
			// Pass through non-string arguments
			expanded = append(expanded, arg)
		}
	}

	return expanded
}

// GetMemory returns the device memory handle for an array
func (kr *Runner) GetMemory(arrayName string) *gocca.OCCAMemory {
	if mem, exists := kr.PooledMemory[arrayName+"_global"]; exists {
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
		offsetsMem.CopyTo(unsafe.Pointer(&offsets[0]), int64(numOffsets*kr.GetIntSize()))
	}

	// Calculate total logical size
	logicalSize, err := kr.GetArrayLogicalSize(name)
	if err != nil {
		return nil, err
	}

	// Allocate result without padding
	result := make([]T, logicalSize)

	// Calculate elements per partition
	totalElements := kr.GetTotalElements()
	elementsPerValue := logicalSize / totalElements

	// Copy each partition's data
	destOffset := 0
	for i := 0; i < kr.NumPartitions; i++ {
		partitionElements := kr.K[i] * elementsPerValue
		partitionBytes := int64(partitionElements) * int64(unsafe.Sizeof(sample))

		// Read partition data directly to result slice
		srcOffset := offsets[i] * int64(unsafe.Sizeof(sample))
		memory.CopyToWithOffset(unsafe.Pointer(&result[destOffset]), partitionBytes, srcOffset)

		destOffset += partitionElements
	}

	return result, nil
}

// CopyPartitionToHost copies a single partition's data from device to host
func CopyPartitionToHost[T any](kr *Runner, name string, partitionID int) ([]T, error) {
	if partitionID < 0 || partitionID >= kr.NumPartitions {
		return nil, fmt.Errorf("invalid partition ID: %d (must be 0-%d)", partitionID, kr.NumPartitions-1)
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

	// Handle empty partition case - return empty slice immediately
	if kr.K[partitionID] == 0 {
		return make([]T, 0), nil
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

	// Read just the offsets we need (partition and partition+1)
	var partitionStart int64

	if kr.GetIntSize() == 4 {
		offsets32 := make([]int32, 2)
		srcOffset := int64(partitionID * 4)
		offsetsMem.CopyToWithOffset(unsafe.Pointer(&offsets32[0]), 8, srcOffset)
		partitionStart = int64(offsets32[0])
	} else {
		offsets := make([]int64, 2)
		srcOffset := int64(partitionID * 8)
		offsetsMem.CopyToWithOffset(unsafe.Pointer(&offsets[0]), 16, srcOffset)
		partitionStart = offsets[0]
	}

	// Calculate partition size
	logicalSize, err := kr.GetArrayLogicalSize(name)
	if err != nil {
		return nil, err
	}

	totalElements := kr.GetTotalElements()
	elementsPerValue := logicalSize / totalElements
	partitionElements := kr.K[partitionID] * elementsPerValue

	// Allocate result
	result := make([]T, partitionElements)

	// Copy partition data
	partitionBytes := int64(partitionElements) * int64(unsafe.Sizeof(sample))
	srcOffset := partitionStart * int64(unsafe.Sizeof(sample))
	memory.CopyToWithOffset(unsafe.Pointer(&result[0]), partitionBytes, srcOffset)

	return result, nil
}

// getDataTypeFromSample infers DataType from a sample value
func getDataTypeFromSample[T any](sample T) builder.DataType {
	switch any(sample).(type) {
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

// AllocateArrays allocates Device memory with automatic offset calculation
func (kr *Runner) AllocateArrays(specs []builder.ArraySpec) error {
	for _, spec := range specs {
		if err := kr.allocateSingleArray(spec); err != nil {
			return fmt.Errorf("failed to allocate %s: %w", spec.Name, err)
		}
	}
	return nil
}

// allocateSingleArray handles allocation of a single array
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

	// Track allocation
	kr.AllocatedArrays = append(kr.AllocatedArrays, spec.Name)
	kr.arrayMetadata[spec.Name] = ArrayMetadata{
		spec:     spec,
		dataType: spec.DataType,
	}

	return nil
}

// BuildKernel compiles and registers a kernel with the program
func (kr *Runner) BuildKernel(kernelSource, kernelName string) (*gocca.OCCAKernel, error) {
	// Generate preamble if not already done
	if kr.KernelPreamble == "" {
		kr.GeneratePreamble()
	}

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

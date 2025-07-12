package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"github.com/notargets/gocca"
	"gonum.org/v1/gonum/mat"
	"strings"
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
	SortStrings(deviceMatrixNames)

	for _, name := range deviceMatrixNames {
		if mem, exists := kr.PooledMemory[name]; exists {
			args = append(args, mem)
		} else {
			return nil, fmt.Errorf("device matrix %s not found", name)
		}
	}

	// Add arrays
	for _, p := range def.Parameters {
		if p.Direction == builder.DirectionScalar || p.IsMatrix {
			continue // Skip scalars and matrices
		}

		// Add global data pointer
		globalMem, exists := kr.PooledMemory[p.Name+"_global"]
		if !exists {
			return nil, fmt.Errorf("memory for %s not found", p.Name)
		}
		args = append(args, globalMem)

		// Add offset array
		offsetMem, exists := kr.PooledMemory[p.Name+"_offsets"]
		if !exists {
			return nil, fmt.Errorf("offsets for %s not found", p.Name)
		}
		args = append(args, offsetMem)
	}

	// Add scalars last
	scalarIdx := 0
	for _, p := range def.Parameters {
		if p.Direction == builder.DirectionScalar {
			if scalarIdx >= len(scalarValues) {
				// Use bound value if available
				if p.HostBinding != nil {
					args = append(args, p.HostBinding)
				} else {
					return nil, fmt.Errorf("no value provided for scalar %s", p.Name)
				}
			} else {
				args = append(args, scalarValues[scalarIdx])
				scalarIdx++
			}
		}
	}

	return args, nil
}

// GetKernelSignature generates the signature for a defined kernel
func (kr *Runner) GetKernelSignature(kernelName string) (string, error) {
	def, exists := kr.kernelDefinitions[kernelName]
	if !exists {
		return "", fmt.Errorf("kernel %s not defined", kernelName)
	}

	args := kr.GetKernelArgumentsForDefinition(def)
	params := make([]string, 0, len(args))

	for _, karg := range args {
		if karg.Category == "scalar" {
			// Scalars have explicit const in the type
			params = append(params, fmt.Sprintf("const %s %s", karg.Type, karg.Name))
		} else {
			// Arrays and matrices
			constStr := ""
			if karg.IsConst {
				constStr = "const "
			}
			params = append(params, fmt.Sprintf("%s%s %s", constStr, karg.Type, karg.Name))
		}
	}

	return strings.Join(params, ",\n\t"), nil
}

// AddStaticMatrix adds a matrix that will be embedded as static const in kernels
func (kr *Runner) AddStaticMatrix(name string, m mat.Matrix) {
	kr.Builder.AddStaticMatrix(name, m)
}

// AddDeviceMatrix adds a matrix that will be allocated in device global memory
func (kr *Runner) AddDeviceMatrix(name string, m mat.Matrix) {
	kr.Builder.AddDeviceMatrix(name, m)
}

// AllocateDeviceMatrices allocates all device matrices that were added
func (kr *Runner) AllocateDeviceMatrices() error {
	// Allocate each device matrix
	for name, matrix := range kr.DeviceMatrices {
		if _, exists := kr.PooledMemory[name]; exists {
			continue // Already allocated
		}

		rows, cols := matrix.Dims()
		totalElements := rows * cols

		var mem *gocca.OCCAMemory
		if kr.FloatType == builder.Float64 {
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
	kr.AllocatedArrays = append(kr.AllocatedArrays, spec.Name)
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

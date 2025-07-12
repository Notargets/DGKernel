package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
	"sort"
	"strings"
)

// KernelArgument represents a single kernel argument with metadata
type KernelArgument struct {
	Name        string
	Type        string // "int_t*", "real_t*"
	MemoryKey   string // Key in PooledMemory map
	IsConst     bool
	Category    string // "system", "matrix", "array_data", "array_offset"
	UserArgName string // For user arrays, the base name (e.g., "U" for "U_global")
}

// GetKernelArguments returns the ordered list of kernel arguments
// This is the SINGLE SOURCE OF TRUTH for kernel argument ordering
func (kr *Runner) GetKernelArguments() []KernelArgument {
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
	deviceMatrixNames := make([]string, 0, len(kr.DeviceMatrices))
	for name := range kr.DeviceMatrices {
		deviceMatrixNames = append(deviceMatrixNames, name)
	}
	sort.Strings(deviceMatrixNames)

	for _, name := range deviceMatrixNames {
		args = append(args, KernelArgument{
			Name:      name,
			Type:      "real_t*",
			MemoryKey: name,
			IsConst:   true,
			Category:  "matrix",
		})
	}

	// 3. Allocated arrays (global pointer and offsets for each)
	for _, arrayName := range kr.GetAllocatedArrays() {
		// Get the actual metadata instead of guessing
		metadata, exists := kr.arrayMetadata[arrayName]
		if !exists {
			panic(fmt.Sprintf("Array metadata not found for %s", arrayName))
		}

		// Global data pointer
		args = append(args, KernelArgument{
			Name:        arrayName + "_global",
			Type:        "real_t*",
			MemoryKey:   arrayName + "_global",
			IsConst:     !metadata.isOutput, // Use explicit flag
			Category:    "array_data",
			UserArgName: arrayName,
		})

		// Offset array
		args = append(args, KernelArgument{
			Name:        arrayName + "_offsets",
			Type:        "int_t*",
			MemoryKey:   arrayName + "_offsets",
			IsConst:     true,
			Category:    "array_offset",
			UserArgName: arrayName,
		})
	}

	return args
}

// GenerateKernelSignature generates the parameter list for kernel functions
// This is used for the old-style kernel building approach
func (kr *Runner) GenerateKernelSignature() string {
	kernelArgs := kr.GetKernelArguments()
	params := make([]string, 0, len(kernelArgs))

	for _, karg := range kernelArgs {
		constStr := ""
		if karg.IsConst {
			constStr = "const "
		}
		params = append(params, fmt.Sprintf("%s%s %s", constStr, karg.Type, karg.Name))
	}

	return strings.Join(params, ",\n\t")
}

// GetExpectedUserArguments returns the list of array names the kernel expects
func (kr *Runner) GetExpectedUserArguments() []string {
	seen := make(map[string]bool)
	var result []string

	for _, karg := range kr.GetKernelArguments() {
		if karg.UserArgName != "" && !seen[karg.UserArgName] {
			seen[karg.UserArgName] = true
			result = append(result, karg.UserArgName)
		}
	}

	return result
}

// DebugKernelArguments prints detailed information about kernel arguments
func (kr *Runner) DebugKernelArguments() {
	fmt.Println("=== Kernel Arguments Debug Info ===")
	for i, karg := range kr.GetKernelArguments() {
		fmt.Printf("%2d: %-20s %-10s %-8s mem_key=%-20s user_arg=%s\n",
			i, karg.Name, karg.Type, karg.Category, karg.MemoryKey, karg.UserArgName)

		// Check if memory exists
		if _, exists := kr.PooledMemory[karg.MemoryKey]; !exists {
			fmt.Printf("    WARNING: Memory key '%s' not found in PooledMemory!\n", karg.MemoryKey)
		}
	}
	fmt.Println("===================================")
}

// GetKernelArgumentsForDefinition returns kernel arguments using parameter specs
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

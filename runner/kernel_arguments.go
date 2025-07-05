package runner

import (
	"fmt"
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
	for _, arrayName := range kr.AllocatedArrays {
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

// expandKernelArgs transforms user array names to kernel parameter names
// REWRITTEN to use GetKernelArguments for consistency
func (kr *Runner) expandKernelArgs(userArgs []interface{}) []interface{} {
	kernelArgs := kr.GetKernelArguments()
	expanded := make([]interface{}, 0, len(kernelArgs))

	// Create a map of user arg names to track what's been provided
	userArgSet := make(map[string]bool)
	for _, arg := range userArgs {
		if argName, ok := arg.(string); ok {
			userArgSet[argName] = true
		}
	}

	// Build the expanded argument list in the correct order
	for _, karg := range kernelArgs {
		// System and matrix arguments are always included
		if karg.Category == "system" || karg.Category == "matrix" {
			mem, exists := kr.PooledMemory[karg.MemoryKey]
			if !exists {
				panic(fmt.Sprintf("Required memory '%s' not found in PooledMemory", karg.MemoryKey))
			}
			expanded = append(expanded, mem)
			continue
		}

		// For array arguments, only include if the user specified the array
		if karg.UserArgName != "" && userArgSet[karg.UserArgName] {
			mem, exists := kr.PooledMemory[karg.MemoryKey]
			if !exists {
				panic(fmt.Sprintf("Memory for array '%s' (key: %s) not found", karg.UserArgName, karg.MemoryKey))
			}
			expanded = append(expanded, mem)
		}
	}

	// Add any non-string arguments at the end (for backward compatibility)
	for _, arg := range userArgs {
		if _, isString := arg.(string); !isString {
			expanded = append(expanded, arg)
		}
	}

	return expanded
}

// GenerateKernelSignature generates the parameter list for kernel functions
// REWRITTEN to use GetKernelArguments for consistency
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

// ValidateKernelArguments checks if the user arguments match what the kernel expects
func (kr *Runner) ValidateKernelArguments(userArgs []interface{}) error {
	kernelArgs := kr.GetKernelArguments()

	// Create set of expected user arguments
	expectedUserArgs := make(map[string]bool)
	for _, karg := range kernelArgs {
		if karg.UserArgName != "" {
			expectedUserArgs[karg.UserArgName] = true
		}
	}

	// Check each user argument
	for _, arg := range userArgs {
		if argName, ok := arg.(string); ok {
			if !expectedUserArgs[argName] {
				// Check if it's a valid memory key directly
				if _, exists := kr.PooledMemory[argName]; !exists {
					return fmt.Errorf("unknown array argument: %s", argName)
				}
			}
		}
	}

	return nil
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

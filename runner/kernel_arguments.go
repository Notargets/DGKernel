package runner

import (
	"github.com/notargets/DGKernel/runner/builder"
	"sort"
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

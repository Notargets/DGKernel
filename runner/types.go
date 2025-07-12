// runner/types.go
package runner

import (
	"github.com/notargets/DGKernel/runner/builder"
	"sort"
)

// SizeOfType returns the size in bytes of a data type
func SizeOfType(dt builder.DataType) int64 {
	switch dt {
	case builder.Float32, builder.INT32:
		return 4
	case builder.Float64, builder.INT64:
		return 8
	default:
		return 8
	}
}

// TypeName returns the C type name for a given DataType
func TypeName(dt builder.DataType, isReal bool) string {
	if isReal {
		// For real_t type
		switch dt {
		case builder.Float32:
			return "float"
		case builder.Float64:
			return "double"
		default:
			return "double"
		}
	} else {
		// For int_t type
		switch dt {
		case builder.INT32:
			return "int"
		case builder.INT64:
			return "long"
		default:
			return "long"
		}
	}
}

// TypeSuffix returns the numeric suffix for floating point literals
func TypeSuffix(dt builder.DataType) string {
	if dt == builder.Float32 {
		return "f"
	}
	return ""
}

// GetScalarTypeName returns the C type name for scalar parameters
func GetScalarTypeName(dt builder.DataType) string {
	switch dt {
	case builder.Float32:
		return "float"
	case builder.Float64:
		return "double"
	case builder.INT32:
		return "int"
	case builder.INT64:
		return "long"
	default:
		// Default based on whether it seems like an int or float type
		if dt == builder.INT32 || dt == builder.INT64 {
			return "int"
		}
		return "double"
	}
}

// SortStrings is a simple wrapper around sort.Strings
// This replaces all the duplicate bubble sort implementations
func SortStrings(strings []string) {
	sort.Strings(strings)
}

// GetDataTypeFromSample returns the DataType based on a sample value
func GetDataTypeFromSample(sample interface{}) builder.DataType {
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

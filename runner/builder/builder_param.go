package builder

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"reflect"
)

// Direction indicates parameter data flow
type Direction int

const (
	DirectionInput Direction = iota
	DirectionOutput
	DirectionInOut
	DirectionTemp
	DirectionScalar
)

// ParamBuilder provides a fluent interface for building kernel parameters
type ParamBuilder struct {
	Spec ParamSpec
}

// ParamSpec holds the complete specification for a kernel parameter
type ParamSpec struct {
	Name        string
	Direction   Direction
	HostBinding interface{}

	// Type and size (inferred or explicit)
	DataType DataType
	Size     int64

	// Data movement
	DoCopyTo    bool
	DoCopyBack  bool
	ConvertType DataType // 0 means no conversion

	// Memory attributes
	Alignment AlignmentType

	// Matrix attributes
	IsMatrix   bool
	IsStatic   bool
	MatrixRows int
	MatrixCols int
	Stride     int // For flat array to matrix promotion

	// NEW: Partitioned data support
	IsPartitioned  bool
	PartitionCount int
}

// Input creates a parameter specification for a const input
func Input(deviceName string) *ParamBuilder {
	return &ParamBuilder{
		Spec: ParamSpec{
			Name:      deviceName,
			Direction: DirectionInput,
		},
	}
}

// Output creates a parameter specification for a non-const output
func Output(deviceName string) *ParamBuilder {
	return &ParamBuilder{
		Spec: ParamSpec{
			Name:      deviceName,
			Direction: DirectionOutput,
		},
	}
}

// InOut creates a parameter specification for a non-const input/output
func InOut(deviceName string) *ParamBuilder {
	return &ParamBuilder{
		Spec: ParamSpec{
			Name:      deviceName,
			Direction: DirectionInOut,
		},
	}
}

// Scalar creates a parameter specification for a scalar value
func Scalar(deviceName string) *ParamBuilder {
	return &ParamBuilder{
		Spec: ParamSpec{
			Name:      deviceName,
			Direction: DirectionScalar,
		},
	}
}

// Temp creates a parameter specification for a device-only temporary array
func Temp(deviceName string) *ParamBuilder {
	return &ParamBuilder{
		Spec: ParamSpec{
			Name:      deviceName,
			Direction: DirectionTemp,
		},
	}
}

// Bind associates a host variable with this parameter
func (p *ParamBuilder) Bind(hostVar interface{}) *ParamBuilder {
	p.Spec.HostBinding = hostVar

	// Infer type and size if possible
	p.inferFromBinding()

	return p
}

// Copy sets bidirectional copy (host→device before, device→host after)
func (p *ParamBuilder) Copy() *ParamBuilder {
	p.Spec.DoCopyTo = true
	p.Spec.DoCopyBack = true
	return p
}

// CopyTo sets host→device copy before kernel execution
func (p *ParamBuilder) CopyTo() *ParamBuilder {
	p.Spec.DoCopyTo = true
	return p
}

// CopyBack sets device→host copy after kernel execution
func (p *ParamBuilder) CopyBack() *ParamBuilder {
	p.Spec.DoCopyBack = true
	return p
}

// NoCopy explicitly disables data movement
func (p *ParamBuilder) NoCopy() *ParamBuilder {
	p.Spec.DoCopyTo = false
	p.Spec.DoCopyBack = false
	return p
}

// Convert sets type conversion during copy operations
func (p *ParamBuilder) Convert(toType DataType) *ParamBuilder {
	p.Spec.ConvertType = toType
	return p
}

// Type sets explicit type (mainly for Temp arrays)
func (p *ParamBuilder) Type(dataType DataType) *ParamBuilder {
	p.Spec.DataType = dataType
	return p
}

// Size sets explicit size (mainly for Temp arrays)
func (p *ParamBuilder) Size(elements int) *ParamBuilder {
	p.Spec.Size = int64(elements)
	return p
}

// ToMatrix marks this parameter as a matrix, enabling MATMUL macro generation
func (p *ParamBuilder) ToMatrix() *ParamBuilder {
	p.Spec.IsMatrix = true

	// If bound to a mat.Matrix, extract dimensions
	if p.Spec.HostBinding != nil {
		if m, ok := p.Spec.HostBinding.(mat.Matrix); ok {
			p.Spec.MatrixRows, p.Spec.MatrixCols = m.Dims()
		}
	}

	return p
}

// Static marks a matrix for static embedding (const array in kernel)
func (p *ParamBuilder) Static() *ParamBuilder {
	p.Spec.IsStatic = true
	return p
}

// Stride sets the stride for promoting flat arrays to matrices
func (p *ParamBuilder) Stride(stride int) *ParamBuilder {
	p.Spec.Stride = stride
	p.Spec.MatrixRows = int(p.Spec.Size) / stride
	p.Spec.MatrixCols = stride
	return p
}

// Align sets memory alignment requirements
func (p *ParamBuilder) Align(alignment AlignmentType) *ParamBuilder {
	p.Spec.Alignment = alignment
	return p
}

// runner/builder/builder_param.go - Update inferFromBinding to detect partitioned data
func (p *ParamBuilder) inferFromBinding() {
	if p.Spec.HostBinding == nil {
		return
	}

	v := reflect.ValueOf(p.Spec.HostBinding)
	t := v.Type()

	// NEW: Check for partitioned arrays (slice of slices)
	if t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Slice {
		p.Spec.IsPartitioned = true
		p.Spec.PartitionCount = v.Len()

		// Calculate total size
		totalSize := int64(0)
		for i := 0; i < v.Len(); i++ {
			partition := v.Index(i)
			totalSize += int64(partition.Len())
		}
		p.Spec.Size = totalSize

		// Infer element type from first partition
		if v.Len() > 0 {
			elemType := t.Elem().Elem()
			switch elemType.Kind() {
			case reflect.Float32:
				p.Spec.DataType = Float32
			case reflect.Float64:
				p.Spec.DataType = Float64
			case reflect.Int32:
				p.Spec.DataType = INT32
			case reflect.Int64:
				p.Spec.DataType = INT64
			}
		}
		return
	}

	// NEW: Check for partitioned matrices (slice of mat.Matrix)
	if t.Kind() == reflect.Slice && v.Len() > 0 {
		if _, ok := v.Index(0).Interface().(mat.Matrix); ok {
			p.Spec.IsPartitioned = true
			p.Spec.PartitionCount = v.Len()

			// Calculate total size and get dimensions
			totalSize := int64(0)
			for i := 0; i < v.Len(); i++ {
				if m, ok := v.Index(i).Interface().(mat.Matrix); ok {
					rows, cols := m.Dims()
					if i == 0 {
						p.Spec.MatrixRows = rows
						p.Spec.MatrixCols = cols
					}
					totalSize += int64(rows * cols)
				}
			}
			p.Spec.Size = totalSize
			p.Spec.DataType = Float64
			return
		}
	}

	// EXISTING CODE for non-partitioned data remains unchanged
	// Handle slices
	if t.Kind() == reflect.Slice {
		p.Spec.Size = int64(v.Len())

		// Infer element type
		elemType := t.Elem()
		switch elemType.Kind() {
		case reflect.Float32:
			p.Spec.DataType = Float32
		case reflect.Float64:
			p.Spec.DataType = Float64
		case reflect.Int32:
			p.Spec.DataType = INT32
		case reflect.Int64:
			p.Spec.DataType = INT64
		}
		return
	}

	// Handle mat.Matrix
	if m, ok := p.Spec.HostBinding.(mat.Matrix); ok {
		rows, cols := m.Dims()
		p.Spec.Size = int64(rows * cols)
		p.Spec.DataType = Float64 // gonum matrices are float64
		p.Spec.MatrixRows = rows
		p.Spec.MatrixCols = cols
		return
	}

	// Handle scalars
	switch t.Kind() {
	case reflect.Float32:
		p.Spec.DataType = Float32
		p.Spec.Size = 1
	case reflect.Float64:
		p.Spec.DataType = Float64
		p.Spec.Size = 1
	case reflect.Int, reflect.Int64:
		p.Spec.DataType = INT64
		p.Spec.Size = 1
	case reflect.Int32:
		p.Spec.DataType = INT32
		p.Spec.Size = 1
	}
}

// runner/builder/builder_param.go - Update Validate to check partitioned constraints
func (p *ParamSpec) Validate() error {
	// EXISTING validation code...

	// NEW: Add partitioned data validation
	if p.IsPartitioned {
		if p.IsMatrix && p.IsStatic {
			return fmt.Errorf("static matrices cannot be partitioned")
		}
		if p.PartitionCount == 0 {
			return fmt.Errorf("partitioned data %s has no partitions", p.Name)
		}
	}

	return nil
}

// IsConst returns whether this parameter should be const in the kernel signature
func (p *ParamSpec) IsConst() bool {
	switch p.Direction {
	case DirectionInput, DirectionScalar:
		return true
	case DirectionOutput, DirectionInOut, DirectionTemp:
		return false
	default:
		return true
	}
}

// NeedsCopyTo returns whether this parameter needs host→device copy
func (p *ParamSpec) NeedsCopyTo() bool {
	return p.DoCopyTo && p.HostBinding != nil
}

// NeedsCopyBack returns whether this parameter needs device→host copy
func (p *ParamSpec) NeedsCopyBack() bool {
	return p.DoCopyBack && p.HostBinding != nil
}

// GetEffectiveType returns the type to use on device (considering conversion)
func (p *ParamSpec) GetEffectiveType() DataType {
	if p.ConvertType != 0 {
		return p.ConvertType
	}
	return p.DataType
}

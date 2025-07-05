package runner

import (
	"fmt"
	"github.com/notargets/DGKernel/runner/builder"
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
	spec ParamSpec
}

// ParamSpec holds the complete specification for a kernel parameter
type ParamSpec struct {
	Name        string
	Direction   Direction
	HostBinding interface{}

	// Type and size (inferred or explicit)
	DataType builder.DataType
	Size     int64

	// Data movement
	DoCopyTo    bool
	DoCopyBack  bool
	ConvertType builder.DataType // 0 means no conversion

	// Memory attributes
	Alignment builder.AlignmentType

	// Matrix attributes
	IsMatrix   bool
	IsStatic   bool
	MatrixRows int
	MatrixCols int
	Stride     int // For flat array to matrix promotion
}

// Input creates a parameter specification for a const input
func Input(deviceName string) *ParamBuilder {
	return &ParamBuilder{
		spec: ParamSpec{
			Name:      deviceName,
			Direction: DirectionInput,
		},
	}
}

// Output creates a parameter specification for a non-const output
func Output(deviceName string) *ParamBuilder {
	return &ParamBuilder{
		spec: ParamSpec{
			Name:      deviceName,
			Direction: DirectionOutput,
		},
	}
}

// InOut creates a parameter specification for a non-const input/output
func InOut(deviceName string) *ParamBuilder {
	return &ParamBuilder{
		spec: ParamSpec{
			Name:      deviceName,
			Direction: DirectionInOut,
		},
	}
}

// Scalar creates a parameter specification for a scalar value
func Scalar(deviceName string) *ParamBuilder {
	return &ParamBuilder{
		spec: ParamSpec{
			Name:      deviceName,
			Direction: DirectionScalar,
		},
	}
}

// Temp creates a parameter specification for a device-only temporary array
func Temp(deviceName string) *ParamBuilder {
	return &ParamBuilder{
		spec: ParamSpec{
			Name:      deviceName,
			Direction: DirectionTemp,
		},
	}
}

// Bind associates a host variable with this parameter
func (p *ParamBuilder) Bind(hostVar interface{}) *ParamBuilder {
	p.spec.HostBinding = hostVar

	// Infer type and size if possible
	p.inferFromBinding()

	return p
}

// Copy sets bidirectional copy (host→device before, device→host after)
func (p *ParamBuilder) Copy() *ParamBuilder {
	p.spec.DoCopyTo = true
	p.spec.DoCopyBack = true
	return p
}

// CopyTo sets host→device copy before kernel execution
func (p *ParamBuilder) CopyTo() *ParamBuilder {
	p.spec.DoCopyTo = true
	return p
}

// CopyBack sets device→host copy after kernel execution
func (p *ParamBuilder) CopyBack() *ParamBuilder {
	p.spec.DoCopyBack = true
	return p
}

// NoCopy explicitly disables data movement
func (p *ParamBuilder) NoCopy() *ParamBuilder {
	p.spec.DoCopyTo = false
	p.spec.DoCopyBack = false
	return p
}

// Convert sets type conversion during copy operations
func (p *ParamBuilder) Convert(toType builder.DataType) *ParamBuilder {
	p.spec.ConvertType = toType
	return p
}

// Type sets explicit type (mainly for Temp arrays)
func (p *ParamBuilder) Type(dataType builder.DataType) *ParamBuilder {
	p.spec.DataType = dataType
	return p
}

// Size sets explicit size (mainly for Temp arrays)
func (p *ParamBuilder) Size(elements int) *ParamBuilder {
	p.spec.Size = int64(elements)
	return p
}

// ToMatrix marks this parameter as a matrix, enabling MATMUL macro generation
func (p *ParamBuilder) ToMatrix() *ParamBuilder {
	p.spec.IsMatrix = true

	// If bound to a mat.Matrix, extract dimensions
	if p.spec.HostBinding != nil {
		if m, ok := p.spec.HostBinding.(mat.Matrix); ok {
			p.spec.MatrixRows, p.spec.MatrixCols = m.Dims()
		}
	}

	return p
}

// Static marks a matrix for static embedding (const array in kernel)
func (p *ParamBuilder) Static() *ParamBuilder {
	p.spec.IsStatic = true
	return p
}

// Stride sets the stride for promoting flat arrays to matrices
func (p *ParamBuilder) Stride(stride int) *ParamBuilder {
	p.spec.Stride = stride
	p.spec.MatrixRows = int(p.spec.Size) / stride
	p.spec.MatrixCols = stride
	return p
}

// Align sets memory alignment requirements
func (p *ParamBuilder) Align(alignment builder.AlignmentType) *ParamBuilder {
	p.spec.Alignment = alignment
	return p
}

// inferFromBinding extracts type and size information from the host binding
func (p *ParamBuilder) inferFromBinding() {
	if p.spec.HostBinding == nil {
		return
	}

	v := reflect.ValueOf(p.spec.HostBinding)
	t := v.Type()

	// Handle slices
	if t.Kind() == reflect.Slice {
		p.spec.Size = int64(v.Len())

		// Infer element type
		elemType := t.Elem()
		switch elemType.Kind() {
		case reflect.Float32:
			p.spec.DataType = builder.Float32
		case reflect.Float64:
			p.spec.DataType = builder.Float64
		case reflect.Int32:
			p.spec.DataType = builder.INT32
		case reflect.Int64:
			p.spec.DataType = builder.INT64
		}
		return
	}

	// Handle mat.Matrix
	if m, ok := p.spec.HostBinding.(mat.Matrix); ok {
		rows, cols := m.Dims()
		p.spec.Size = int64(rows * cols)
		p.spec.DataType = builder.Float64 // gonum matrices are float64
		p.spec.MatrixRows = rows
		p.spec.MatrixCols = cols
		return
	}

	// Handle scalars
	switch t.Kind() {
	case reflect.Float32:
		p.spec.DataType = builder.Float32
		p.spec.Size = 1
	case reflect.Float64:
		p.spec.DataType = builder.Float64
		p.spec.Size = 1
	case reflect.Int, reflect.Int64:
		p.spec.DataType = builder.INT64
		p.spec.Size = 1
	case reflect.Int32:
		p.spec.DataType = builder.INT32
		p.spec.Size = 1
	}
}

// Validate checks if the parameter specification is complete and valid
func (p *ParamSpec) Validate() error {
	if p.Name == "" {
		return fmt.Errorf("parameter name cannot be empty")
	}

	// Scalars don't need size
	if p.Direction == DirectionScalar {
		if p.DataType == 0 && p.HostBinding == nil {
			return fmt.Errorf("scalar %s needs type or binding", p.Name)
		}
		return nil
	}

	// Arrays need size and type
	if p.Direction != DirectionScalar {
		if p.Size == 0 {
			return fmt.Errorf("array %s needs size", p.Name)
		}
		if p.DataType == 0 {
			return fmt.Errorf("array %s needs type", p.Name)
		}
	}

	// Matrix validation
	if p.IsMatrix {
		if p.Direction == DirectionScalar {
			return fmt.Errorf("scalars cannot be matrices")
		}
		if p.IsStatic && p.HostBinding == nil {
			return fmt.Errorf("static matrix %s needs binding", p.Name)
		}
	}

	// Temp arrays cannot have host bindings or copy operations
	if p.Direction == DirectionTemp {
		if p.HostBinding != nil {
			return fmt.Errorf("temp array %s cannot have host binding", p.Name)
		}
		if p.DoCopyTo || p.DoCopyBack {
			return fmt.Errorf("temp array %s cannot have copy operations", p.Name)
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
func (p *ParamSpec) GetEffectiveType() builder.DataType {
	if p.ConvertType != 0 {
		return p.ConvertType
	}
	return p.DataType
}

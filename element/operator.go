package element

type ArgumentType uint8

const (
	INT ArgumentType = iota
	FLOAT32
	FLOAT64
	INTSLICE
	FLOAT32SLICE
	FLOAT64SLICE
	MATMATRIX
)

type OperandType uint8

const (
	INPUT OperandType = iota
	OUTPUT
	TEMPORARY
)

type Argument struct {
	Name    string
	IO      ArgumentType
	Operand OperandType
	Stride  int
}

type Operator interface {
	GetMacro() string // Used for operators defined as a macro
	GetArguments() []Argument
	GetKernelSource() string
}

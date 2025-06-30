package gonudg

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// GeometricFactors3D computes the metric elements for the local mappings of the elements
func (dg *DG3D) GeometricFactors3D() error {
	// Calculate geometric factors
	// xr = Dr*X, xs = Ds*X, xt = Dt*X
	var xr, xs, xt mat.Dense
	xr.Mul(dg.Dr, dg.X)
	xs.Mul(dg.Ds, dg.X)
	xt.Mul(dg.Dt, dg.X)

	var yr, ys, yt mat.Dense
	yr.Mul(dg.Dr, dg.Y)
	ys.Mul(dg.Ds, dg.Y)
	yt.Mul(dg.Dt, dg.Y)

	var zr, zs, zt mat.Dense
	zr.Mul(dg.Dr, dg.Z)
	zs.Mul(dg.Ds, dg.Z)
	zt.Mul(dg.Dt, dg.Z)

	// Compute Jacobian determinant
	// J = xr*(ys*zt - zs*yt) - yr*(xs*zt - zs*xt) + zr*(xs*yt - ys*xt)
	dg.J = mat.NewDense(dg.Np, dg.K, nil)
	for i := 0; i < dg.Np; i++ {
		for k := 0; k < dg.K; k++ {
			J := xr.At(i, k)*(ys.At(i, k)*zt.At(i, k)-zs.At(i, k)*yt.At(i, k)) -
				yr.At(i, k)*(xs.At(i, k)*zt.At(i, k)-zs.At(i, k)*xt.At(i, k)) +
				zr.At(i, k)*(xs.At(i, k)*yt.At(i, k)-ys.At(i, k)*xt.At(i, k))

			if J <= 0 {
				return fmt.Errorf("negative Jacobian at node %d, element %d: %f", i, k, J)
			}
			dg.J.Set(i, k, J)
		}
	}

	// Initialize metric term matrices
	dg.Rx = mat.NewDense(dg.Np, dg.K, nil)
	dg.Ry = mat.NewDense(dg.Np, dg.K, nil)
	dg.Rz = mat.NewDense(dg.Np, dg.K, nil)
	dg.Sx = mat.NewDense(dg.Np, dg.K, nil)
	dg.Sy = mat.NewDense(dg.Np, dg.K, nil)
	dg.Sz = mat.NewDense(dg.Np, dg.K, nil)
	dg.Tx = mat.NewDense(dg.Np, dg.K, nil)
	dg.Ty = mat.NewDense(dg.Np, dg.K, nil)
	dg.Tz = mat.NewDense(dg.Np, dg.K, nil)

	// Compute inverse metric terms
	for i := 0; i < dg.Np; i++ {
		for k := 0; k < dg.K; k++ {
			J := dg.J.At(i, k)

			// Rx = (ys*zt - zs*yt)/J, Ry = -(xs*zt - zs*xt)/J, Rz = (xs*yt - ys*xt)/J
			dg.Rx.Set(i, k, (ys.At(i, k)*zt.At(i, k)-zs.At(i, k)*yt.At(i, k))/J)
			dg.Ry.Set(i, k, -(xs.At(i, k)*zt.At(i, k)-zs.At(i, k)*xt.At(i, k))/J)
			dg.Rz.Set(i, k, (xs.At(i, k)*yt.At(i, k)-ys.At(i, k)*xt.At(i, k))/J)

			// Sx = -(yr*zt - zr*yt)/J, Sy = (xr*zt - zr*xt)/J, Sz = -(xr*yt - yr*xt)/J
			dg.Sx.Set(i, k, -(yr.At(i, k)*zt.At(i, k)-zr.At(i, k)*yt.At(i, k))/J)
			dg.Sy.Set(i, k, (xr.At(i, k)*zt.At(i, k)-zr.At(i, k)*xt.At(i, k))/J)
			dg.Sz.Set(i, k, -(xr.At(i, k)*yt.At(i, k)-yr.At(i, k)*xt.At(i, k))/J)

			// Tx = (yr*zs - zr*ys)/J, Ty = -(xr*zs - zr*xs)/J, Tz = (xr*ys - yr*xs)/J
			dg.Tx.Set(i, k, (yr.At(i, k)*zs.At(i, k)-zr.At(i, k)*ys.At(i, k))/J)
			dg.Ty.Set(i, k, -(xr.At(i, k)*zs.At(i, k)-zr.At(i, k)*xs.At(i, k))/J)
			dg.Tz.Set(i, k, (xr.At(i, k)*ys.At(i, k)-yr.At(i, k)*xs.At(i, k))/J)
		}
	}

	return nil
}

// Normals3D computes outward pointing normals at element faces and surface Jacobians
func (dg *DG3D) Normals3D() error {
	// First compute geometric factors
	if err := dg.GeometricFactors3D(); err != nil {
		return err
	}

	// Initialize normal and surface Jacobian matrices
	Nfp := dg.Nfp
	Nfaces := dg.Nfaces
	K := dg.K

	dg.Nx = mat.NewDense(Nfp*Nfaces, K, nil)
	dg.Ny = mat.NewDense(Nfp*Nfaces, K, nil)
	dg.Nz = mat.NewDense(Nfp*Nfaces, K, nil)
	dg.SJ = mat.NewDense(Nfp*Nfaces, K, nil)

	// Build normals for each face
	// Face 1: t = -1, normal = -[Tx, Ty, Tz]
	for k := 0; k < K; k++ {
		for i := 0; i < Nfp; i++ {
			vid := dg.Fmask[0][i] // volume node index
			row := 0*Nfp + i      // face node index

			dg.Nx.Set(row, k, -dg.Tx.At(vid, k))
			dg.Ny.Set(row, k, -dg.Ty.At(vid, k))
			dg.Nz.Set(row, k, -dg.Tz.At(vid, k))
		}
	}

	// Face 2: s = -1, normal = -[Sx, Sy, Sz]
	for k := 0; k < K; k++ {
		for i := 0; i < Nfp; i++ {
			vid := dg.Fmask[1][i]
			row := 1*Nfp + i

			dg.Nx.Set(row, k, -dg.Sx.At(vid, k))
			dg.Ny.Set(row, k, -dg.Sy.At(vid, k))
			dg.Nz.Set(row, k, -dg.Sz.At(vid, k))
		}
	}

	// Face 3: r+s+t = -1, normal = [Rx+Sx+Tx, Ry+Sy+Ty, Rz+Sz+Tz]
	for k := 0; k < K; k++ {
		for i := 0; i < Nfp; i++ {
			vid := dg.Fmask[2][i]
			row := 2*Nfp + i

			dg.Nx.Set(row, k, dg.Rx.At(vid, k)+dg.Sx.At(vid, k)+dg.Tx.At(vid, k))
			dg.Ny.Set(row, k, dg.Ry.At(vid, k)+dg.Sy.At(vid, k)+dg.Ty.At(vid, k))
			dg.Nz.Set(row, k, dg.Rz.At(vid, k)+dg.Sz.At(vid, k)+dg.Tz.At(vid, k))
		}
	}

	// Face 4: r = -1, normal = -[Rx, Ry, Rz]
	for k := 0; k < K; k++ {
		for i := 0; i < Nfp; i++ {
			vid := dg.Fmask[3][i]
			row := 3*Nfp + i

			dg.Nx.Set(row, k, -dg.Rx.At(vid, k))
			dg.Ny.Set(row, k, -dg.Ry.At(vid, k))
			dg.Nz.Set(row, k, -dg.Rz.At(vid, k))
		}
	}

	// Normalize normals and compute surface Jacobian
	for i := 0; i < Nfp*Nfaces; i++ {
		for k := 0; k < K; k++ {
			nx := dg.Nx.At(i, k)
			ny := dg.Ny.At(i, k)
			nz := dg.Nz.At(i, k)

			// Compute magnitude (surface Jacobian before scaling)
			mag := math.Sqrt(nx*nx + ny*ny + nz*nz)

			if mag < 1e-14 {
				return fmt.Errorf("zero normal magnitude at face node %d, element %d", i, k)
			}

			// Normalize to unit vector
			dg.Nx.Set(i, k, nx/mag)
			dg.Ny.Set(i, k, ny/mag)
			dg.Nz.Set(i, k, nz/mag)

			// Store magnitude temporarily
			dg.SJ.Set(i, k, mag)
		}
	}

	// Scale surface Jacobian by volume Jacobian at face nodes
	// SJ = SJ * J(Fmask(:),:)
	for face := 0; face < Nfaces; face++ {
		for i := 0; i < Nfp; i++ {
			vid := dg.Fmask[face][i] // volume node index
			row := face*Nfp + i      // face node index

			for k := 0; k < K; k++ {
				dg.SJ.Set(row, k, dg.SJ.At(row, k)*dg.J.At(vid, k))
			}
		}
	}

	return nil
}

// ComputeFscale computes the ratio of face Jacobians for flux computations
func (dg *DG3D) ComputeFscale() {
	// Compute Fscale = SJ ./ J(Fmask,:)
	Nfp := dg.Nfp
	Nfaces := dg.Nfaces
	K := dg.K

	dg.Fscale = mat.NewDense(Nfp*Nfaces, K, nil)

	for face := 0; face < Nfaces; face++ {
		for i := 0; i < Nfp; i++ {
			vid := dg.Fmask[face][i] // volume node index
			row := face*Nfp + i      // face node index

			for k := 0; k < K; k++ {
				// Fscale = SJ / J at face nodes
				dg.Fscale.Set(row, k, dg.SJ.At(row, k)/dg.J.At(vid, k))
			}
		}
	}
}

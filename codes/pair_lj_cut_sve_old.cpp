/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */


#include "pair_lj_cut_sve.h"

#include "atom.h"
#include "cluster_neigh.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "respa.h"
#include "thr_sve.h"
#include "update.h"

#include <arm_sve.h>
#include <cmath>
#include <cstring>
#include <ctime>
#include <omp.h>
#include <cassert>
#include "sve_util.h"

using namespace LAMMPS_NS;
using namespace MathConst;

uint64_t rdtsc() {
    //Please note we read CNTVCT cpu system register which provides
    //the accross-system consistent value of the virtual system counter.
    uint64_t cntvct;
    asm volatile ("mrs %0, cntvct_el0; " : "=r"(cntvct) :: "memory");
    return cntvct;
}

/* ---------------------------------------------------------------------- */

PairLJCutSVE::PairLJCutSVE(LAMMPS *lmp) : PairLJCut(lmp)
{
  respa_enable = 0;
  born_matrix_enable = 1;
  writedata = 1;
  nall_alloc = 0;
  thrf = nullptr;
  dirtyf = nullptr;
  cut_respa = nullptr;
  inner_time = 0;
  thr = new ThrSVE(lmp);
}
static constexpr int unroll_count = 4;
template<typename T>
using usv = usv_t<T, unroll_count>;
// svint64_t operator *(svint64_t a, int64_t b) {
//   return svmul_n_s64_x(svptrue_b64(), a, b);
// }
void set_range(int &st, int &ed, int n, int nthr, int ithr) {
  int perthr = n / nthr;
  int remain = n % nthr;
  st = perthr * ithr + std::min(ithr, remain);
  ed = st + perthr + (ithr < remain);
}
/* ---------------------------------------------------------------------- */
template <int MOL, int NEWTON, int EVFLAG> void PairLJCutSVE::eval(int eflag, int vflag)
{
  using NeighEnt = ClusterNeighEntry<!MOL>;
  double evdwl = 0.0;
  double (*x)[3] = (double(*)[3])atom->x[0];
  double (*f)[3] = (double(*)[3])atom->f[0];
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int ntypes = atom->ntypes;
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int *numneigh_cluster = list->numneigh_inner;
  NeighEnt **firstneigh_cluster = (NeighEnt **) list->firstneigh_inner;
  svuint64_t svmaskbits = svlsl_m(svptrue_b64(), svdup_u64(1), svindex_u64(0, 1));

  double *flatx = (double*)x;
  static constexpr int dirty_page = 64;
  static constexpr int align = std::max(dirty_page, CLUSTERSIZE);
  // loop over neighbors of my atoms
  int icnum = (inum + CLUSTERSIZE - 1) / CLUSTERSIZE;
  int nall = (nlocal+atom->nghost);
  int nall_align = (nall + align - 1) / align * align;
  int nthr = omp_get_max_threads();
  thr->allocate(nall, eflag_atom, vflag_atom);
  // if (nall_align > nall_alloc) {
  //   memory->sfree(thrf);
  //   memory->destroy(dirtyf);
  //   nall_alloc = nall_align * 1.5;
  //   thrf = (double(*)[3])memory->smalloc(3*nall_alloc*sizeof(double)*nthr, "thrf");
  //   memory->create(dirtyf, nall_alloc*nthr*sizeof(char), "dirtyf");
  // }
  double loop_time = 0;
  // long st = rdtsc();
  std::fill_n(virial, 6, 0);
  int ndirty = 0;
  double v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0;
  #pragma omp parallel num_threads(nthr) //reduction(+:eng_vdwl, ndirty, v1, v2, v3, v4, v5) reduction(max:loop_time)
  {
    double(*lj1i)[CLUSTERSIZE] = (double(*)[8]) alloca((ntypes + 1) * CLUSTERSIZE * sizeof(double));
    double(*lj2i)[CLUSTERSIZE] = (double(*)[8]) alloca((ntypes + 1) * CLUSTERSIZE * sizeof(double));
    double(*lj3i)[CLUSTERSIZE] = (double(*)[8]) alloca((ntypes + 1) * CLUSTERSIZE * sizeof(double));
    double(*lj4i)[CLUSTERSIZE] = (double(*)[8]) alloca((ntypes + 1) * CLUSTERSIZE * sizeof(double));
    double(*cutsqi)[CLUSTERSIZE] = (double(*)[8]) alloca((ntypes + 1) * CLUSTERSIZE * sizeof(double));
    double(*offseti)[CLUSTERSIZE] = (double(*)[8]) alloca((ntypes + 1) * CLUSTERSIZE * sizeof(double));

    // while(1) {asm volatile("");}

    int ithr = omp_get_thread_num();
    int nthr = omp_get_num_threads();
    int icst, iced;
    thr->set_range(icst, iced, ithr, icnum);

    int ndirtypage = (nall_align + dirty_page - 1) / dirty_page;

    struct timespec st, ed;
    if (neighbor->ago == 0) {
      thr->inspect_cluster<MOL>(ithr, list);
    }
    thr->init_pair(ithr, nall, eflag_atom, vflag_atom, vflag_fdotr);
    double (*my_f)[3] = thr->force_thr(ithr);
    double *eng_virial = thr->eng_virial_thr(ithr);
    for (int ic = icst; ic < iced; ic++) {
      int iis = ic * CLUSTERSIZE;
      int iie = std::min(iis + CLUSTERSIZE, inum);
      int iicnt = iie - iis;

      svbool_t predi32 = svwhilelt_b32(0, iicnt);
      svbool_t predi64 = svwhilelt_b64(0, iicnt);

      svint32_t is32 = svld1(predi32, ilist + iis);
      svint64_t is64 = svunpklo_s64(is32);

      svd_t xi = svld1_gather_offset(predi64, (double*)x, is64*24);
      svd_t yi = svld1_gather_offset(predi64, (double*)x, is64*24+8);
      svd_t zi = svld1_gather_offset(predi64, (double*)x, is64*24+16);
      svint64_t itype = svunpklo_s64(svld1_gather_offset(predi32, type, is32*4));

      svint64_t ibase = itype*(ntypes+1) * 8;
      for (int jtype = 0; jtype <= ntypes; jtype ++) {
        svd_t vlj1i = svld1_gather_offset(predi64, lj1[0], ibase+jtype * 8);
        svd_t vlj2i = svld1_gather_offset(predi64, lj2[0], ibase+jtype * 8);
        svd_t vcutsqi = svld1_gather_offset(predi64, cutsq[0], ibase+jtype * 8);
        svd_t voffseti = svld1_gather_offset(predi64, offset[0], ibase+jtype * 8);
        svst1(predi64, lj1i[jtype], vlj1i);
        svst1(predi64, lj2i[jtype], vlj2i);
        svst1(predi64, cutsqi[jtype], vcutsqi);
        svst1(predi64, offseti[jtype], voffseti);
        if (EVFLAG) {
          svd_t vlj3i = svld1_gather_offset(predi64, lj3[0], ibase+jtype * 8);
          svd_t vlj4i = svld1_gather_offset(predi64, lj4[0], ibase+jtype * 8);
          svst1(predi64, lj3i[jtype], vlj3i);
          svst1(predi64, lj4i[jtype], vlj4i);
        }
      }

      svd_t fxi = svdup_f64(0), fyi = svdup_f64(0), fzi = svdup_f64(0);
      NeighEnt *jlist = firstneigh_cluster[ic];
      int jnum = numneigh_cluster[ic];
      for (int jjj = 0; jjj < jnum; jjj+=unroll_count) {
        #pragma unroll(4)
        for (int jj = jjj; jj < jjj + 4; jj ++) {
          NeighEnt jent0 = jlist[jj];
          int j0 = jent0.j;
          int jtype0 = type[j0];
          svuint64_t neighmask0 = svdup_u64(jent0.get_allmask());
          svbool_t inmask0 = svcmpgt(svptrue_b64(), svand_m(svptrue_b64(), neighmask0, svmaskbits), 0);
          
          svd_t xj0 = svdup_f64(x[j0][0]);
          svd_t yj0 = svdup_f64(x[j0][1]);
          svd_t zj0 = svdup_f64(x[j0][2]);
          svd_t delx0 = xi - xj0;
          svd_t dely0 = yi - yj0;
          svd_t delz0 = zi - zj0;
          svd_t rsq0 = delx0 * delx0 + dely0 * dely0 + delz0 * delz0;
          svd_t cutijsq0 = svld1_f64(inmask0, cutsqi[jtype0]);
          svbool_t incut0 = svcmplt(inmask0, rsq0, cutijsq0);

          svd_t r2inv0 = svdiv_m(svptrue_b64(), svdup_f64(1.0), rsq0);
          // r2inv = svrecps_f64(rsq, r2inv);
          svd_t r6inv0 = r2inv0 * r2inv0 * r2inv0;
          svd_t vlj1_0 = svld1_f64(svptrue_b64(), lj1i[jtype0]);
          svd_t vlj2_0 = svld1_f64(svptrue_b64(), lj2i[jtype0]);
          svd_t forcelj0 = r6inv0 * (vlj1_0 * r6inv0 - vlj2_0);
          svd_t fpair0 = forcelj0 * r2inv0;
          
          fxi = svadd_m(incut0, fxi, fpair0 * delx0);
          fyi = svadd_m(incut0, fyi, fpair0 * dely0);
          fzi = svadd_m(incut0, fzi, fpair0 * delz0);
          
          if (NEWTON) {
            my_f[j0][0] -= svaddv_f64(incut0, fpair0 * delx0);
            my_f[j0][1] -= svaddv_f64(incut0, fpair0 * dely0);
            my_f[j0][2] -= svaddv_f64(incut0, fpair0 * delz0);
          } else {
            svbool_t sub0 = j0 < nlocal ? incut0 : svpfalse();
            my_f[j0][0] -= svaddv_f64(sub0, fpair0 * delx0); 
            my_f[j0][1] -= svaddv_f64(sub0, fpair0 * dely0);
            my_f[j0][2] -= svaddv_f64(sub0, fpair0 * delz0);
          }
          if (EVFLAG) {
            if (eflag) {
              svd_t vlj3_0 = svld1_f64(svptrue_b64(), lj3i[jtype0]);
              svd_t vlj4_0 = svld1_f64(svptrue_b64(), lj4i[jtype0]);
              svd_t voffset0 = svld1_f64(svptrue_b64(), offseti[jtype0]);
              svd_t evdwl0 = r6inv0 * (vlj3_0 * r6inv0 - vlj4_0) - voffset0;    // - offset[itype][jtype];
              // evdwl *= factor_lj;
              eng_virial[6] += svaddv_f64(incut0, evdwl0);
            }
          }
        }
      }
      svd_t cfxi = svld1_gather_offset(predi64, (double*)my_f, is64*24);
      svd_t cfyi = svld1_gather_offset(predi64, (double*)my_f, is64*24+8);
      svd_t cfzi = svld1_gather_offset(predi64, (double*)my_f, is64*24+16);
      svst1_scatter_offset(predi64, (double*)my_f, is64 * 24, cfxi + fxi);
      svst1_scatter_offset(predi64, (double*)my_f, is64 * 24 + 8, cfyi + fyi);
      svst1_scatter_offset(predi64, (double*)my_f, is64 * 24 + 16, cfzi + fzi);
    }

    #pragma omp barrier
    if (vflag_fdotr)
      thr->reduce_pair<1>(atom, this, ithr, nthr, eflag_atom, vflag_atom);
    else
      thr->reduce_pair<0>(atom, this, ithr, nthr, eflag_atom, vflag_atom);

  }
  thr->reduce_scalar(this, eflag_global, vflag_global || vflag_fdotr);
  // this->inner_time += loop_time;
  // if (update->ntimestep == 1000) {
  //   printf("core loop time: %f %d\n", this->inner_time, ndirty);
  // }
}
void PairLJCutSVE::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  if (atom->molecular == 0) {
    if (evflag == 0) {
      if (force->newton_pair == 0) {
        eval<0, 0, 0>(eflag, vflag);
      } else {
        eval<0, 1, 0>(eflag, vflag);
      }
    } else {
      if (force->newton_pair == 0) {
        eval<0, 0, 1>(eflag, vflag);
      } else {
        eval<0, 1, 1>(eflag, vflag);
      }
    }
  } else {
    if (evflag == 0) {
      if (force->newton_pair == 0) {
        eval<1, 0, 0>(eflag, vflag);
      } else {
        eval<1, 1, 0>(eflag, vflag);
      }
    } else {
      if (force->newton_pair == 0) {
        eval<1, 0, 1>(eflag, vflag);
      } else {
        eval<1, 1, 1>(eflag, vflag);
      }
    }
  }
  // if (vflag_fdotr) virial_fdotr_compute();
  // int i, j, ii, jj, inum, jnum, itype, jtype;
  // double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
  // double rsq, r2inv, r6inv, forcelj, factor_lj;
  // int *ilist, *jlist, *numneigh, **firstneigh;

  // evdwl = 0.0;
  //
  // double **x = atom->x;
  // double **f = atom->f;
  // int *type = atom->type;
  // int nlocal = atom->nlocal;
  // double *special_lj = force->special_lj;
  // int newton_pair = force->newton_pair;

  // inum = list->inum;
  // ilist = list->ilist;
  // numneigh = list->numneigh;
  // firstneigh = list->firstneigh;

  // // loop over neighbors of my atoms

  // for (ii = 0; ii < inum; ii++) {
  //   i = ilist[ii];
  //   xtmp = x[i][0];
  //   ytmp = x[i][1];
  //   ztmp = x[i][2];
  //   itype = type[i];
  //   jlist = firstneigh[i];
  //   jnum = numneigh[i];

  //   for (jj = 0; jj < jnum; jj++) {
  //     j = jlist[jj];
  //     factor_lj = special_lj[sbmask(j)];
  //     j &= NEIGHMASK;

  //     delx = xtmp - x[j][0];
  //     dely = ytmp - x[j][1];
  //     delz = ztmp - x[j][2];
  //     rsq = delx * delx + dely * dely + delz * delz;
  //     jtype = type[j];

  //     if (rsq < cutsq[itype][jtype]) {
  //       r2inv = 1.0 / rsq;
  //       r6inv = r2inv * r2inv * r2inv;
  //       forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
  //       fpair = factor_lj * forcelj * r2inv;

  //       f[i][0] += delx * fpair;
  //       f[i][1] += dely * fpair;
  //       f[i][2] += delz * fpair;
  //       if (newton_pair || j < nlocal) {
  //         f[j][0] -= delx * fpair;
  //         f[j][1] -= dely * fpair;
  //         f[j][2] -= delz * fpair;
  //       }

  //       if (eflag) {
  //         evdwl = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
  //         evdwl *= factor_lj;
  //       }

  //       if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
  //     }
  //   }
  // }

  // if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLJCutSVE::allocate()
{
  PairLJCut::allocate();
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLJCutSVE::init_style()
{
  // request regular or rRESPA neighbor list

  int list_style = NeighConst::REQ_DEFAULT | NeighConst::REQ_CLUSTER;

  neighbor->add_request(this, list_style);

}

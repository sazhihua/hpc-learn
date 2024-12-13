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
#include "respa.h"
#include "update.h"

#include <arm_sve.h>
#include <cmath>
#include <cstring>
#include <mpi.h>
#include <sys/cdefs.h>
#include <utility>
#include "sve_util.h"
using namespace LAMMPS_NS;
using namespace MathConst;
#include "lwpf.hpp"
enum nonbev
{
    ALL,
    ILOOP,
    JLOOP,
    WRITEJ,
    LOADJ,
    LSFJ,
    ADDVJ
};
static lwpfpp<nonbev, cycle_machine> e;
/* ---------------------------------------------------------------------- */

PairLJCutSVE::PairLJCutSVE(LAMMPS *lmp) : PairLJCut(lmp)
{
  respa_enable = 0;
  born_matrix_enable = 1;
  writedata = 1;
  cut_respa = nullptr;
}
static constexpr int unroll_count = 4;
__always_inline svd_t sumup(svd_t v0, svd_t v1, svd_t v2){
  svd_t v01o = svuzp2(v0, v1); // 从 v0 和 v1 提取奇数索引元素 odd
  svd_t v01e = svuzp1(v0, v1); // 从 v0 和 v1 提取偶数索引元素 even
  svd_t v23o = svuzp2(v2, svdup_f64(0.0)); // 从 v2 和零向量提取奇数索引元素
  svd_t v23e = svuzp1(v2, svdup_f64(0.0)); // 从 v2 和零向量提取偶数索引元素
  svd_t vm0 = v01o + v01e;
  svd_t vm1 = v23o + v23e;
  svd_t vm01o = svuzp2(vm0, vm1);
  svd_t vm01e = svuzp1(vm0, vm1);
  svd_t vm01 = vm01o + vm01e;
  svd_t vm0123o = svuzp2(vm01, vm01);
  svd_t vm0123e = svuzp1(vm01, vm01);
  return vm0123o + vm0123e;
}
template<typename T>
using usv = usv_t<T, unroll_count>;
/* ---------------------------------------------------------------------- */
template <int MOL, int NEWTON, int EVFLAG, int ...J> void PairLJCutSVE::eval(int eflag, int vflag, std::integer_sequence<int, J...>)
{
  //if ( update->ntimestep == 0)
  //  e.init(1);
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
  enum ParamIdx {LJ1I, LJ2I, LJ3I, LJ4I, CUTSQI, OFFSETI, PARM_N};
  double (*param)[PARM_N][CLUSTERSIZE] = (double(*)[PARM_N][8]) alloca((ntypes + 1) * PARM_N * CLUSTERSIZE * sizeof(double));
  // MPI_Barrier(MPI_COMM_WORLD);
  double *flatx = (double*)x;
  // loop over neighbors of my atoms
  //e.start(ILOOP);
  for (int ic = 0; ic * CLUSTERSIZE < inum; ic++) {
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
      svst1(predi64, param[jtype][LJ1I], vlj1i);
      svst1(predi64, param[jtype][LJ2I], vlj2i);
      svst1(predi64, param[jtype][CUTSQI], vcutsqi);
      svst1(predi64, param[jtype][OFFSETI], voffseti);
      
      if (EVFLAG) {
        svd_t vlj3i = svld1_gather_offset(predi64, lj3[0], ibase+jtype * 8);
        svd_t vlj4i = svld1_gather_offset(predi64, lj4[0], ibase+jtype * 8);
        svst1(predi64, param[jtype][LJ3I], vlj3i);
        svst1(predi64, param[jtype][LJ4I], vlj4i);
      }
    }

    svd_t fxi = svdup_f64(0), fyi = svdup_f64(0), fzi = svdup_f64(0);
    NeighEnt *jlist = firstneigh_cluster[ic];
    int jnum = numneigh_cluster[ic];
    // e.start(JLOOP);
    for (int jjj = 0; jjj < jnum; jjj+=sizeof...(J)) {
        NeighEnt jents[] = {jlist[jjj+J]...};
        int js[] = {jents[J].j...};
        svul_t neighmasks[] = {svdup_u64(jents[J].get_allmask())...};
        svp_t inmasks[] = {svcmpgt(svptrue_b64(), svand_m(svptrue_b64(), neighmasks[J], svmaskbits), 0)...};

        // e.start(LOADJ);
        int jtypes[] = {type[js[J]]...};

        svd_t xj[] = {svdup_f64(x[js[J]][0])...};
        svd_t yj[] = {svdup_f64(x[js[J]][1])...};
        svd_t zj[] = {svdup_f64(x[js[J]][2])...};
        // e.stop(LOADJ);
        int js_next[] = {jlist[std::min(jnum - 1, jjj+J+unroll_count)].j...};
        {__attribute__((unused)) int a[] = {(svprfw(svwhilelt_b32(0, 1), type + js_next[J], SV_PLDL1KEEP), 0)...};}
        {__attribute__((unused)) int a[] = {(svprfd(svwhilelt_b64(0, 3), x[js_next[J]], SV_PLDL1KEEP), 0)...};}
        {__attribute__((unused)) int a[] = {(svprfd(svwhilelt_b64(0, 3), f[js[J]], SV_PLDL1KEEP), 0)...};}
        svd_t delx[] = {xi - xj[J]...};
        svd_t dely[] = {yi - yj[J]...};
        svd_t delz[] = {zi - zj[J]...};
        
        svd_t rsqs[] = {delx[J]*delx[J] + dely[J]*dely[J] + delz[J]*delz[J]...};

        svd_t cutijsqs[] = { svld1_f64(inmasks[J], param[jtypes[J]][CUTSQI]) ...};

        svp_t incuts[] = { svcmplt(inmasks[J], rsqs[J], cutijsqs[J]) ... };
        // asm volatile("");
        // svd_t r2invs0[] = {svrecpe_f64(rsqs[J])...};
        svd_t r2invs[] = {1.0 / rsqs[J] ...};
        // svd_t r2invs1[] = {r2invs0[J] * (svd_t)svrecps(r2invs0[J], rsqs[J])...};
        // svd_t r2invs2[] = {r2invs1[J] * (svd_t)svrecps(r2invs1[J], rsqs[J])...};
        // svd_t r2invs[] = {r2invs2[J] * (svd_t)svrecps(r2invs2[J], rsqs[J])...};
        svd_t r4invs[] = {r2invs[J] * r2invs[J] ...};
        svd_t r6invs[] = {r4invs[J] * r2invs[J] ...};
        svd_t r8invs[] = {r4invs[J] * r4invs[J] ...};
        svd_t vlj1s[] = {svld1_f64(incuts[J], param[jtypes[J]][LJ1I])...};
        svd_t vlj2s[] = {svld1_f64(incuts[J], param[jtypes[J]][LJ2I])...};
        
        // svd_t fpairs[] = { r8invs[J] * (svd_t)(svnmsb_f64_x(svptrue_b64(), vlj1s[J], r6invs[J], vlj2s[J])) ...};
        svd_t fpairs[] = {r8invs[J] * (r6invs[J] * vlj1s[J] - vlj2s[J])...};
        svd_t fx[] = {fpairs[J] * delx[J] ...};
        svd_t fy[] = {fpairs[J] * dely[J] ...};
        svd_t fz[] = {fpairs[J] * delz[J] ...};

        {__attribute__((unused)) svd_t a[] = {fxi = svadd_m(incuts[J], fxi, fx[J])...};}
        {__attribute__((unused)) svd_t a[] = {fyi = svadd_m(incuts[J], fyi, fy[J])...};}
        {__attribute__((unused)) svd_t a[] = {fzi = svadd_m(incuts[J], fzi, fz[J])...};}
        // e.start(WRITEJ);
        if (NEWTON) {
          // for (int j = 0; j < unroll_count; j ++) {
          // e.start(ADDVJ);
          svbool_t lt3 = svwhilelt_b64(0, 3);
          svd_t fjvs[] = {(svd_t)svld1(lt3, f[js[J]])...};
          svd_t fj[] = {sumup(svsel(incuts[J], fx[J], svdup_f64(0.0)), svsel(incuts[J], fy[J], svdup_f64(0.0)), svsel(incuts[J], fz[J], svdup_f64(0.0)))...};
          // double fj[][3] = {{svaddv_f64(incuts[J], fx[J]), svaddv_f64(incuts[J], fy[J]), s vaddv_f64(incuts[J], fz[J])}...};
          // double fxjs[sizeof...(J)][8];
          // double fyjs[sizeof...(J)][8];
          // double fzjs[sizeof...(J)][8];
          
          // {__attribute__((unused))int a[]{(svst1(svptrue_b64(), fxjs[J], svdup_f64(0.0)), svst1(incuts[J], fxjs[J], fx[J]), 0)...};}
          // {__attribute__((unused))int a[]{(svst1(svptrue_b64(), fyjs[J], svdup_f64(0.0)), svst1(incuts[J], fyjs[J], fy[J]), 0)...};}
          // {__attribute__((unused))int a[]{(svst1(svptrue_b64(), fzjs[J], svdup_f64(0.0)), svst1(incuts[J], fzjs[J], fz[J]), 0)...};}
          // double fjs[sizeof...(J)][3];
          // for (int j = 0; j < sizeof...(J); j ++) {
          //   fjs[j][0] = fjs[j][1] = fjs[j][2] = 0.0;
          //   for (int i = 0; i < 8; i ++) {
          //     fjs[j][0] += fxjs[j][i];
          //     fjs[j][1] += fyjs[j][i];
          //     fjs[j][2] += fzjs[j][i];
          //   }
          // }
          // e.start(LSFJ);
          {__attribute__((unused))int a[]{(svst1(lt3, f[js[J]], fjvs[J] - fj[J]), 0)...};}
          // e.stop(LSFJ);
          // e.stop(ADDVJ);

          // {__attribute__((unused)) double a[] = {f[js[J]][0] -= svaddv_f64(incuts[J], fx[J])...};}
          // {__attribute__((unused)) double a[] = {f[js[J]][1] -= svaddv_f64(incuts[J], fy[J])...};}
          // {__attribute__((unused)) double a[] = {f[js[J]][2] -= svaddv_f64(incuts[J], fz[J])...};}
        } else {
          svbool_t sub0 = js[0] < nlocal ? (svbool_t) incuts[0] : svpfalse();
          svbool_t sub1 = js[1] < nlocal ? (svbool_t) incuts[1] : svpfalse();
          svbool_t sub2 = js[2] < nlocal ? (svbool_t) incuts[2] : svpfalse();
          svbool_t sub3 = js[3] < nlocal ? (svbool_t) incuts[3] : svpfalse();
          f[js[0]][0] -= svaddv_f64(sub0, fx[0]);
          f[js[1]][0] -= svaddv_f64(sub1, fx[1]);
          f[js[2]][0] -= svaddv_f64(sub2, fx[2]);
          f[js[3]][0] -= svaddv_f64(sub3, fx[3]);
          f[js[0]][1] -= svaddv_f64(sub0, fy[0]);
          f[js[1]][1] -= svaddv_f64(sub1, fy[1]);
          f[js[2]][1] -= svaddv_f64(sub2, fy[2]);
          f[js[3]][1] -= svaddv_f64(sub3, fy[3]);
          f[js[0]][2] -= svaddv_f64(sub0, fz[0]);
          f[js[1]][2] -= svaddv_f64(sub1, fz[1]);
          f[js[2]][2] -= svaddv_f64(sub2, fz[2]);
          f[js[3]][2] -= svaddv_f64(sub3, fz[3]);
        }
        // e.stop(WRITEJ);
        if (EVFLAG) {
          if (eflag) {
            svd_t vlj3_0 = svld1_f64(svptrue_b64(), param[jtypes[0]][LJ3I]);
            svd_t vlj3_1 = svld1_f64(svptrue_b64(), param[jtypes[1]][LJ3I]);
            svd_t vlj3_2 = svld1_f64(svptrue_b64(), param[jtypes[2]][LJ3I]);
            svd_t vlj3_3 = svld1_f64(svptrue_b64(), param[jtypes[3]][LJ3I]);
            svd_t vlj4_0 = svld1_f64(svptrue_b64(), param[jtypes[0]][LJ4I]);
            svd_t vlj4_1 = svld1_f64(svptrue_b64(), param[jtypes[1]][LJ4I]);
            svd_t vlj4_2 = svld1_f64(svptrue_b64(), param[jtypes[2]][LJ4I]);
            svd_t vlj4_3 = svld1_f64(svptrue_b64(), param[jtypes[3]][LJ4I]);
            svd_t voffset0 = svld1_f64(svptrue_b64(), param[jtypes[0]][OFFSETI]);
            svd_t voffset1 = svld1_f64(svptrue_b64(), param[jtypes[1]][OFFSETI]);
            svd_t voffset2 = svld1_f64(svptrue_b64(), param[jtypes[2]][OFFSETI]);
            svd_t voffset3 = svld1_f64(svptrue_b64(), param[jtypes[3]][OFFSETI]);
            svd_t evdwl0 = r6invs[0] * (vlj3_0 * r6invs[0] - vlj4_0) - voffset0;    // - offset[itype][jtype];
            svd_t evdwl1 = r6invs[1] * (vlj3_1 * r6invs[1] - vlj4_1) - voffset1;    // - offset[itype][jtype];
            svd_t evdwl2 = r6invs[2] * (vlj3_2 * r6invs[2] - vlj4_2) - voffset2;    // - offset[itype][jtype];
            svd_t evdwl3 = r6invs[3] * (vlj3_3 * r6invs[3] - vlj4_3) - voffset3;    // - offset[itype][jtype];
            // evdwl *= factor_lj;
            eng_vdwl += svaddv_f64(incuts[0], evdwl0);
            eng_vdwl += svaddv_f64(incuts[1], evdwl1);
            eng_vdwl += svaddv_f64(incuts[2], evdwl2);
            eng_vdwl += svaddv_f64(incuts[3], evdwl3);
          }
        }
    }
    // e.stop(JLOOP);
    svd_t cfxi = svld1_gather_offset(predi64, (double*)f, is64*24);
    svd_t cfyi = svld1_gather_offset(predi64, (double*)f, is64*24+8);
    svd_t cfzi = svld1_gather_offset(predi64, (double*)f, is64*24+16);
    svst1_scatter_offset(predi64, (double*)f, is64 * 24, cfxi + fxi);
    svst1_scatter_offset(predi64, (double*)f, is64 * 24 + 8, cfyi + fyi);
    svst1_scatter_offset(predi64, (double*)f, is64 * 24 + 16, cfzi + fzi);
  }
  //e.stop(ILOOP);
  //if (update->ntimestep == 900)
  //  e.report(stdout);
  // MPI_Barrier(MPI_COMM_WORLD);
}
template <int MOL, int NEWTON, int EVFLAG> void PairLJCutSVE::eval(int eflag, int vflag)
{
  eval<MOL, NEWTON, EVFLAG>(eflag, vflag, std::make_integer_sequence<int, 4>{});
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
  if (vflag_fdotr) virial_fdotr_compute();
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

//-mcpu=linxicore9100 --aarch64-aggressive-fma-fusion --aarch64-enable-copy-propagation --aarch64-enable-early-sve-libcall-opts  --aarch64-enable-sve-intrinsic-opts --aarch64-enable-sve-libcall-opts --enable-misched --enable-post-misched --enable-reduction-reassoc --enable-spill-copy-elim --misched-limit=10000 

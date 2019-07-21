#include "chemistry_file.H"
#ifdef AMREX_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

/*the molecular weights in g/mol */
void egtransetWT(double* WT ) {
    WT[0] = 1.00797000E+00;
    WT[1] = 3.19988000E+01;
    WT[2] = 1.59994000E+01;
    WT[3] = 1.70073700E+01;
    WT[4] = 2.01594000E+00;
    WT[5] = 1.80153400E+01;
    WT[6] = 4.40099500E+01;
    WT[7] = 3.30067700E+01;
    WT[8] = 3.40147400E+01;
    WT[9] = 2.80105500E+01;
    WT[10] = 2.90185200E+01;
    WT[11] = 1.20111500E+01;
    WT[12] = 1.30191200E+01;
    WT[13] = 1.40270900E+01;
    WT[14] = 1.50350600E+01;
    WT[15] = 3.00264900E+01;
    WT[16] = 4.10296700E+01;
    WT[17] = 2.50302700E+01;
    WT[18] = 4.20376400E+01;
    WT[19] = 2.60382400E+01;
    WT[20] = 1.40270900E+01;
    WT[21] = 3.99480000E+01;
    WT[22] = 3.20424300E+01;
    WT[23] = 3.10344600E+01;
    WT[24] = 3.10344600E+01;
    WT[25] = 1.60430300E+01;
    WT[26] = 4.70338600E+01;
    WT[27] = 2.70462100E+01;
    WT[28] = 2.80541800E+01;
    WT[29] = 2.90621500E+01;
    WT[30] = 4.20376400E+01;
    WT[31] = 4.30456100E+01;
    WT[32] = 4.40535800E+01;
    WT[33] = 2.60382400E+01;
    WT[34] = 4.50615500E+01;
    WT[35] = 4.30892400E+01;
    WT[36] = 3.00701200E+01;
    WT[37] = 4.40972100E+01;
    WT[38] = 4.20812700E+01;
    WT[39] = 3.90573600E+01;
    WT[40] = 4.00653300E+01;
    WT[41] = 4.00653300E+01;
    WT[42] = 4.10733000E+01;
    WT[43] = 5.10685100E+01;
    WT[44] = 5.60647300E+01;
    WT[45] = 4.10733000E+01;
    WT[46] = 4.00217000E+01;
    WT[47] = 5.20764800E+01;
    WT[48] = 3.80493900E+01;
    WT[49] = 5.40487900E+01;
    WT[50] = 5.00605400E+01;
    WT[51] = 5.10685100E+01;
    WT[52] = 4.10733000E+01;
    WT[53] = 5.70727000E+01;
    WT[54] = 4.90525700E+01;
    WT[55] = 9.81051400E+01;
    WT[56] = 7.40828400E+01;
    WT[57] = 5.40924200E+01;
    WT[58] = 5.30844500E+01;
    WT[59] = 5.30844500E+01;
    WT[60] = 7.81147200E+01;
    WT[61] = 1.00205570E+02;
    WT[62] = 7.11434200E+01;
    WT[63] = 5.71163300E+01;
    WT[64] = 9.91976000E+01;
    WT[65] = 5.61083600E+01;
    WT[66] = 7.01354500E+01;
    WT[67] = 9.81896300E+01;
    WT[68] = 1.15197000E+02;
    WT[69] = 7.21077600E+01;
    WT[70] = 5.51003900E+01;
    WT[71] = 9.71816600E+01;
    WT[72] = 6.91274800E+01;
    WT[73] = 7.10997900E+01;
    WT[74] = 5.90886400E+01;
    WT[75] = 1.14232660E+02;
    WT[76] = 9.91976000E+01;
    WT[77] = 5.61083600E+01;
    WT[78] = 4.30892400E+01;
    WT[79] = 5.71163300E+01;
    WT[80] = 1.13224690E+02;
    WT[81] = 9.81896300E+01;
    WT[82] = 1.29224090E+02;
    WT[83] = 5.80806700E+01;
    WT[84] = 5.51003900E+01;
    WT[85] = 9.71816600E+01;
    WT[86] = 7.00918200E+01;
    WT[87] = 7.31157300E+01;
    WT[88] = 7.10997900E+01;
    WT[89] = 7.81147200E+01;
    WT[90] = 7.71067500E+01;
    WT[91] = 1.03144990E+02;
    WT[92] = 1.04152960E+02;
    WT[93] = 1.02137020E+02;
    WT[94] = 1.01129050E+02;
    WT[95] = 1.03144990E+02;
    WT[96] = 1.27167290E+02;
    WT[97] = 1.28175260E+02;
    WT[98] = 1.27167290E+02;
    WT[99] = 1.53205530E+02;
    WT[100] = 1.53205530E+02;
    WT[101] = 1.52197560E+02;
    WT[102] = 1.52197560E+02;
    WT[103] = 1.51189590E+02;
    WT[104] = 1.51189590E+02;
    WT[105] = 1.52197560E+02;
    WT[106] = 1.51189590E+02;
    WT[107] = 1.77227830E+02;
    WT[108] = 1.76219860E+02;
    WT[109] = 1.75211890E+02;
    WT[110] = 1.54213500E+02;
    WT[111] = 1.53205530E+02;
    WT[112] = 1.77227830E+02;
    WT[113] = 1.78235800E+02;
    WT[114] = 1.77227830E+02;
    WT[115] = 2.01250130E+02;
    WT[116] = 2.02258100E+02;
    WT[117] = 2.02258100E+02;
    WT[118] = 2.01250130E+02;
    WT[119] = 2.26280400E+02;
    WT[120] = 2.02258100E+02;
    WT[121] = 6.61035700E+01;
    WT[122] = 6.50956000E+01;
    WT[123] = 8.10950000E+01;
    WT[124] = 8.00870300E+01;
    WT[125] = 8.10950000E+01;
    WT[126] = 1.16164110E+02;
    WT[127] = 1.15156140E+02;
    WT[128] = 9.11338400E+01;
    WT[129] = 1.30147570E+02;
    WT[130] = 7.60987800E+01;
    WT[131] = 9.21418100E+01;
    WT[132] = 9.41141200E+01;
    WT[133] = 1.08141210E+02;
    WT[134] = 1.07133240E+02;
    WT[135] = 1.07133240E+02;
    WT[136] = 1.08141210E+02;
    WT[137] = 1.06125270E+02;
    WT[138] = 9.31061500E+01;
    WT[139] = 9.11338400E+01;
    WT[140] = 1.05160930E+02;
    WT[141] = 1.06168900E+02;
    WT[142] = 1.37159730E+02;
    WT[143] = 1.37159730E+02;
    WT[144] = 1.52151160E+02;
    WT[145] = 1.06168900E+02;
    WT[146] = 1.05160930E+02;
    WT[147] = 1.20152360E+02;
    WT[148] = 1.42202350E+02;
    WT[149] = 1.19144390E+02;
    WT[150] = 1.34135820E+02;
    WT[151] = 1.44174660E+02;
    WT[152] = 1.41194380E+02;
    WT[153] = 1.57193780E+02;
    WT[154] = 1.56185810E+02;
    WT[155] = 1.43166690E+02;
    WT[156] = 1.08097580E+02;
    WT[157] = 2.80134000E+01;
}


/*the lennard-jones potential well depth eps/kb in K */
void egtransetEPS(double* EPS ) {
    EPS[139] = 4.95300000E+02;
    EPS[118] = 8.34900000E+02;
    EPS[97] = 6.30400000E+02;
    EPS[140] = 5.46200000E+02;
    EPS[119] = 8.34900000E+02;
    EPS[98] = 6.30400000E+02;
    EPS[141] = 5.46200000E+02;
    EPS[120] = 8.34900000E+02;
    EPS[99] = 6.93100000E+02;
    EPS[142] = 5.46200000E+02;
    EPS[121] = 4.00000000E+02;
    EPS[100] = 6.93100000E+02;
    EPS[157] = 9.75300000E+01;
    EPS[0] = 1.45000000E+02;
    EPS[143] = 5.46200000E+02;
    EPS[122] = 4.00000000E+02;
    EPS[1] = 1.07400000E+02;
    EPS[101] = 6.93100000E+02;
    EPS[2] = 8.00000000E+01;
    EPS[3] = 8.00000000E+01;
    EPS[144] = 5.46200000E+02;
    EPS[4] = 3.80000000E+01;
    EPS[5] = 5.72400000E+02;
    EPS[123] = 4.84000000E+02;
    EPS[6] = 2.44000000E+02;
    EPS[7] = 1.07400000E+02;
    EPS[102] = 6.93100000E+02;
    EPS[8] = 1.07400000E+02;
    EPS[9] = 9.81000000E+01;
    EPS[145] = 5.46200000E+02;
    EPS[10] = 4.98000000E+02;
    EPS[11] = 7.14000000E+01;
    EPS[124] = 4.84000000E+02;
    EPS[12] = 8.00000000E+01;
    EPS[13] = 1.44000000E+02;
    EPS[103] = 6.93100000E+02;
    EPS[14] = 1.44000000E+02;
    EPS[15] = 4.98000000E+02;
    EPS[146] = 5.46200000E+02;
    EPS[16] = 1.50000000E+02;
    EPS[17] = 2.09000000E+02;
    EPS[125] = 6.17000000E+02;
    EPS[18] = 4.36000000E+02;
    EPS[19] = 2.09000000E+02;
    EPS[104] = 6.93100000E+02;
    EPS[20] = 1.44000000E+02;
    EPS[21] = 1.36500000E+02;
    EPS[147] = 5.46200000E+02;
    EPS[22] = 4.81800000E+02;
    EPS[23] = 4.17000000E+02;
    EPS[126] = 6.30400000E+02;
    EPS[24] = 4.17000000E+02;
    EPS[25] = 1.41400000E+02;
    EPS[105] = 6.93100000E+02;
    EPS[26] = 4.81800000E+02;
    EPS[27] = 2.09000000E+02;
    EPS[148] = 6.93100000E+02;
    EPS[28] = 2.80800000E+02;
    EPS[29] = 2.52300000E+02;
    EPS[127] = 6.30400000E+02;
    EPS[30] = 4.36000000E+02;
    EPS[31] = 4.36000000E+02;
    EPS[106] = 6.93100000E+02;
    EPS[32] = 4.36000000E+02;
    EPS[33] = 2.09000000E+02;
    EPS[149] = 5.46200000E+02;
    EPS[34] = 4.70600000E+02;
    EPS[128] = 4.95300000E+02;
    EPS[35] = 2.66800000E+02;
    EPS[36] = 2.52300000E+02;
    EPS[107] = 7.72000000E+02;
    EPS[37] = 2.66800000E+02;
    EPS[38] = 2.66800000E+02;
    EPS[150] = 5.46200000E+02;
    EPS[39] = 2.52000000E+02;
    EPS[40] = 2.52000000E+02;
    EPS[129] = 6.30400000E+02;
    EPS[41] = 2.52000000E+02;
    EPS[42] = 2.66800000E+02;
    EPS[108] = 7.72800000E+02;
    EPS[43] = 3.57000000E+02;
    EPS[44] = 4.28800000E+02;
    EPS[151] = 6.30400000E+02;
    EPS[45] = 2.66800000E+02;
    EPS[46] = 2.32400000E+02;
    EPS[130] = 4.64800000E+02;
    EPS[47] = 3.57000000E+02;
    EPS[48] = 2.09000000E+02;
    EPS[109] = 7.72800000E+02;
    EPS[49] = 2.52000000E+02;
    EPS[50] = 3.57000000E+02;
    EPS[152] = 6.93100000E+02;
    EPS[51] = 3.57000000E+02;
    EPS[52] = 2.66800000E+02;
    EPS[131] = 4.95300000E+02;
    EPS[53] = 4.11000000E+02;
    EPS[54] = 3.57000000E+02;
    EPS[110] = 6.76500000E+02;
    EPS[55] = 4.95300000E+02;
    EPS[56] = 4.95300000E+02;
    EPS[153] = 6.93100000E+02;
    EPS[57] = 3.57000000E+02;
    EPS[58] = 3.57000000E+02;
    EPS[132] = 4.10000000E+02;
    EPS[59] = 3.57000000E+02;
    EPS[60] = 4.64800000E+02;
    EPS[111] = 6.76500000E+02;
    EPS[61] = 4.59600000E+02;
    EPS[62] = 4.40735000E+02;
    EPS[154] = 6.93100000E+02;
    EPS[63] = 3.52000000E+02;
    EPS[64] = 4.59600000E+02;
    EPS[133] = 4.95300000E+02;
    EPS[65] = 3.45700000E+02;
    EPS[66] = 3.86200000E+02;
    EPS[112] = 7.72000000E+02;
    EPS[67] = 4.57800000E+02;
    EPS[68] = 5.61000000E+02;
    EPS[155] = 6.30400000E+02;
    EPS[69] = 4.64200000E+02;
    EPS[70] = 3.55000000E+02;
    EPS[134] = 4.95300000E+02;
    EPS[71] = 4.57800000E+02;
    EPS[72] = 3.96800000E+02;
    EPS[113] = 7.72000000E+02;
    EPS[73] = 4.96000000E+02;
    EPS[74] = 4.81500000E+02;
    EPS[156] = 4.64800000E+02;
    EPS[75] = 4.58500000E+02;
    EPS[76] = 4.37300000E+02;
    EPS[135] = 4.95300000E+02;
    EPS[77] = 3.44500000E+02;
    EPS[78] = 3.03400000E+02;
    EPS[114] = 7.72000000E+02;
    EPS[79] = 3.52000000E+02;
    EPS[80] = 4.58500000E+02;
    EPS[81] = 4.39200000E+02;
    EPS[82] = 5.81300000E+02;
    EPS[136] = 4.95300000E+02;
    EPS[83] = 4.35500000E+02;
    EPS[84] = 3.55000000E+02;
    EPS[115] = 8.37500000E+02;
    EPS[85] = 4.39200000E+02;
    EPS[86] = 4.36400000E+02;
    EPS[87] = 4.96000000E+02;
    EPS[88] = 4.96000000E+02;
    EPS[137] = 4.95300000E+02;
    EPS[89] = 4.64800000E+02;
    EPS[90] = 4.64800000E+02;
    EPS[116] = 8.37500000E+02;
    EPS[91] = 5.46200000E+02;
    EPS[92] = 5.46200000E+02;
    EPS[93] = 5.35600000E+02;
    EPS[94] = 5.35600000E+02;
    EPS[138] = 4.10000000E+02;
    EPS[95] = 5.46200000E+02;
    EPS[96] = 6.30400000E+02;
    EPS[117] = 8.34900000E+02;
}


/*the lennard-jones collision diameter in Angstroms */
void egtransetSIG(double* SIG ) {
    SIG[139] = 5.68000000E+00;
    SIG[118] = 7.24000000E+00;
    SIG[97] = 6.18000000E+00;
    SIG[140] = 6.00000000E+00;
    SIG[119] = 7.24000000E+00;
    SIG[98] = 6.18000000E+00;
    SIG[141] = 6.00000000E+00;
    SIG[120] = 7.24000000E+00;
    SIG[99] = 6.47000000E+00;
    SIG[142] = 6.00000000E+00;
    SIG[121] = 5.20000000E+00;
    SIG[100] = 6.47000000E+00;
    SIG[157] = 3.62100000E+00;
    SIG[0] = 2.05000000E+00;
    SIG[143] = 6.00000000E+00;
    SIG[122] = 5.20000000E+00;
    SIG[1] = 3.45800000E+00;
    SIG[101] = 6.47000000E+00;
    SIG[2] = 2.75000000E+00;
    SIG[3] = 2.75000000E+00;
    SIG[144] = 6.00000000E+00;
    SIG[4] = 2.92000000E+00;
    SIG[5] = 2.60500000E+00;
    SIG[123] = 5.10000000E+00;
    SIG[6] = 3.76300000E+00;
    SIG[7] = 3.45800000E+00;
    SIG[102] = 6.47000000E+00;
    SIG[8] = 3.45800000E+00;
    SIG[9] = 3.65000000E+00;
    SIG[145] = 6.00000000E+00;
    SIG[10] = 3.59000000E+00;
    SIG[11] = 3.29800000E+00;
    SIG[124] = 5.10000000E+00;
    SIG[12] = 2.75000000E+00;
    SIG[13] = 3.80000000E+00;
    SIG[103] = 6.47000000E+00;
    SIG[14] = 3.80000000E+00;
    SIG[15] = 3.59000000E+00;
    SIG[146] = 6.00000000E+00;
    SIG[16] = 2.50000000E+00;
    SIG[17] = 4.10000000E+00;
    SIG[125] = 5.82000000E+00;
    SIG[18] = 3.97000000E+00;
    SIG[19] = 4.10000000E+00;
    SIG[104] = 6.47000000E+00;
    SIG[20] = 3.80000000E+00;
    SIG[21] = 3.33000000E+00;
    SIG[147] = 6.00000000E+00;
    SIG[22] = 3.62600000E+00;
    SIG[23] = 3.69000000E+00;
    SIG[126] = 6.18000000E+00;
    SIG[24] = 3.69000000E+00;
    SIG[25] = 3.74600000E+00;
    SIG[105] = 6.47000000E+00;
    SIG[26] = 3.62600000E+00;
    SIG[27] = 4.10000000E+00;
    SIG[148] = 6.47000000E+00;
    SIG[28] = 3.97100000E+00;
    SIG[29] = 4.30200000E+00;
    SIG[127] = 6.18000000E+00;
    SIG[30] = 3.97000000E+00;
    SIG[31] = 3.97000000E+00;
    SIG[106] = 6.47000000E+00;
    SIG[32] = 3.97000000E+00;
    SIG[33] = 4.10000000E+00;
    SIG[149] = 6.00000000E+00;
    SIG[34] = 4.41000000E+00;
    SIG[128] = 5.68000000E+00;
    SIG[35] = 4.98200000E+00;
    SIG[36] = 4.30200000E+00;
    SIG[107] = 6.96000000E+00;
    SIG[37] = 4.98200000E+00;
    SIG[38] = 4.98200000E+00;
    SIG[150] = 6.00000000E+00;
    SIG[39] = 4.76000000E+00;
    SIG[40] = 4.76000000E+00;
    SIG[129] = 6.18000000E+00;
    SIG[41] = 4.76000000E+00;
    SIG[42] = 4.98200000E+00;
    SIG[108] = 6.94000000E+00;
    SIG[43] = 5.18000000E+00;
    SIG[44] = 4.95800000E+00;
    SIG[151] = 6.18000000E+00;
    SIG[45] = 4.98200000E+00;
    SIG[46] = 3.82800000E+00;
    SIG[130] = 5.29000000E+00;
    SIG[47] = 5.18000000E+00;
    SIG[48] = 4.10000000E+00;
    SIG[109] = 6.94000000E+00;
    SIG[49] = 4.76000000E+00;
    SIG[50] = 5.18000000E+00;
    SIG[152] = 6.47000000E+00;
    SIG[51] = 5.18000000E+00;
    SIG[52] = 4.98200000E+00;
    SIG[131] = 5.68000000E+00;
    SIG[53] = 4.82000000E+00;
    SIG[54] = 5.18000000E+00;
    SIG[110] = 6.31000000E+00;
    SIG[55] = 5.68000000E+00;
    SIG[56] = 5.68000000E+00;
    SIG[153] = 6.47000000E+00;
    SIG[57] = 5.18000000E+00;
    SIG[58] = 5.18000000E+00;
    SIG[132] = 5.92000000E+00;
    SIG[59] = 5.18000000E+00;
    SIG[60] = 5.29000000E+00;
    SIG[111] = 6.31000000E+00;
    SIG[61] = 6.25300000E+00;
    SIG[62] = 5.04100000E+00;
    SIG[154] = 6.47000000E+00;
    SIG[63] = 5.24000000E+00;
    SIG[64] = 6.25300000E+00;
    SIG[133] = 5.68000000E+00;
    SIG[65] = 5.08800000E+00;
    SIG[66] = 5.48900000E+00;
    SIG[112] = 6.96000000E+00;
    SIG[67] = 6.17300000E+00;
    SIG[68] = 6.31700000E+00;
    SIG[155] = 6.18000000E+00;
    SIG[69] = 5.00900000E+00;
    SIG[70] = 4.65000000E+00;
    SIG[134] = 5.68000000E+00;
    SIG[71] = 6.17300000E+00;
    SIG[72] = 5.45800000E+00;
    SIG[113] = 6.96000000E+00;
    SIG[73] = 5.20000000E+00;
    SIG[74] = 4.99700000E+00;
    SIG[156] = 5.29000000E+00;
    SIG[75] = 6.41400000E+00;
    SIG[76] = 6.16800000E+00;
    SIG[135] = 5.68000000E+00;
    SIG[77] = 5.08900000E+00;
    SIG[78] = 4.81000000E+00;
    SIG[114] = 6.96000000E+00;
    SIG[79] = 5.24000000E+00;
    SIG[80] = 6.41400000E+00;
    SIG[81] = 6.15100000E+00;
    SIG[82] = 6.50600000E+00;
    SIG[136] = 5.68000000E+00;
    SIG[83] = 4.86000000E+00;
    SIG[84] = 4.65000000E+00;
    SIG[115] = 7.28000000E+00;
    SIG[85] = 6.15100000E+00;
    SIG[86] = 5.35200000E+00;
    SIG[87] = 5.20000000E+00;
    SIG[88] = 5.20000000E+00;
    SIG[137] = 5.68000000E+00;
    SIG[89] = 5.29000000E+00;
    SIG[90] = 5.29000000E+00;
    SIG[116] = 7.28000000E+00;
    SIG[91] = 6.00000000E+00;
    SIG[92] = 6.00000000E+00;
    SIG[93] = 5.72000000E+00;
    SIG[94] = 5.72000000E+00;
    SIG[138] = 5.92000000E+00;
    SIG[95] = 6.00000000E+00;
    SIG[96] = 6.18000000E+00;
    SIG[117] = 7.24000000E+00;
}


/*the dipole moment in Debye */
void egtransetDIP(double* DIP ) {
    DIP[139] = 4.30000000E-01;
    DIP[118] = 0.00000000E+00;
    DIP[97] = 0.00000000E+00;
    DIP[140] = 1.30000000E-01;
    DIP[119] = 0.00000000E+00;
    DIP[98] = 0.00000000E+00;
    DIP[141] = 1.30000000E-01;
    DIP[120] = 0.00000000E+00;
    DIP[99] = 0.00000000E+00;
    DIP[142] = 1.30000000E-01;
    DIP[121] = 0.00000000E+00;
    DIP[100] = 0.00000000E+00;
    DIP[157] = 0.00000000E+00;
    DIP[0] = 0.00000000E+00;
    DIP[143] = 1.30000000E-01;
    DIP[122] = 0.00000000E+00;
    DIP[1] = 0.00000000E+00;
    DIP[101] = 0.00000000E+00;
    DIP[2] = 0.00000000E+00;
    DIP[3] = 0.00000000E+00;
    DIP[144] = 1.30000000E-01;
    DIP[4] = 0.00000000E+00;
    DIP[5] = 1.84400000E+00;
    DIP[123] = 0.00000000E+00;
    DIP[6] = 0.00000000E+00;
    DIP[7] = 0.00000000E+00;
    DIP[102] = 0.00000000E+00;
    DIP[8] = 0.00000000E+00;
    DIP[9] = 0.00000000E+00;
    DIP[145] = 1.30000000E-01;
    DIP[10] = 0.00000000E+00;
    DIP[11] = 0.00000000E+00;
    DIP[124] = 0.00000000E+00;
    DIP[12] = 0.00000000E+00;
    DIP[13] = 0.00000000E+00;
    DIP[103] = 0.00000000E+00;
    DIP[14] = 0.00000000E+00;
    DIP[15] = 0.00000000E+00;
    DIP[146] = 1.30000000E-01;
    DIP[16] = 0.00000000E+00;
    DIP[17] = 0.00000000E+00;
    DIP[125] = 0.00000000E+00;
    DIP[18] = 0.00000000E+00;
    DIP[19] = 0.00000000E+00;
    DIP[104] = 0.00000000E+00;
    DIP[20] = 0.00000000E+00;
    DIP[21] = 0.00000000E+00;
    DIP[147] = 1.30000000E-01;
    DIP[22] = 0.00000000E+00;
    DIP[23] = 1.70000000E+00;
    DIP[126] = 0.00000000E+00;
    DIP[24] = 1.70000000E+00;
    DIP[25] = 0.00000000E+00;
    DIP[105] = 0.00000000E+00;
    DIP[26] = 0.00000000E+00;
    DIP[27] = 0.00000000E+00;
    DIP[148] = 0.00000000E+00;
    DIP[28] = 0.00000000E+00;
    DIP[29] = 0.00000000E+00;
    DIP[127] = 0.00000000E+00;
    DIP[30] = 0.00000000E+00;
    DIP[31] = 0.00000000E+00;
    DIP[106] = 0.00000000E+00;
    DIP[32] = 0.00000000E+00;
    DIP[33] = 0.00000000E+00;
    DIP[149] = 1.30000000E-01;
    DIP[34] = 0.00000000E+00;
    DIP[128] = 4.30000000E-01;
    DIP[35] = 0.00000000E+00;
    DIP[36] = 0.00000000E+00;
    DIP[107] = 0.00000000E+00;
    DIP[37] = 0.00000000E+00;
    DIP[38] = 0.00000000E+00;
    DIP[150] = 1.30000000E-01;
    DIP[39] = 0.00000000E+00;
    DIP[40] = 0.00000000E+00;
    DIP[129] = 0.00000000E+00;
    DIP[41] = 0.00000000E+00;
    DIP[42] = 0.00000000E+00;
    DIP[108] = 0.00000000E+00;
    DIP[43] = 0.00000000E+00;
    DIP[44] = 2.90000000E+00;
    DIP[151] = 0.00000000E+00;
    DIP[45] = 0.00000000E+00;
    DIP[46] = 0.00000000E+00;
    DIP[130] = 0.00000000E+00;
    DIP[47] = 0.00000000E+00;
    DIP[48] = 0.00000000E+00;
    DIP[109] = 0.00000000E+00;
    DIP[49] = 0.00000000E+00;
    DIP[50] = 0.00000000E+00;
    DIP[152] = 0.00000000E+00;
    DIP[51] = 0.00000000E+00;
    DIP[52] = 0.00000000E+00;
    DIP[131] = 4.30000000E-01;
    DIP[53] = 0.00000000E+00;
    DIP[54] = 0.00000000E+00;
    DIP[110] = 0.00000000E+00;
    DIP[55] = 4.30000000E-01;
    DIP[56] = 4.30000000E-01;
    DIP[153] = 0.00000000E+00;
    DIP[57] = 0.00000000E+00;
    DIP[58] = 0.00000000E+00;
    DIP[132] = 0.00000000E+00;
    DIP[59] = 0.00000000E+00;
    DIP[60] = 0.00000000E+00;
    DIP[111] = 0.00000000E+00;
    DIP[61] = 0.00000000E+00;
    DIP[62] = 0.00000000E+00;
    DIP[154] = 0.00000000E+00;
    DIP[63] = 0.00000000E+00;
    DIP[64] = 0.00000000E+00;
    DIP[133] = 4.30000000E-01;
    DIP[65] = 3.00000000E-01;
    DIP[66] = 4.00000000E-01;
    DIP[112] = 0.00000000E+00;
    DIP[67] = 3.00000000E-01;
    DIP[68] = 1.70000000E+00;
    DIP[155] = 0.00000000E+00;
    DIP[69] = 2.60000000E+00;
    DIP[70] = 0.00000000E+00;
    DIP[134] = 4.30000000E-01;
    DIP[71] = 3.00000000E-01;
    DIP[72] = 0.00000000E+00;
    DIP[113] = 0.00000000E+00;
    DIP[73] = 0.00000000E+00;
    DIP[74] = 0.00000000E+00;
    DIP[156] = 0.00000000E+00;
    DIP[75] = 0.00000000E+00;
    DIP[76] = 0.00000000E+00;
    DIP[135] = 4.30000000E-01;
    DIP[77] = 0.00000000E+00;
    DIP[78] = 0.00000000E+00;
    DIP[114] = 0.00000000E+00;
    DIP[79] = 0.00000000E+00;
    DIP[80] = 0.00000000E+00;
    DIP[81] = 0.00000000E+00;
    DIP[82] = 0.00000000E+00;
    DIP[136] = 4.30000000E-01;
    DIP[83] = 0.00000000E+00;
    DIP[84] = 0.00000000E+00;
    DIP[115] = 0.00000000E+00;
    DIP[85] = 0.00000000E+00;
    DIP[86] = 0.00000000E+00;
    DIP[87] = 0.00000000E+00;
    DIP[88] = 0.00000000E+00;
    DIP[137] = 4.30000000E-01;
    DIP[89] = 0.00000000E+00;
    DIP[90] = 0.00000000E+00;
    DIP[116] = 0.00000000E+00;
    DIP[91] = 1.30000000E-01;
    DIP[92] = 1.30000000E-01;
    DIP[93] = 7.70000000E-01;
    DIP[94] = 7.70000000E-01;
    DIP[138] = 0.00000000E+00;
    DIP[95] = 1.30000000E-01;
    DIP[96] = 0.00000000E+00;
    DIP[117] = 0.00000000E+00;
}


/*the polarizability in cubic Angstroms */
void egtransetPOL(double* POL ) {
    POL[139] = 1.23000000E+01;
    POL[118] = 4.50000000E+01;
    POL[97] = 1.65000000E+01;
    POL[140] = 1.50000000E+01;
    POL[119] = 4.50000000E+01;
    POL[98] = 1.65000000E+01;
    POL[141] = 1.50000000E+01;
    POL[120] = 4.50000000E+01;
    POL[99] = 1.80000000E+01;
    POL[142] = 1.50000000E+01;
    POL[121] = 0.00000000E+00;
    POL[100] = 1.80000000E+01;
    POL[157] = 1.76000000E+00;
    POL[0] = 0.00000000E+00;
    POL[143] = 1.50000000E+01;
    POL[122] = 0.00000000E+00;
    POL[1] = 1.60000000E+00;
    POL[101] = 1.80000000E+01;
    POL[2] = 0.00000000E+00;
    POL[3] = 0.00000000E+00;
    POL[144] = 1.50000000E+01;
    POL[4] = 7.90000000E-01;
    POL[5] = 0.00000000E+00;
    POL[123] = 0.00000000E+00;
    POL[6] = 2.65000000E+00;
    POL[7] = 0.00000000E+00;
    POL[102] = 1.80000000E+01;
    POL[8] = 0.00000000E+00;
    POL[9] = 1.95000000E+00;
    POL[145] = 1.50000000E+01;
    POL[10] = 0.00000000E+00;
    POL[11] = 0.00000000E+00;
    POL[124] = 0.00000000E+00;
    POL[12] = 0.00000000E+00;
    POL[13] = 0.00000000E+00;
    POL[103] = 1.80000000E+01;
    POL[14] = 0.00000000E+00;
    POL[15] = 0.00000000E+00;
    POL[146] = 1.50000000E+01;
    POL[16] = 0.00000000E+00;
    POL[17] = 0.00000000E+00;
    POL[125] = 0.00000000E+00;
    POL[18] = 0.00000000E+00;
    POL[19] = 0.00000000E+00;
    POL[104] = 1.80000000E+01;
    POL[20] = 0.00000000E+00;
    POL[21] = 0.00000000E+00;
    POL[147] = 1.50000000E+01;
    POL[22] = 0.00000000E+00;
    POL[23] = 0.00000000E+00;
    POL[126] = 1.65000000E+01;
    POL[24] = 0.00000000E+00;
    POL[25] = 2.60000000E+00;
    POL[105] = 1.80000000E+01;
    POL[26] = 0.00000000E+00;
    POL[27] = 0.00000000E+00;
    POL[148] = 1.80000000E+01;
    POL[28] = 0.00000000E+00;
    POL[29] = 0.00000000E+00;
    POL[127] = 1.65000000E+01;
    POL[30] = 0.00000000E+00;
    POL[31] = 0.00000000E+00;
    POL[106] = 1.80000000E+01;
    POL[32] = 0.00000000E+00;
    POL[33] = 0.00000000E+00;
    POL[149] = 1.50000000E+01;
    POL[34] = 0.00000000E+00;
    POL[128] = 1.23000000E+01;
    POL[35] = 0.00000000E+00;
    POL[36] = 0.00000000E+00;
    POL[107] = 3.88000000E+01;
    POL[37] = 0.00000000E+00;
    POL[38] = 0.00000000E+00;
    POL[150] = 1.50000000E+01;
    POL[39] = 0.00000000E+00;
    POL[40] = 0.00000000E+00;
    POL[129] = 1.65000000E+01;
    POL[41] = 0.00000000E+00;
    POL[42] = 0.00000000E+00;
    POL[108] = 1.80000000E+01;
    POL[43] = 0.00000000E+00;
    POL[44] = 0.00000000E+00;
    POL[151] = 1.65000000E+01;
    POL[45] = 0.00000000E+00;
    POL[46] = 0.00000000E+00;
    POL[130] = 1.03200000E+01;
    POL[47] = 0.00000000E+00;
    POL[48] = 0.00000000E+00;
    POL[109] = 1.80000000E+01;
    POL[49] = 0.00000000E+00;
    POL[50] = 0.00000000E+00;
    POL[152] = 1.80000000E+01;
    POL[51] = 0.00000000E+00;
    POL[52] = 0.00000000E+00;
    POL[131] = 1.23000000E+01;
    POL[53] = 0.00000000E+00;
    POL[54] = 0.00000000E+00;
    POL[110] = 2.00000000E+01;
    POL[55] = 1.23000000E+01;
    POL[56] = 1.23000000E+01;
    POL[153] = 1.80000000E+01;
    POL[57] = 0.00000000E+00;
    POL[58] = 0.00000000E+00;
    POL[132] = 0.00000000E+00;
    POL[59] = 0.00000000E+00;
    POL[60] = 1.03200000E+01;
    POL[111] = 2.00000000E+01;
    POL[61] = 0.00000000E+00;
    POL[62] = 0.00000000E+00;
    POL[154] = 1.80000000E+01;
    POL[63] = 0.00000000E+00;
    POL[64] = 0.00000000E+00;
    POL[133] = 1.23000000E+01;
    POL[65] = 0.00000000E+00;
    POL[66] = 0.00000000E+00;
    POL[112] = 3.88000000E+01;
    POL[67] = 0.00000000E+00;
    POL[68] = 0.00000000E+00;
    POL[155] = 1.65000000E+01;
    POL[69] = 0.00000000E+00;
    POL[70] = 0.00000000E+00;
    POL[134] = 1.23000000E+01;
    POL[71] = 0.00000000E+00;
    POL[72] = 0.00000000E+00;
    POL[113] = 3.88000000E+01;
    POL[73] = 0.00000000E+00;
    POL[74] = 0.00000000E+00;
    POL[156] = 1.03200000E+01;
    POL[75] = 0.00000000E+00;
    POL[76] = 0.00000000E+00;
    POL[135] = 1.23000000E+01;
    POL[77] = 0.00000000E+00;
    POL[78] = 0.00000000E+00;
    POL[114] = 3.88000000E+01;
    POL[79] = 0.00000000E+00;
    POL[80] = 0.00000000E+00;
    POL[81] = 0.00000000E+00;
    POL[82] = 0.00000000E+00;
    POL[136] = 1.23000000E+01;
    POL[83] = 0.00000000E+00;
    POL[84] = 0.00000000E+00;
    POL[115] = 0.00000000E+00;
    POL[85] = 0.00000000E+00;
    POL[86] = 0.00000000E+00;
    POL[87] = 0.00000000E+00;
    POL[88] = 0.00000000E+00;
    POL[137] = 1.23000000E+01;
    POL[89] = 1.03200000E+01;
    POL[90] = 1.03200000E+01;
    POL[116] = 0.00000000E+00;
    POL[91] = 1.50000000E+01;
    POL[92] = 1.50000000E+01;
    POL[93] = 1.20000000E+01;
    POL[94] = 1.20000000E+01;
    POL[138] = 0.00000000E+00;
    POL[95] = 1.50000000E+01;
    POL[96] = 1.65000000E+01;
    POL[117] = 4.50000000E+01;
}


/*the rotational relaxation collision number at 298 K */
void egtransetZROT(double* ZROT ) {
    ZROT[139] = 1.00000000E+00;
    ZROT[118] = 1.00000000E+00;
    ZROT[97] = 1.00000000E+00;
    ZROT[140] = 1.00000000E+00;
    ZROT[119] = 1.00000000E+00;
    ZROT[98] = 1.00000000E+00;
    ZROT[141] = 1.00000000E+00;
    ZROT[120] = 1.00000000E+00;
    ZROT[99] = 1.00000000E+00;
    ZROT[142] = 1.00000000E+00;
    ZROT[121] = 0.00000000E+00;
    ZROT[100] = 1.00000000E+00;
    ZROT[157] = 4.00000000E+00;
    ZROT[0] = 0.00000000E+00;
    ZROT[143] = 1.00000000E+00;
    ZROT[122] = 1.00000000E+00;
    ZROT[1] = 3.80000000E+00;
    ZROT[101] = 1.00000000E+00;
    ZROT[2] = 0.00000000E+00;
    ZROT[3] = 0.00000000E+00;
    ZROT[144] = 1.00000000E+00;
    ZROT[4] = 2.80000000E+02;
    ZROT[5] = 4.00000000E+00;
    ZROT[123] = 0.00000000E+00;
    ZROT[6] = 2.10000000E+00;
    ZROT[7] = 1.00000000E+00;
    ZROT[102] = 1.00000000E+00;
    ZROT[8] = 3.80000000E+00;
    ZROT[9] = 1.80000000E+00;
    ZROT[145] = 1.00000000E+00;
    ZROT[10] = 0.00000000E+00;
    ZROT[11] = 0.00000000E+00;
    ZROT[124] = 0.00000000E+00;
    ZROT[12] = 0.00000000E+00;
    ZROT[13] = 0.00000000E+00;
    ZROT[103] = 1.00000000E+00;
    ZROT[14] = 0.00000000E+00;
    ZROT[15] = 2.00000000E+00;
    ZROT[146] = 1.00000000E+00;
    ZROT[16] = 1.00000000E+00;
    ZROT[17] = 2.50000000E+00;
    ZROT[125] = 1.00000000E+00;
    ZROT[18] = 2.00000000E+00;
    ZROT[19] = 2.50000000E+00;
    ZROT[104] = 1.00000000E+00;
    ZROT[20] = 0.00000000E+00;
    ZROT[21] = 0.00000000E+00;
    ZROT[147] = 1.00000000E+00;
    ZROT[22] = 1.00000000E+00;
    ZROT[23] = 2.00000000E+00;
    ZROT[126] = 1.00000000E+00;
    ZROT[24] = 2.00000000E+00;
    ZROT[25] = 1.30000000E+01;
    ZROT[105] = 1.00000000E+00;
    ZROT[26] = 1.00000000E+00;
    ZROT[27] = 1.00000000E+00;
    ZROT[148] = 1.00000000E+00;
    ZROT[28] = 1.50000000E+00;
    ZROT[29] = 1.50000000E+00;
    ZROT[127] = 1.00000000E+00;
    ZROT[30] = 2.00000000E+00;
    ZROT[31] = 2.00000000E+00;
    ZROT[106] = 1.00000000E+00;
    ZROT[32] = 2.00000000E+00;
    ZROT[33] = 2.50000000E+00;
    ZROT[149] = 1.00000000E+00;
    ZROT[34] = 1.50000000E+00;
    ZROT[128] = 1.00000000E+00;
    ZROT[35] = 1.00000000E+00;
    ZROT[36] = 1.50000000E+00;
    ZROT[107] = 1.00000000E+00;
    ZROT[37] = 1.00000000E+00;
    ZROT[38] = 1.00000000E+00;
    ZROT[150] = 1.00000000E+00;
    ZROT[39] = 1.00000000E+00;
    ZROT[40] = 1.00000000E+00;
    ZROT[129] = 1.00000000E+00;
    ZROT[41] = 1.00000000E+00;
    ZROT[42] = 1.00000000E+00;
    ZROT[108] = 1.00000000E+00;
    ZROT[43] = 1.00000000E+00;
    ZROT[44] = 1.00000000E+00;
    ZROT[151] = 1.00000000E+00;
    ZROT[45] = 1.00000000E+00;
    ZROT[46] = 1.00000000E+00;
    ZROT[130] = 1.00000000E+00;
    ZROT[47] = 1.00000000E+00;
    ZROT[48] = 1.00000000E+00;
    ZROT[109] = 1.00000000E+00;
    ZROT[49] = 1.00000000E+00;
    ZROT[50] = 1.00000000E+00;
    ZROT[152] = 1.00000000E+00;
    ZROT[51] = 1.00000000E+00;
    ZROT[52] = 1.00000000E+00;
    ZROT[131] = 1.00000000E+00;
    ZROT[53] = 1.00000000E+00;
    ZROT[54] = 1.00000000E+00;
    ZROT[110] = 1.00000000E+00;
    ZROT[55] = 1.00000000E+00;
    ZROT[56] = 1.00000000E+00;
    ZROT[153] = 1.00000000E+00;
    ZROT[57] = 1.00000000E+00;
    ZROT[58] = 1.00000000E+00;
    ZROT[132] = 0.00000000E+00;
    ZROT[59] = 1.00000000E+00;
    ZROT[60] = 1.00000000E+00;
    ZROT[111] = 1.00000000E+00;
    ZROT[61] = 1.00000000E+00;
    ZROT[62] = 0.00000000E+00;
    ZROT[154] = 1.00000000E+00;
    ZROT[63] = 1.00000000E+00;
    ZROT[64] = 1.00000000E+00;
    ZROT[133] = 1.00000000E+00;
    ZROT[65] = 1.00000000E+00;
    ZROT[66] = 1.00000000E+00;
    ZROT[112] = 1.00000000E+00;
    ZROT[67] = 1.00000000E+00;
    ZROT[68] = 1.00000000E+00;
    ZROT[155] = 1.00000000E+00;
    ZROT[69] = 1.00000000E+00;
    ZROT[70] = 1.00000000E+00;
    ZROT[134] = 1.00000000E+00;
    ZROT[71] = 1.00000000E+00;
    ZROT[72] = 1.00000000E+00;
    ZROT[113] = 1.00000000E+00;
    ZROT[73] = 1.00000000E+00;
    ZROT[74] = 0.00000000E+00;
    ZROT[156] = 1.00000000E+00;
    ZROT[75] = 1.00000000E+00;
    ZROT[76] = 1.00000000E+00;
    ZROT[135] = 1.00000000E+00;
    ZROT[77] = 1.00000000E+00;
    ZROT[78] = 1.00000000E+00;
    ZROT[114] = 1.00000000E+00;
    ZROT[79] = 1.00000000E+00;
    ZROT[80] = 1.00000000E+00;
    ZROT[81] = 1.00000000E+00;
    ZROT[82] = 1.00000000E+00;
    ZROT[136] = 1.00000000E+00;
    ZROT[83] = 1.00000000E+00;
    ZROT[84] = 1.00000000E+00;
    ZROT[115] = 0.00000000E+00;
    ZROT[85] = 1.00000000E+00;
    ZROT[86] = 1.00000000E+00;
    ZROT[87] = 1.00000000E+00;
    ZROT[88] = 1.00000000E+00;
    ZROT[137] = 1.00000000E+00;
    ZROT[89] = 1.00000000E+00;
    ZROT[90] = 1.00000000E+00;
    ZROT[116] = 0.00000000E+00;
    ZROT[91] = 1.00000000E+00;
    ZROT[92] = 1.00000000E+00;
    ZROT[93] = 1.00000000E+00;
    ZROT[94] = 1.00000000E+00;
    ZROT[138] = 0.00000000E+00;
    ZROT[95] = 1.00000000E+00;
    ZROT[96] = 1.00000000E+00;
    ZROT[117] = 1.00000000E+00;
}


/*0: monoatomic, 1: linear, 2: nonlinear */
void egtransetNLIN(int* NLIN) {
    NLIN[139] = 2;
    NLIN[118] = 2;
    NLIN[97] = 2;
    NLIN[140] = 2;
    NLIN[119] = 2;
    NLIN[98] = 2;
    NLIN[141] = 2;
    NLIN[120] = 2;
    NLIN[99] = 2;
    NLIN[142] = 2;
    NLIN[121] = 2;
    NLIN[100] = 2;
    NLIN[157] = 1;
    NLIN[0] = 0;
    NLIN[143] = 2;
    NLIN[122] = 2;
    NLIN[1] = 1;
    NLIN[101] = 2;
    NLIN[2] = 0;
    NLIN[3] = 1;
    NLIN[144] = 2;
    NLIN[4] = 1;
    NLIN[5] = 2;
    NLIN[123] = 2;
    NLIN[6] = 1;
    NLIN[7] = 2;
    NLIN[102] = 2;
    NLIN[8] = 2;
    NLIN[9] = 1;
    NLIN[145] = 2;
    NLIN[10] = 2;
    NLIN[11] = 0;
    NLIN[124] = 2;
    NLIN[12] = 1;
    NLIN[13] = 1;
    NLIN[103] = 2;
    NLIN[14] = 1;
    NLIN[15] = 2;
    NLIN[146] = 2;
    NLIN[16] = 2;
    NLIN[17] = 1;
    NLIN[125] = 2;
    NLIN[18] = 2;
    NLIN[19] = 1;
    NLIN[104] = 2;
    NLIN[20] = 1;
    NLIN[21] = 0;
    NLIN[147] = 2;
    NLIN[22] = 2;
    NLIN[23] = 2;
    NLIN[126] = 2;
    NLIN[24] = 2;
    NLIN[25] = 2;
    NLIN[105] = 2;
    NLIN[26] = 2;
    NLIN[27] = 2;
    NLIN[148] = 2;
    NLIN[28] = 2;
    NLIN[29] = 2;
    NLIN[127] = 2;
    NLIN[30] = 2;
    NLIN[31] = 2;
    NLIN[106] = 2;
    NLIN[32] = 2;
    NLIN[33] = 2;
    NLIN[149] = 2;
    NLIN[34] = 2;
    NLIN[128] = 2;
    NLIN[35] = 2;
    NLIN[36] = 2;
    NLIN[107] = 2;
    NLIN[37] = 2;
    NLIN[38] = 2;
    NLIN[150] = 2;
    NLIN[39] = 2;
    NLIN[40] = 2;
    NLIN[129] = 2;
    NLIN[41] = 2;
    NLIN[42] = 2;
    NLIN[108] = 2;
    NLIN[43] = 2;
    NLIN[44] = 2;
    NLIN[151] = 2;
    NLIN[45] = 2;
    NLIN[46] = 1;
    NLIN[130] = 2;
    NLIN[47] = 2;
    NLIN[48] = 2;
    NLIN[109] = 2;
    NLIN[49] = 2;
    NLIN[50] = 1;
    NLIN[152] = 2;
    NLIN[51] = 2;
    NLIN[52] = 2;
    NLIN[131] = 2;
    NLIN[53] = 2;
    NLIN[54] = 2;
    NLIN[110] = 2;
    NLIN[55] = 1;
    NLIN[56] = 1;
    NLIN[153] = 2;
    NLIN[57] = 2;
    NLIN[58] = 2;
    NLIN[132] = 2;
    NLIN[59] = 2;
    NLIN[60] = 2;
    NLIN[111] = 2;
    NLIN[61] = 2;
    NLIN[62] = 2;
    NLIN[154] = 2;
    NLIN[63] = 2;
    NLIN[64] = 2;
    NLIN[133] = 2;
    NLIN[65] = 2;
    NLIN[66] = 2;
    NLIN[112] = 2;
    NLIN[67] = 2;
    NLIN[68] = 2;
    NLIN[155] = 2;
    NLIN[69] = 2;
    NLIN[70] = 2;
    NLIN[134] = 2;
    NLIN[71] = 2;
    NLIN[72] = 2;
    NLIN[113] = 2;
    NLIN[73] = 2;
    NLIN[74] = 2;
    NLIN[156] = 2;
    NLIN[75] = 2;
    NLIN[76] = 2;
    NLIN[135] = 2;
    NLIN[77] = 2;
    NLIN[78] = 2;
    NLIN[114] = 2;
    NLIN[79] = 2;
    NLIN[80] = 2;
    NLIN[81] = 2;
    NLIN[82] = 2;
    NLIN[136] = 2;
    NLIN[83] = 2;
    NLIN[84] = 2;
    NLIN[115] = 2;
    NLIN[85] = 2;
    NLIN[86] = 2;
    NLIN[87] = 2;
    NLIN[88] = 2;
    NLIN[137] = 2;
    NLIN[89] = 2;
    NLIN[90] = 2;
    NLIN[116] = 2;
    NLIN[91] = 2;
    NLIN[92] = 2;
    NLIN[93] = 2;
    NLIN[94] = 2;
    NLIN[138] = 2;
    NLIN[95] = 2;
    NLIN[96] = 2;
    NLIN[117] = 2;
}



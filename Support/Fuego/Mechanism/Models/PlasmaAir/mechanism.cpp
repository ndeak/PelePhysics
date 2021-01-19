#include "chemistry_file.H"

#ifndef AMREX_USE_CUDA
namespace thermo
{
    double fwd_A[22], fwd_beta[22], fwd_Ea[22];
    double low_A[22], low_beta[22], low_Ea[22];
    double rev_A[22], rev_beta[22], rev_Ea[22];
    double troe_a[22],troe_Ts[22], troe_Tss[22], troe_Tsss[22];
    double sri_a[22], sri_b[22], sri_c[22], sri_d[22], sri_e[22];
    double activation_units[22], prefactor_units[22], phase_units[22];
    int is_PD[22], troe_len[22], sri_len[22], nTB[22], *TBid[22];
    double *TB[22];
    std::vector<std::vector<double>> kiv(22); 
    std::vector<std::vector<double>> nuv(22); 

    double fwd_A_DEF[22], fwd_beta_DEF[22], fwd_Ea_DEF[22];
    double low_A_DEF[22], low_beta_DEF[22], low_Ea_DEF[22];
    double rev_A_DEF[22], rev_beta_DEF[22], rev_Ea_DEF[22];
    double troe_a_DEF[22],troe_Ts_DEF[22], troe_Tss_DEF[22], troe_Tsss_DEF[22];
    double sri_a_DEF[22], sri_b_DEF[22], sri_c_DEF[22], sri_d_DEF[22], sri_e_DEF[22];
    double activation_units_DEF[22], prefactor_units_DEF[22], phase_units_DEF[22];
    int is_PD_DEF[22], troe_len_DEF[22], sri_len_DEF[22], nTB_DEF[22], *TBid_DEF[22];
    double *TB_DEF[22];
    double *TeData;
    double *ENData;
    int Te_len;
    std::vector<int> rxn_map;
};

using namespace thermo;
#endif

/* Inverse molecular weights */
/* TODO: check necessity on CPU */
static AMREX_GPU_DEVICE_MANAGED double imw[10] = {
    1.0 / 0.000549,  /*E */
    1.0 / 31.998800,  /*O2 */
    1.0 / 28.013400,  /*N2 */
    1.0 / 15.999400,  /*O */
    1.0 / 31.998251,  /*O2+ */
    1.0 / 28.012851,  /*N2+ */
    1.0 / 63.997051,  /*O4+ */
    1.0 / 56.026251,  /*N4+ */
    1.0 / 63.997051,  /*O2pN2 */
    1.0 / 31.999349};  /*O2- */

/* Inverse molecular weights */
/* TODO: check necessity because redundant with molecularWeight */
static AMREX_GPU_DEVICE_MANAGED double molecular_weights[10] = {
    0.000549,  /*E */
    31.998800,  /*O2 */
    28.013400,  /*N2 */
    15.999400,  /*O */
    31.998251,  /*O2+ */
    28.012851,  /*N2+ */
    63.997051,  /*O4+ */
    56.026251,  /*N4+ */
    63.997051,  /*O2pN2 */
    31.999349};  /*O2- */

AMREX_GPU_HOST_DEVICE
void get_imw(double imw_new[]){
    for(int i = 0; i<10; ++i) imw_new[i] = imw[i];
}

/* TODO: check necessity because redundant with CKWT */
AMREX_GPU_HOST_DEVICE
void get_mw(double mw_new[]){
    for(int i = 0; i<10; ++i) mw_new[i] = molecular_weights[i];
}


#ifndef AMREX_USE_CUDA
/* Initializes parameter database */
void CKINIT()
{

    rxn_map = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21};

    // (0):  E + N2 => E + E + N2+
    kiv[0] = {0,2,0,0,5};
    nuv[0] = {-1,-1,1,1,1};
    // (0):  E + N2 => E + E + N2+
    fwd_A[0]     = 1;
    fwd_beta[0]  = 0;
    fwd_Ea[0]    = 0;
    prefactor_units[0]  = 1.0000000000000002e-06;
    activation_units[0] = 0.50321666580471969;
    phase_units[0]      = pow(10,-12.000000);
    is_PD[0] = 0;
    nTB[0] = 0;

    // (1):  E + O2 => E + E + O2+
    kiv[1] = {0,1,0,0,4};
    nuv[1] = {-1,-1,1,1,1};
    // (1):  E + O2 => E + E + O2+
    fwd_A[1]     = 1;
    fwd_beta[1]  = 0;
    fwd_Ea[1]    = 0;
    prefactor_units[1]  = 1.0000000000000002e-06;
    activation_units[1] = 0.50321666580471969;
    phase_units[1]      = pow(10,-12.000000);
    is_PD[1] = 0;
    nTB[1] = 0;

    // (2):  N2+ + N2 + N2 => N4+ + N2
    kiv[2] = {5,2,2,7,2};
    nuv[2] = {-1,-1,-1,1,1};
    // (2):  N2+ + N2 + N2 => N4+ + N2
    fwd_A[2]     = 1.8132242e+19;
    fwd_beta[2]  = 0;
    fwd_Ea[2]    = 0;
    prefactor_units[2]  = 1.0000000000000002e-12;
    activation_units[2] = 0.50321666580471969;
    phase_units[2]      = pow(10,-18.000000);
    is_PD[2] = 0;
    nTB[2] = 0;

    // (3):  N2+ + N2 + O2 => N4+ + O2
    kiv[3] = {5,2,1,7,1};
    nuv[3] = {-1,-1,-1,1,1};
    // (3):  N2+ + N2 + O2 => N4+ + O2
    fwd_A[3]     = 1.8132242e+19;
    fwd_beta[3]  = 0;
    fwd_Ea[3]    = 0;
    prefactor_units[3]  = 1.0000000000000002e-12;
    activation_units[3] = 0.50321666580471969;
    phase_units[3]      = pow(10,-18.000000);
    is_PD[3] = 0;
    nTB[3] = 0;

    // (4):  N4+ + O2 => O2+ + N2 + N2
    kiv[4] = {7,1,4,2,2};
    nuv[4] = {-1,-1,1,1,1};
    // (4):  N4+ + O2 => O2+ + N2 + N2
    fwd_A[4]     = 150550000000000;
    fwd_beta[4]  = 0;
    fwd_Ea[4]    = 0;
    prefactor_units[4]  = 1.0000000000000002e-06;
    activation_units[4] = 0.50321666580471969;
    phase_units[4]      = pow(10,-12.000000);
    is_PD[4] = 0;
    nTB[4] = 0;

    // (5):  N2+ + O2 => O2+ + N2
    kiv[5] = {5,1,4,2};
    nuv[5] = {-1,-1,1,1};
    // (5):  N2+ + O2 => O2+ + N2
    fwd_A[5]     = 36132000000000;
    fwd_beta[5]  = 0;
    fwd_Ea[5]    = 0;
    prefactor_units[5]  = 1.0000000000000002e-06;
    activation_units[5] = 0.50321666580471969;
    phase_units[5]      = pow(10,-12.000000);
    is_PD[5] = 0;
    nTB[5] = 0;

    // (6):  O2+ + N2 + N2 => O2pN2 + N2
    kiv[6] = {4,2,2,8,2};
    nuv[6] = {-1,-1,-1,1,1};
    // (6):  O2+ + N2 + N2 => O2pN2 + N2
    fwd_A[6]     = 3.26380356e+17;
    fwd_beta[6]  = 0;
    fwd_Ea[6]    = 0;
    prefactor_units[6]  = 1.0000000000000002e-12;
    activation_units[6] = 0.50321666580471969;
    phase_units[6]      = pow(10,-18.000000);
    is_PD[6] = 0;
    nTB[6] = 0;

    // (7):  O2pN2 + N2 => O2+ + N2 + N2
    kiv[7] = {8,2,4,2,2};
    nuv[7] = {-1,-1,1,1,1};
    // (7):  O2pN2 + N2 => O2+ + N2 + N2
    fwd_A[7]     = 258946000000000;
    fwd_beta[7]  = 0;
    fwd_Ea[7]    = 0;
    prefactor_units[7]  = 1.0000000000000002e-06;
    activation_units[7] = 0.50321666580471969;
    phase_units[7]      = pow(10,-12.000000);
    is_PD[7] = 0;
    nTB[7] = 0;

    // (8):  O2pN2 + O2 => O4+ + N2
    kiv[8] = {8,1,6,2};
    nuv[8] = {-1,-1,1,1};
    // (8):  O2pN2 + O2 => O4+ + N2
    fwd_A[8]     = 602200000000000;
    fwd_beta[8]  = 0;
    fwd_Ea[8]    = 0;
    prefactor_units[8]  = 1.0000000000000002e-06;
    activation_units[8] = 0.50321666580471969;
    phase_units[8]      = pow(10,-12.000000);
    is_PD[8] = 0;
    nTB[8] = 0;

    // (9):  O2+ + O2 + N2 => O4+ + N2
    kiv[9] = {4,1,2,6,2};
    nuv[9] = {-1,-1,-1,1,1};
    // (9):  O2+ + O2 + N2 => O4+ + N2
    fwd_A[9]     = 8.70347616e+17;
    fwd_beta[9]  = 0;
    fwd_Ea[9]    = 0;
    prefactor_units[9]  = 1.0000000000000002e-12;
    activation_units[9] = 0.50321666580471969;
    phase_units[9]      = pow(10,-18.000000);
    is_PD[9] = 0;
    nTB[9] = 0;

    // (10):  O2+ + O2 + O2 => O4+ + O2
    kiv[10] = {4,1,1,6,1};
    nuv[10] = {-1,-1,-1,1,1};
    // (10):  O2+ + O2 + O2 => O4+ + O2
    fwd_A[10]     = 8.70347616e+17;
    fwd_beta[10]  = 0;
    fwd_Ea[10]    = 0;
    prefactor_units[10]  = 1.0000000000000002e-12;
    activation_units[10] = 0.50321666580471969;
    phase_units[10]      = pow(10,-18.000000);
    is_PD[10] = 0;
    nTB[10] = 0;

    // (11):  E + O4+ => O2 + O2
    kiv[11] = {0,6,1,1};
    nuv[11] = {-1,-1,1,1};
    // (11):  E + O4+ => O2 + O2
    fwd_A[11]     = 1;
    fwd_beta[11]  = 0;
    fwd_Ea[11]    = 0;
    prefactor_units[11]  = 1.0000000000000002e-06;
    activation_units[11] = 0.50321666580471969;
    phase_units[11]      = pow(10,-12.000000);
    is_PD[11] = 0;
    nTB[11] = 0;

    // (12):  E + O2+ => O + O
    kiv[12] = {0,4,3,3};
    nuv[12] = {-1,-1,1,1};
    // (12):  E + O2+ => O + O
    fwd_A[12]     = 1;
    fwd_beta[12]  = 0;
    fwd_Ea[12]    = 0;
    prefactor_units[12]  = 1.0000000000000002e-06;
    activation_units[12] = 0.50321666580471969;
    phase_units[12]      = pow(10,-12.000000);
    is_PD[12] = 0;
    nTB[12] = 0;

    // (13):  E + O2 + O2 => O2- + O2
    kiv[13] = {0,1,1,9,1};
    nuv[13] = {-1,-1,-1,1,1};
    // (13):  E + O2 + O2 => O2- + O2
    fwd_A[13]     = 1;
    fwd_beta[13]  = 0;
    fwd_Ea[13]    = 0;
    prefactor_units[13]  = 1.0000000000000002e-12;
    activation_units[13] = 0.50321666580471969;
    phase_units[13]      = pow(10,-18.000000);
    is_PD[13] = 0;
    nTB[13] = 0;

    // (14):  E + O2 + N2 => O2- + N2
    kiv[14] = {0,1,2,9,2};
    nuv[14] = {-1,-1,-1,1,1};
    // (14):  E + O2 + N2 => O2- + N2
    fwd_A[14]     = 1;
    fwd_beta[14]  = 0;
    fwd_Ea[14]    = 0;
    prefactor_units[14]  = 1.0000000000000002e-12;
    activation_units[14] = 0.50321666580471969;
    phase_units[14]      = pow(10,-18.000000);
    is_PD[14] = 0;
    nTB[14] = 0;

    // (15):  O2- + O4+ => O2 + O2 + O2
    kiv[15] = {9,6,1,1,1};
    nuv[15] = {-1,-1,1,1,1};
    // (15):  O2- + O4+ => O2 + O2 + O2
    fwd_A[15]     = 60220000000000000;
    fwd_beta[15]  = 0;
    fwd_Ea[15]    = 0;
    prefactor_units[15]  = 1.0000000000000002e-06;
    activation_units[15] = 0.50321666580471969;
    phase_units[15]      = pow(10,-12.000000);
    is_PD[15] = 0;
    nTB[15] = 0;

    // (16):  O2- + O4+ + O2 => O2 + O2 + O2 + O2
    kiv[16] = {9,6,1,1,1,1,1};
    nuv[16] = {-1,-1,-1,1,1,1,1};
    // (16):  O2- + O4+ + O2 => O2 + O2 + O2 + O2
    fwd_A[16]     = 7.2528968000000003e+22;
    fwd_beta[16]  = 0;
    fwd_Ea[16]    = 0;
    prefactor_units[16]  = 1.0000000000000002e-12;
    activation_units[16] = 0.50321666580471969;
    phase_units[16]      = pow(10,-18.000000);
    is_PD[16] = 0;
    nTB[16] = 0;

    // (17):  O2- + O4+ + N2 => O2 + O2 + O2 + N2
    kiv[17] = {9,6,2,1,1,1,2};
    nuv[17] = {-1,-1,-1,1,1,1,1};
    // (17):  O2- + O4+ + N2 => O2 + O2 + O2 + N2
    fwd_A[17]     = 7.2528968000000003e+22;
    fwd_beta[17]  = 0;
    fwd_Ea[17]    = 0;
    prefactor_units[17]  = 1.0000000000000002e-12;
    activation_units[17] = 0.50321666580471969;
    phase_units[17]      = pow(10,-18.000000);
    is_PD[17] = 0;
    nTB[17] = 0;

    // (18):  O2- + O2+ + O2 => O2 + O2 + O2
    kiv[18] = {9,4,1,1,1,1};
    nuv[18] = {-1,-1,-1,1,1,1};
    // (18):  O2- + O2+ + O2 => O2 + O2 + O2
    fwd_A[18]     = 7.2528968000000003e+22;
    fwd_beta[18]  = 0;
    fwd_Ea[18]    = 0;
    prefactor_units[18]  = 1.0000000000000002e-12;
    activation_units[18] = 0.50321666580471969;
    phase_units[18]      = pow(10,-18.000000);
    is_PD[18] = 0;
    nTB[18] = 0;

    // (19):  O2- + O2+ + N2 => O2 + O2 + N2
    kiv[19] = {9,4,2,1,1,2};
    nuv[19] = {-1,-1,-1,1,1,1};
    // (19):  O2- + O2+ + N2 => O2 + O2 + N2
    fwd_A[19]     = 7.2528968000000003e+22;
    fwd_beta[19]  = 0;
    fwd_Ea[19]    = 0;
    prefactor_units[19]  = 1.0000000000000002e-12;
    activation_units[19] = 0.50321666580471969;
    phase_units[19]      = pow(10,-18.000000);
    is_PD[19] = 0;
    nTB[19] = 0;

    // (20):  O2- + O2 => E + O2 + O2
    kiv[20] = {9,1,0,1,1};
    nuv[20] = {-1,-1,1,1,1};
    // (20):  O2- + O2 => E + O2 + O2
    fwd_A[20]     = 1;
    fwd_beta[20]  = 0;
    fwd_Ea[20]    = 0;
    prefactor_units[20]  = 1.0000000000000002e-06;
    activation_units[20] = 0.50321666580471969;
    phase_units[20]      = pow(10,-12.000000);
    is_PD[20] = 0;
    nTB[20] = 0;

    // (21):  O2- + N2 => E + O2 + N2
    kiv[21] = {9,2,0,1,2};
    nuv[21] = {-1,-1,1,1,1};
    // (21):  O2- + N2 => E + O2 + N2
    fwd_A[21]     = 1;
    fwd_beta[21]  = 0;
    fwd_Ea[21]    = 0;
    prefactor_units[21]  = 1.0000000000000002e-06;
    activation_units[21] = 0.50321666580471969;
    phase_units[21]      = pow(10,-12.000000);
    is_PD[21] = 0;
    nTB[21] = 0;

    SetAllDefaults();

    /*Load in Te(E/N) data */

    // Assume constant name for datafile for now...
    std::string Te_extrap = "Te.dat";
    std::ifstream Tefile(Te_extrap.c_str());
    if(!Tefile.good()) {
      printf("INPUT ERROR : unable to open Te_extrap file!\n");
      exit(1);
    }

    // Get the number of lines in the file
    Te_len = std::count(std::istreambuf_iterator<char>(Tefile), std::istreambuf_iterator<char>(), '\n') - 1;  // Assumes 1 line for header file

    // Quick checks on file length
    if(Te_len <= 3){
      printf("INPUT ERROR : Te_extrap file must have at least 3 entries!\n");
      exit(1);
    }

    // Create data arrays
    TeData = new double[Te_len]{0.0};
    ENData = new double[Te_len]{0.0};

    // Load data into arrays, line by line
    std::ifstream Loadfile(Te_extrap.c_str());
    std::string line;
    std::getline(Loadfile, line);   // Pull off header file
    for(int i=0; i<Te_len; i++){
      std::getline(Loadfile, line);   // Get data row
      std::istringstream iss(line);
      iss >> ENData[i] >> TeData[i];
    }
    // Check for successive duplicate values
    for(int i=1; i<Te_len; i++){
      if(ENData[i] == ENData[i-1]){
        printf("Duplicate EN entries found in Te file. Exiting!\n");
        exit(1);
      }
    }

}

void GET_REACTION_MAP(int *rmap)
{
    for (int i=0; i<22; ++i) {
        rmap[i] = rxn_map[i] + 1;
    }
}

#include <ReactionData.H>
double* GetParamPtr(int                reaction_id,
                    REACTION_PARAMETER param_id,
                    int                species_id,
                    int                get_default)
{
  double* ret = 0;
  if (reaction_id<0 || reaction_id>=22) {
    printf("Bad reaction id = %d",reaction_id);
    abort();
  };
  int mrid = rxn_map[reaction_id];

  if (param_id == THIRD_BODY) {
    if (species_id<0 || species_id>=10) {
      printf("GetParamPtr: Bad species id = %d",species_id);
      abort();
    }
    if (get_default) {
      for (int i=0; i<nTB_DEF[mrid]; ++i) {
        if (species_id == TBid_DEF[mrid][i]) {
          ret = &(TB_DEF[mrid][i]);
        }
      }
    }
    else {
      for (int i=0; i<nTB[mrid]; ++i) {
        if (species_id == TBid[mrid][i]) {
          ret = &(TB[mrid][i]);
        }
      }
    }
    if (ret == 0) {
      printf("GetParamPtr: No TB for reaction id = %d",reaction_id);
      abort();
    }
  }
  else {
    if (     param_id == FWD_A)     {ret = (get_default ? &(fwd_A_DEF[mrid]) : &(fwd_A[mrid]));}
      else if (param_id == FWD_BETA)  {ret = (get_default ? &(fwd_beta_DEF[mrid]) : &(fwd_beta[mrid]));}
      else if (param_id == FWD_EA)    {ret = (get_default ? &(fwd_Ea_DEF[mrid]) : &(fwd_Ea[mrid]));}
      else if (param_id == LOW_A)     {ret = (get_default ? &(low_A_DEF[mrid]) : &(low_A[mrid]));}
      else if (param_id == LOW_BETA)  {ret = (get_default ? &(low_beta_DEF[mrid]) : &(low_beta[mrid]));}
      else if (param_id == LOW_EA)    {ret = (get_default ? &(low_Ea_DEF[mrid]) : &(low_Ea[mrid]));}
      else if (param_id == REV_A)     {ret = (get_default ? &(rev_A_DEF[mrid]) : &(rev_A[mrid]));}
      else if (param_id == REV_BETA)  {ret = (get_default ? &(rev_beta_DEF[mrid]) : &(rev_beta[mrid]));}
      else if (param_id == REV_EA)    {ret = (get_default ? &(rev_Ea_DEF[mrid]) : &(rev_Ea[mrid]));}
      else if (param_id == TROE_A)    {ret = (get_default ? &(troe_a_DEF[mrid]) : &(troe_a[mrid]));}
      else if (param_id == TROE_TS)   {ret = (get_default ? &(troe_Ts_DEF[mrid]) : &(troe_Ts[mrid]));}
      else if (param_id == TROE_TSS)  {ret = (get_default ? &(troe_Tss_DEF[mrid]) : &(troe_Tss[mrid]));}
      else if (param_id == TROE_TSSS) {ret = (get_default ? &(troe_Tsss_DEF[mrid]) : &(troe_Tsss[mrid]));}
      else if (param_id == SRI_A)     {ret = (get_default ? &(sri_a_DEF[mrid]) : &(sri_a[mrid]));}
      else if (param_id == SRI_B)     {ret = (get_default ? &(sri_b_DEF[mrid]) : &(sri_b[mrid]));}
      else if (param_id == SRI_C)     {ret = (get_default ? &(sri_c_DEF[mrid]) : &(sri_c[mrid]));}
      else if (param_id == SRI_D)     {ret = (get_default ? &(sri_d_DEF[mrid]) : &(sri_d[mrid]));}
      else if (param_id == SRI_E)     {ret = (get_default ? &(sri_e_DEF[mrid]) : &(sri_e[mrid]));}
    else {
      printf("GetParamPtr: Unknown parameter id");
      abort();
    }
  }
  return ret;
}

void ResetAllParametersToDefault()
{
    for (int i=0; i<22; i++) {
        if (nTB[i] != 0) {
            nTB[i] = 0;
            free(TB[i]);
            free(TBid[i]);
        }

        fwd_A[i]    = fwd_A_DEF[i];
        fwd_beta[i] = fwd_beta_DEF[i];
        fwd_Ea[i]   = fwd_Ea_DEF[i];

        low_A[i]    = low_A_DEF[i];
        low_beta[i] = low_beta_DEF[i];
        low_Ea[i]   = low_Ea_DEF[i];

        rev_A[i]    = rev_A_DEF[i];
        rev_beta[i] = rev_beta_DEF[i];
        rev_Ea[i]   = rev_Ea_DEF[i];

        troe_a[i]    = troe_a_DEF[i];
        troe_Ts[i]   = troe_Ts_DEF[i];
        troe_Tss[i]  = troe_Tss_DEF[i];
        troe_Tsss[i] = troe_Tsss_DEF[i];

        sri_a[i] = sri_a_DEF[i];
        sri_b[i] = sri_b_DEF[i];
        sri_c[i] = sri_c_DEF[i];
        sri_d[i] = sri_d_DEF[i];
        sri_e[i] = sri_e_DEF[i];

        is_PD[i]    = is_PD_DEF[i];
        troe_len[i] = troe_len_DEF[i];
        sri_len[i]  = sri_len_DEF[i];

        activation_units[i] = activation_units_DEF[i];
        prefactor_units[i]  = prefactor_units_DEF[i];
        phase_units[i]      = phase_units_DEF[i];

        nTB[i]  = nTB_DEF[i];
        if (nTB[i] != 0) {
           TB[i] = (double *) malloc(sizeof(double) * nTB[i]);
           TBid[i] = (int *) malloc(sizeof(int) * nTB[i]);
           for (int j=0; j<nTB[i]; j++) {
             TB[i][j] = TB_DEF[i][j];
             TBid[i][j] = TBid_DEF[i][j];
           }
        }
    }
}

void SetAllDefaults()
{
    for (int i=0; i<22; i++) {
        if (nTB_DEF[i] != 0) {
            nTB_DEF[i] = 0;
            free(TB_DEF[i]);
            free(TBid_DEF[i]);
        }

        fwd_A_DEF[i]    = fwd_A[i];
        fwd_beta_DEF[i] = fwd_beta[i];
        fwd_Ea_DEF[i]   = fwd_Ea[i];

        low_A_DEF[i]    = low_A[i];
        low_beta_DEF[i] = low_beta[i];
        low_Ea_DEF[i]   = low_Ea[i];

        rev_A_DEF[i]    = rev_A[i];
        rev_beta_DEF[i] = rev_beta[i];
        rev_Ea_DEF[i]   = rev_Ea[i];

        troe_a_DEF[i]    = troe_a[i];
        troe_Ts_DEF[i]   = troe_Ts[i];
        troe_Tss_DEF[i]  = troe_Tss[i];
        troe_Tsss_DEF[i] = troe_Tsss[i];

        sri_a_DEF[i] = sri_a[i];
        sri_b_DEF[i] = sri_b[i];
        sri_c_DEF[i] = sri_c[i];
        sri_d_DEF[i] = sri_d[i];
        sri_e_DEF[i] = sri_e[i];

        is_PD_DEF[i]    = is_PD[i];
        troe_len_DEF[i] = troe_len[i];
        sri_len_DEF[i]  = sri_len[i];

        activation_units_DEF[i] = activation_units[i];
        prefactor_units_DEF[i]  = prefactor_units[i];
        phase_units_DEF[i]      = phase_units[i];

        nTB_DEF[i]  = nTB[i];
        if (nTB_DEF[i] != 0) {
           TB_DEF[i] = (double *) malloc(sizeof(double) * nTB_DEF[i]);
           TBid_DEF[i] = (int *) malloc(sizeof(int) * nTB_DEF[i]);
           for (int j=0; j<nTB_DEF[i]; j++) {
             TB_DEF[i][j] = TB[i][j];
             TBid_DEF[i][j] = TBid[i][j];
           }
        }
    }
}

/* Finalizes parameter database */
void CKFINALIZE()
{
  for (int i=0; i<22; ++i) {
    free(TB[i]); TB[i] = 0; 
    free(TBid[i]); TBid[i] = 0;
    nTB[i] = 0;

    free(TB_DEF[i]); TB_DEF[i] = 0; 
    free(TBid_DEF[i]); TBid_DEF[i] = 0;
    nTB_DEF[i] = 0;
  }
}

#else
/* TODO: Remove on GPU, right now needed by chemistry_module on FORTRAN */
AMREX_GPU_HOST_DEVICE void CKINIT()
{
}

AMREX_GPU_HOST_DEVICE void CKFINALIZE()
{
}

#endif


/*A few mechanism parameters */
void CKINDX(int * mm, int * kk, int * ii, int * nfit)
{
    *mm = 3;
    *kk = 10;
    *ii = 22;
    *nfit = -1; /*Why do you need this anyway ?  */
}



/* ckxnum... for parsing strings  */
void CKXNUM(char * line, int * nexp, int * lout, int * nval, double *  rval, int * kerr, int lenline )
{
    int n,i; /*Loop Counters */
    char cstr[1000];
    char *saveptr;
    char *p; /*String Tokens */
    /* Strip Comments  */
    for (i=0; i<lenline; ++i) {
        if (line[i]=='!') {
            break;
        }
        cstr[i] = line[i];
    }
    cstr[i] = '\0';

    p = strtok_r(cstr," ", &saveptr);
    if (!p) {
        *nval = 0;
        *kerr = 1;
        return;
    }
    for (n=0; n<*nexp; ++n) {
        rval[n] = atof(p);
        p = strtok_r(NULL, " ", &saveptr);
        if (!p) break;
    }
    *nval = n+1;
    if (*nval < *nexp) *kerr = 1;
    return;
}


/* cksnum... for parsing strings  */
void CKSNUM(char * line, int * nexp, int * lout, char * kray, int * nn, int * knum, int * nval, double *  rval, int * kerr, int lenline, int lenkray)
{
    /*Not done yet ... */
}


/* Returns the vector of strings of element names */
void CKSYME_STR(amrex::Vector<std::string>& ename)
{
    ename.resize(3);
    ename[0] = "O";
    ename[1] = "N";
    ename[2] = "E";
}


/* Returns the char strings of element names */
void CKSYME(int * kname, int * plenkname )
{
    int i; /*Loop Counter */
    int lenkname = *plenkname;
    /*clear kname */
    for (i=0; i<lenkname*3; i++) {
        kname[i] = ' ';
    }

    /* O  */
    kname[ 0*lenkname + 0 ] = 'O';
    kname[ 0*lenkname + 1 ] = ' ';

    /* N  */
    kname[ 1*lenkname + 0 ] = 'N';
    kname[ 1*lenkname + 1 ] = ' ';

    /* E  */
    kname[ 2*lenkname + 0 ] = 'E';
    kname[ 2*lenkname + 1 ] = ' ';

}


/* Returns the vector of strings of species names */
void CKSYMS_STR(amrex::Vector<std::string>& kname)
{
    kname.resize(10);
    kname[0] = "E";
    kname[1] = "O2";
    kname[2] = "N2";
    kname[3] = "O";
    kname[4] = "O2+";
    kname[5] = "N2+";
    kname[6] = "O4+";
    kname[7] = "N4+";
    kname[8] = "O2pN2";
    kname[9] = "O2-";
}


/* Returns the char strings of species names */
void CKSYMS(int * kname, int * plenkname )
{
    int i; /*Loop Counter */
    int lenkname = *plenkname;
    /*clear kname */
    for (i=0; i<lenkname*10; i++) {
        kname[i] = ' ';
    }

    /* E  */
    kname[ 0*lenkname + 0 ] = 'E';
    kname[ 0*lenkname + 1 ] = ' ';

    /* O2  */
    kname[ 1*lenkname + 0 ] = 'O';
    kname[ 1*lenkname + 1 ] = '2';
    kname[ 1*lenkname + 2 ] = ' ';

    /* N2  */
    kname[ 2*lenkname + 0 ] = 'N';
    kname[ 2*lenkname + 1 ] = '2';
    kname[ 2*lenkname + 2 ] = ' ';

    /* O  */
    kname[ 3*lenkname + 0 ] = 'O';
    kname[ 3*lenkname + 1 ] = ' ';

    /* O2+  */
    kname[ 4*lenkname + 0 ] = 'O';
    kname[ 4*lenkname + 1 ] = '2';
    kname[ 4*lenkname + 2 ] = '+';
    kname[ 4*lenkname + 3 ] = ' ';

    /* N2+  */
    kname[ 5*lenkname + 0 ] = 'N';
    kname[ 5*lenkname + 1 ] = '2';
    kname[ 5*lenkname + 2 ] = '+';
    kname[ 5*lenkname + 3 ] = ' ';

    /* O4+  */
    kname[ 6*lenkname + 0 ] = 'O';
    kname[ 6*lenkname + 1 ] = '4';
    kname[ 6*lenkname + 2 ] = '+';
    kname[ 6*lenkname + 3 ] = ' ';

    /* N4+  */
    kname[ 7*lenkname + 0 ] = 'N';
    kname[ 7*lenkname + 1 ] = '4';
    kname[ 7*lenkname + 2 ] = '+';
    kname[ 7*lenkname + 3 ] = ' ';

    /* O2pN2  */
    kname[ 8*lenkname + 0 ] = 'O';
    kname[ 8*lenkname + 1 ] = '2';
    kname[ 8*lenkname + 2 ] = 'P';
    kname[ 8*lenkname + 3 ] = 'N';
    kname[ 8*lenkname + 4 ] = '2';
    kname[ 8*lenkname + 5 ] = ' ';

    /* O2-  */
    kname[ 9*lenkname + 0 ] = 'O';
    kname[ 9*lenkname + 1 ] = '2';
    kname[ 9*lenkname + 2 ] = '-';
    kname[ 9*lenkname + 3 ] = ' ';

}


/* Returns R, Rc, Patm */
void CKRP(double *  ru, double *  ruc, double *  pa)
{
     *ru  = 8.31446261815324e+07; 
     *ruc = 1.98721558317399615845; 
     *pa  = 1.01325e+06; 
}


/*Compute P = rhoRT/W(x) */
void CKPX(double *  rho, double *  T, double *  x, double *  P)
{
    double XW = 0;/* To hold mean molecular wt */
    XW += x[0]*0.000549; /*E */
    XW += x[1]*31.998800; /*O2 */
    XW += x[2]*28.013400; /*N2 */
    XW += x[3]*15.999400; /*O */
    XW += x[4]*31.998251; /*O2+ */
    XW += x[5]*28.012851; /*N2+ */
    XW += x[6]*63.997051; /*O4+ */
    XW += x[7]*56.026251; /*N4+ */
    XW += x[8]*63.997051; /*O2pN2 */
    XW += x[9]*31.999349; /*O2- */
    *P = *rho * 8.31446261815324e+07 * (*T) / XW; /*P = rho*R*T/W */

    return;
}


/*Compute P = rhoRT/W(y) */
AMREX_GPU_HOST_DEVICE void CKPY(double *  rho, double *  T, double *  y,  double *  P)
{
    double YOW = 0;/* for computing mean MW */
    YOW += y[0]*imw[0]; /*E */
    YOW += y[1]*imw[1]; /*O2 */
    YOW += y[2]*imw[2]; /*N2 */
    YOW += y[3]*imw[3]; /*O */
    YOW += y[4]*imw[4]; /*O2+ */
    YOW += y[5]*imw[5]; /*N2+ */
    YOW += y[6]*imw[6]; /*O4+ */
    YOW += y[7]*imw[7]; /*N4+ */
    YOW += y[8]*imw[8]; /*O2pN2 */
    YOW += y[9]*imw[9]; /*O2- */
    *P = *rho * 8.31446261815324e+07 * (*T) * YOW; /*P = rho*R*T/W */

    return;
}


#ifndef AMREX_USE_CUDA
/*Compute P = rhoRT/W(y) */
void VCKPY(int *  np, double *  rho, double *  T, double *  y,  double *  P)
{
    double YOW[*np];
    for (int i=0; i<(*np); i++) {
        YOW[i] = 0.0;
    }

    for (int n=0; n<10; n++) {
        for (int i=0; i<(*np); i++) {
            YOW[i] += y[n*(*np)+i] * imw[n];
        }
    }

    for (int i=0; i<(*np); i++) {
        P[i] = rho[i] * 8.31446261815324e+07 * T[i] * YOW[i]; /*P = rho*R*T/W */
    }

    return;
}
#endif


/*Compute P = rhoRT/W(c) */
void CKPC(double *  rho, double *  T, double *  c,  double *  P)
{
    int id; /*loop counter */
    /*See Eq 5 in CK Manual */
    double W = 0;
    double sumC = 0;
    W += c[0]*0.000549; /*E */
    W += c[1]*31.998800; /*O2 */
    W += c[2]*28.013400; /*N2 */
    W += c[3]*15.999400; /*O */
    W += c[4]*31.998251; /*O2+ */
    W += c[5]*28.012851; /*N2+ */
    W += c[6]*63.997051; /*O4+ */
    W += c[7]*56.026251; /*N4+ */
    W += c[8]*63.997051; /*O2pN2 */
    W += c[9]*31.999349; /*O2- */

    for (id = 0; id < 10; ++id) {
        sumC += c[id];
    }
    *P = *rho * 8.31446261815324e+07 * (*T) * sumC / W; /*P = rho*R*T/W */

    return;
}


/*Compute rho = PW(x)/RT */
void CKRHOX(double *  P, double *  T, double *  x,  double *  rho)
{
    double XW = 0;/* To hold mean molecular wt */
    XW += x[0]*0.000549; /*E */
    XW += x[1]*31.998800; /*O2 */
    XW += x[2]*28.013400; /*N2 */
    XW += x[3]*15.999400; /*O */
    XW += x[4]*31.998251; /*O2+ */
    XW += x[5]*28.012851; /*N2+ */
    XW += x[6]*63.997051; /*O4+ */
    XW += x[7]*56.026251; /*N4+ */
    XW += x[8]*63.997051; /*O2pN2 */
    XW += x[9]*31.999349; /*O2- */
    *rho = *P * XW / (8.31446261815324e+07 * (*T)); /*rho = P*W/(R*T) */

    return;
}


/*Compute rho = P*W(y)/RT */
AMREX_GPU_HOST_DEVICE void CKRHOY(double *  P, double *  T, double *  y,  double *  rho)
{
    double YOW = 0;
    double tmp[10];

    for (int i = 0; i < 10; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 10; i++)
    {
        YOW += tmp[i];
    }

    *rho = *P / (8.31446261815324e+07 * (*T) * YOW);/*rho = P*W/(R*T) */
    return;
}


/*Compute rho = P*W(c)/(R*T) */
void CKRHOC(double *  P, double *  T, double *  c,  double *  rho)
{
    int id; /*loop counter */
    /*See Eq 5 in CK Manual */
    double W = 0;
    double sumC = 0;
    W += c[0]*0.000549; /*E */
    W += c[1]*31.998800; /*O2 */
    W += c[2]*28.013400; /*N2 */
    W += c[3]*15.999400; /*O */
    W += c[4]*31.998251; /*O2+ */
    W += c[5]*28.012851; /*N2+ */
    W += c[6]*63.997051; /*O4+ */
    W += c[7]*56.026251; /*N4+ */
    W += c[8]*63.997051; /*O2pN2 */
    W += c[9]*31.999349; /*O2- */

    for (id = 0; id < 10; ++id) {
        sumC += c[id];
    }
    *rho = *P * W / (sumC * (*T) * 8.31446261815324e+07); /*rho = PW/(R*T) */

    return;
}


/*get molecular weight for all species */
void CKWT( double *  wt)
{
    get_mw(wt);
}


/*get atomic weight for all elements */
void CKAWT( double *  awt)
{
    atomicWeight(awt);
}


/*given y[species]: mass fractions */
/*returns mean molecular weight (gm/mole) */
AMREX_GPU_HOST_DEVICE void CKMMWY(double *  y,  double *  wtm)
{
    double YOW = 0;
    double tmp[10];

    for (int i = 0; i < 10; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 10; i++)
    {
        YOW += tmp[i];
    }

    *wtm = 1.0 / YOW;
    return;
}


/*given x[species]: mole fractions */
/*returns mean molecular weight (gm/mole) */
void CKMMWX(double *  x,  double *  wtm)
{
    double XW = 0;/* see Eq 4 in CK Manual */
    XW += x[0]*0.000549; /*E */
    XW += x[1]*31.998800; /*O2 */
    XW += x[2]*28.013400; /*N2 */
    XW += x[3]*15.999400; /*O */
    XW += x[4]*31.998251; /*O2+ */
    XW += x[5]*28.012851; /*N2+ */
    XW += x[6]*63.997051; /*O4+ */
    XW += x[7]*56.026251; /*N4+ */
    XW += x[8]*63.997051; /*O2pN2 */
    XW += x[9]*31.999349; /*O2- */
    *wtm = XW;

    return;
}


/*given c[species]: molar concentration */
/*returns mean molecular weight (gm/mole) */
void CKMMWC(double *  c,  double *  wtm)
{
    int id; /*loop counter */
    /*See Eq 5 in CK Manual */
    double W = 0;
    double sumC = 0;
    W += c[0]*0.000549; /*E */
    W += c[1]*31.998800; /*O2 */
    W += c[2]*28.013400; /*N2 */
    W += c[3]*15.999400; /*O */
    W += c[4]*31.998251; /*O2+ */
    W += c[5]*28.012851; /*N2+ */
    W += c[6]*63.997051; /*O4+ */
    W += c[7]*56.026251; /*N4+ */
    W += c[8]*63.997051; /*O2pN2 */
    W += c[9]*31.999349; /*O2- */

    for (id = 0; id < 10; ++id) {
        sumC += c[id];
    }
    /* CK provides no guard against divison by zero */
    *wtm = W/sumC;

    return;
}


/*convert y[species] (mass fracs) to x[species] (mole fracs) */
AMREX_GPU_HOST_DEVICE void CKYTX(double *  y,  double *  x)
{
    double YOW = 0;
    double tmp[10];

    for (int i = 0; i < 10; i++)
    {
        tmp[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 10; i++)
    {
        YOW += tmp[i];
    }

    double YOWINV = 1.0/YOW;

    for (int i = 0; i < 10; i++)
    {
        x[i] = y[i]*imw[i]*YOWINV;
    }
    return;
}


#ifndef AMREX_USE_CUDA
/*convert y[npoints*species] (mass fracs) to x[npoints*species] (mole fracs) */
void VCKYTX(int *  np, double *  y,  double *  x)
{
    double YOW[*np];
    for (int i=0; i<(*np); i++) {
        YOW[i] = 0.0;
    }

    for (int n=0; n<10; n++) {
        for (int i=0; i<(*np); i++) {
            x[n*(*np)+i] = y[n*(*np)+i] * imw[n];
            YOW[i] += x[n*(*np)+i];
        }
    }

    for (int i=0; i<(*np); i++) {
        YOW[i] = 1.0/YOW[i];
    }

    for (int n=0; n<10; n++) {
        for (int i=0; i<(*np); i++) {
            x[n*(*np)+i] *=  YOW[i];
        }
    }
}
#else
/*TODO: remove this on GPU */
void VCKYTX(int *  np, double *  y,  double *  x)
{
}
#endif


/*convert y[species] (mass fracs) to c[species] (molar conc) */
void CKYTCP(double *  P, double *  T, double *  y,  double *  c)
{
    double YOW = 0;
    double PWORT;

    /*Compute inverse of mean molecular wt first */
    for (int i = 0; i < 10; i++)
    {
        c[i] = y[i]*imw[i];
    }
    for (int i = 0; i < 10; i++)
    {
        YOW += c[i];
    }

    /*PW/RT (see Eq. 7) */
    PWORT = (*P)/(YOW * 8.31446261815324e+07 * (*T)); 
    /*Now compute conversion */

    for (int i = 0; i < 10; i++)
    {
        c[i] = PWORT * y[i] * imw[i];
    }
    return;
}


/*convert y[species] (mass fracs) to c[species] (molar conc) */
AMREX_GPU_HOST_DEVICE void CKYTCR(double *  rho, double *  T, double *  y,  double *  c)
{
    for (int i = 0; i < 10; i++)
    {
        c[i] = (*rho)  * y[i] * imw[i];
    }
}


/*convert x[species] (mole fracs) to y[species] (mass fracs) */
AMREX_GPU_HOST_DEVICE void CKXTY(double *  x,  double *  y)
{
    double XW = 0; /*See Eq 4, 9 in CK Manual */
    /*Compute mean molecular wt first */
    XW += x[0]*0.000549; /*E */
    XW += x[1]*31.998800; /*O2 */
    XW += x[2]*28.013400; /*N2 */
    XW += x[3]*15.999400; /*O */
    XW += x[4]*31.998251; /*O2+ */
    XW += x[5]*28.012851; /*N2+ */
    XW += x[6]*63.997051; /*O4+ */
    XW += x[7]*56.026251; /*N4+ */
    XW += x[8]*63.997051; /*O2pN2 */
    XW += x[9]*31.999349; /*O2- */
    /*Now compute conversion */
    double XWinv = 1.0/XW;
    y[0] = x[0]*0.000549*XWinv; 
    y[1] = x[1]*31.998800*XWinv; 
    y[2] = x[2]*28.013400*XWinv; 
    y[3] = x[3]*15.999400*XWinv; 
    y[4] = x[4]*31.998251*XWinv; 
    y[5] = x[5]*28.012851*XWinv; 
    y[6] = x[6]*63.997051*XWinv; 
    y[7] = x[7]*56.026251*XWinv; 
    y[8] = x[8]*63.997051*XWinv; 
    y[9] = x[9]*31.999349*XWinv; 

    return;
}


/*convert x[species] (mole fracs) to c[species] (molar conc) */
void CKXTCP(double *  P, double *  T, double *  x,  double *  c)
{
    int id; /*loop counter */
    double PORT = (*P)/(8.31446261815324e+07 * (*T)); /*P/RT */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 10; ++id) {
        c[id] = x[id]*PORT;
    }

    return;
}


/*convert x[species] (mole fracs) to c[species] (molar conc) */
void CKXTCR(double *  rho, double *  T, double *  x, double *  c)
{
    int id; /*loop counter */
    double XW = 0; /*See Eq 4, 11 in CK Manual */
    double ROW; 
    /*Compute mean molecular wt first */
    XW += x[0]*0.000549; /*E */
    XW += x[1]*31.998800; /*O2 */
    XW += x[2]*28.013400; /*N2 */
    XW += x[3]*15.999400; /*O */
    XW += x[4]*31.998251; /*O2+ */
    XW += x[5]*28.012851; /*N2+ */
    XW += x[6]*63.997051; /*O4+ */
    XW += x[7]*56.026251; /*N4+ */
    XW += x[8]*63.997051; /*O2pN2 */
    XW += x[9]*31.999349; /*O2- */
    ROW = (*rho) / XW;

    /*Compute conversion, see Eq 11 */
    for (id = 0; id < 10; ++id) {
        c[id] = x[id]*ROW;
    }

    return;
}


/*convert c[species] (molar conc) to x[species] (mole fracs) */
void CKCTX(double *  c, double *  x)
{
    int id; /*loop counter */
    double sumC = 0; 

    /*compute sum of c  */
    for (id = 0; id < 10; ++id) {
        sumC += c[id];
    }

    /* See Eq 13  */
    double sumCinv = 1.0/sumC;
    for (id = 0; id < 10; ++id) {
        x[id] = c[id]*sumCinv;
    }

    return;
}


/*convert c[species] (molar conc) to y[species] (mass fracs) */
void CKCTY(double *  c, double *  y)
{
    double CW = 0; /*See Eq 12 in CK Manual */
    /*compute denominator in eq 12 first */
    CW += c[0]*0.000549; /*E */
    CW += c[1]*31.998800; /*O2 */
    CW += c[2]*28.013400; /*N2 */
    CW += c[3]*15.999400; /*O */
    CW += c[4]*31.998251; /*O2+ */
    CW += c[5]*28.012851; /*N2+ */
    CW += c[6]*63.997051; /*O4+ */
    CW += c[7]*56.026251; /*N4+ */
    CW += c[8]*63.997051; /*O2pN2 */
    CW += c[9]*31.999349; /*O2- */
    /*Now compute conversion */
    double CWinv = 1.0/CW;
    y[0] = c[0]*0.000549*CWinv; 
    y[1] = c[1]*31.998800*CWinv; 
    y[2] = c[2]*28.013400*CWinv; 
    y[3] = c[3]*15.999400*CWinv; 
    y[4] = c[4]*31.998251*CWinv; 
    y[5] = c[5]*28.012851*CWinv; 
    y[6] = c[6]*63.997051*CWinv; 
    y[7] = c[7]*56.026251*CWinv; 
    y[8] = c[8]*63.997051*CWinv; 
    y[9] = c[9]*31.999349*CWinv; 

    return;
}


/*get Cp/R as a function of T  */
/*for all species (Eq 19) */
void CKCPOR(double *  T, double *  cpor)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    cp_R(cpor, tc);
}


/*get H/RT as a function of T  */
/*for all species (Eq 20) */
void CKHORT(double *  T, double *  hort)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    speciesEnthalpy(hort, tc);
}


/*get S/R as a function of T  */
/*for all species (Eq 21) */
void CKSOR(double *  T, double *  sor)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    speciesEntropy(sor, tc);
}


/*get specific heat at constant volume as a function  */
/*of T for all species (molar units) */
void CKCVML(double *  T,  double *  cvml)
{
    int id; /*loop counter */
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    cv_R(cvml, tc);

    /*convert to chemkin units */
    for (id = 0; id < 10; ++id) {
        cvml[id] *= 8.31446261815324e+07;
    }
}


/*get specific heat at constant pressure as a  */
/*function of T for all species (molar units) */
void CKCPML(double *  T,  double *  cpml)
{
    int id; /*loop counter */
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    cp_R(cpml, tc);

    /*convert to chemkin units */
    for (id = 0; id < 10; ++id) {
        cpml[id] *= 8.31446261815324e+07;
    }
}


/*get internal energy as a function  */
/*of T for all species (molar units) */
void CKUML(double *  T,  double *  uml)
{
    int id; /*loop counter */
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    speciesInternalEnergy(uml, tc);

    /*convert to chemkin units */
    for (id = 0; id < 10; ++id) {
        uml[id] *= RT;
    }
}


/*get enthalpy as a function  */
/*of T for all species (molar units) */
void CKHML(double *  T,  double *  hml)
{
    int id; /*loop counter */
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    speciesEnthalpy(hml, tc);

    /*convert to chemkin units */
    for (id = 0; id < 10; ++id) {
        hml[id] *= RT;
    }
}


/*get standard-state Gibbs energy as a function  */
/*of T for all species (molar units) */
void CKGML(double *  T,  double *  gml)
{
    int id; /*loop counter */
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    gibbs(gml, tc);

    /*convert to chemkin units */
    for (id = 0; id < 10; ++id) {
        gml[id] *= RT;
    }
}


/*get standard-state Helmholtz free energy as a  */
/*function of T for all species (molar units) */
void CKAML(double *  T,  double *  aml)
{
    int id; /*loop counter */
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    helmholtz(aml, tc);

    /*convert to chemkin units */
    for (id = 0; id < 10; ++id) {
        aml[id] *= RT;
    }
}


/*Returns the standard-state entropies in molar units */
void CKSML(double *  T,  double *  sml)
{
    int id; /*loop counter */
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    speciesEntropy(sml, tc);

    /*convert to chemkin units */
    for (id = 0; id < 10; ++id) {
        sml[id] *= 8.31446261815324e+07;
    }
}


/*Returns the specific heats at constant volume */
/*in mass units (Eq. 29) */
AMREX_GPU_HOST_DEVICE void CKCVMS(double *  T,  double *  cvms)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    cv_R(cvms, tc);
    /*multiply by R/molecularweight */
    cvms[0] *= 1.515633290043376e+11; /*E */
    cvms[1] *= 2.598367006935648e+06; /*O2 */
    cvms[2] *= 2.968030520448514e+06; /*N2 */
    cvms[3] *= 5.196734013871295e+06; /*O */
    cvms[4] *= 2.598411553508327e+06; /*O2+ */
    cvms[5] *= 2.968088643859633e+06; /*N2+ */
    cvms[6] *= 1.299194640015531e+06; /*O4+ */
    cvms[7] *= 1.484029790934758e+06; /*N4+ */
    cvms[8] *= 1.299194640015531e+06; /*O2pN2 */
    cvms[9] *= 2.598322461890334e+06; /*O2- */
}


/*Returns the specific heats at constant pressure */
/*in mass units (Eq. 26) */
AMREX_GPU_HOST_DEVICE void CKCPMS(double *  T,  double *  cpms)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    cp_R(cpms, tc);
    /*multiply by R/molecularweight */
    cpms[0] *= 1.515633290043376e+11; /*E */
    cpms[1] *= 2.598367006935648e+06; /*O2 */
    cpms[2] *= 2.968030520448514e+06; /*N2 */
    cpms[3] *= 5.196734013871295e+06; /*O */
    cpms[4] *= 2.598411553508327e+06; /*O2+ */
    cpms[5] *= 2.968088643859633e+06; /*N2+ */
    cpms[6] *= 1.299194640015531e+06; /*O4+ */
    cpms[7] *= 1.484029790934758e+06; /*N4+ */
    cpms[8] *= 1.299194640015531e+06; /*O2pN2 */
    cpms[9] *= 2.598322461890334e+06; /*O2- */
}


/*Returns internal energy in mass units (Eq 30.) */
AMREX_GPU_HOST_DEVICE void CKUMS(double *  T,  double *  ums)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    speciesInternalEnergy(ums, tc);
    for (int i = 0; i < 10; i++)
    {
        ums[i] *= RT*imw[i];
    }
}


/*Returns enthalpy in mass units (Eq 27.) */
AMREX_GPU_HOST_DEVICE void CKHMS(double *  T,  double *  hms)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    speciesEnthalpy(hms, tc);
    for (int i = 0; i < 10; i++)
    {
        hms[i] *= RT*imw[i];
    }
}


#ifndef AMREX_USE_CUDA
/*Returns enthalpy in mass units (Eq 27.) */
void VCKHMS(int *  np, double *  T,  double *  hms)
{
    double tc[5], h[10];

    for (int i=0; i<(*np); i++) {
        tc[0] = 0.0;
        tc[1] = T[i];
        tc[2] = T[i]*T[i];
        tc[3] = T[i]*T[i]*T[i];
        tc[4] = T[i]*T[i]*T[i]*T[i];

        speciesEnthalpy(h, tc);

        hms[0*(*np)+i] = h[0];
        hms[1*(*np)+i] = h[1];
        hms[2*(*np)+i] = h[2];
        hms[3*(*np)+i] = h[3];
        hms[4*(*np)+i] = h[4];
        hms[5*(*np)+i] = h[5];
        hms[6*(*np)+i] = h[6];
        hms[7*(*np)+i] = h[7];
        hms[8*(*np)+i] = h[8];
        hms[9*(*np)+i] = h[9];
    }

    for (int n=0; n<10; n++) {
        for (int i=0; i<(*np); i++) {
            hms[n*(*np)+i] *= 8.31446261815324e+07 * T[i] * imw[n];
        }
    }
}
#else
/*TODO: remove this on GPU */
void VCKHMS(int *  np, double *  T,  double *  hms)
{
}
#endif


/*Returns gibbs in mass units (Eq 31.) */
void CKGMS(double *  T,  double *  gms)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    gibbs(gms, tc);
    for (int i = 0; i < 10; i++)
    {
        gms[i] *= RT*imw[i];
    }
}


/*Returns helmholtz in mass units (Eq 32.) */
void CKAMS(double *  T,  double *  ams)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    helmholtz(ams, tc);
    for (int i = 0; i < 10; i++)
    {
        ams[i] *= RT*imw[i];
    }
}


/*Returns the entropies in mass units (Eq 28.) */
void CKSMS(double *  T,  double *  sms)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    speciesEntropy(sms, tc);
    /*multiply by R/molecularweight */
    sms[0] *= 1.515633290043376e+11; /*E */
    sms[1] *= 2.598367006935648e+06; /*O2 */
    sms[2] *= 2.968030520448514e+06; /*N2 */
    sms[3] *= 5.196734013871295e+06; /*O */
    sms[4] *= 2.598411553508327e+06; /*O2+ */
    sms[5] *= 2.968088643859633e+06; /*N2+ */
    sms[6] *= 1.299194640015531e+06; /*O4+ */
    sms[7] *= 1.484029790934758e+06; /*N4+ */
    sms[8] *= 1.299194640015531e+06; /*O2pN2 */
    sms[9] *= 2.598322461890334e+06; /*O2- */
}


/*Returns the mean specific heat at CP (Eq. 33) */
void CKCPBL(double *  T, double *  x,  double *  cpbl)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double cpor[10]; /* temporary storage */
    cp_R(cpor, tc);

    /*perform dot product */
    for (id = 0; id < 10; ++id) {
        result += x[id]*cpor[id];
    }

    *cpbl = result * 8.31446261815324e+07;
}


/*Returns the mean specific heat at CP (Eq. 34) */
AMREX_GPU_HOST_DEVICE void CKCPBS(double *  T, double *  y,  double *  cpbs)
{
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double cpor[10], tresult[10]; /* temporary storage */
    cp_R(cpor, tc);
    for (int i = 0; i < 10; i++)
    {
        tresult[i] = cpor[i]*y[i]*imw[i];

    }
    for (int i = 0; i < 10; i++)
    {
        result += tresult[i];
    }

    *cpbs = result * 8.31446261815324e+07;
}


/*Returns the mean specific heat at CV (Eq. 35) */
void CKCVBL(double *  T, double *  x,  double *  cvbl)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double cvor[10]; /* temporary storage */
    cv_R(cvor, tc);

    /*perform dot product */
    for (id = 0; id < 10; ++id) {
        result += x[id]*cvor[id];
    }

    *cvbl = result * 8.31446261815324e+07;
}


/*Returns the mean specific heat at CV (Eq. 36) */
AMREX_GPU_HOST_DEVICE void CKCVBS(double *  T, double *  y,  double *  cvbs)
{
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double cvor[10]; /* temporary storage */
    cv_R(cvor, tc);
    /*multiply by y/molecularweight */
    result += cvor[0]*y[0]*imw[0]; /*E */
    result += cvor[1]*y[1]*imw[1]; /*O2 */
    result += cvor[2]*y[2]*imw[2]; /*N2 */
    result += cvor[3]*y[3]*imw[3]; /*O */
    result += cvor[4]*y[4]*imw[4]; /*O2+ */
    result += cvor[5]*y[5]*imw[5]; /*N2+ */
    result += cvor[6]*y[6]*imw[6]; /*O4+ */
    result += cvor[7]*y[7]*imw[7]; /*N4+ */
    result += cvor[8]*y[8]*imw[8]; /*O2pN2 */
    result += cvor[9]*y[9]*imw[9]; /*O2- */

    *cvbs = result * 8.31446261815324e+07;
}


/*Returns the mean enthalpy of the mixture in molar units */
void CKHBML(double *  T, double *  x,  double *  hbml)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double hml[10]; /* temporary storage */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    speciesEnthalpy(hml, tc);

    /*perform dot product */
    for (id = 0; id < 10; ++id) {
        result += x[id]*hml[id];
    }

    *hbml = result * RT;
}


/*Returns mean enthalpy of mixture in mass units */
AMREX_GPU_HOST_DEVICE void CKHBMS(double *  T, double *  y,  double *  hbms)
{
    double result = 0;
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double hml[10], tmp[10]; /* temporary storage */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    speciesEnthalpy(hml, tc);
    int id;
    for (id = 0; id < 10; ++id) {
        tmp[id] = y[id]*hml[id]*imw[id];
    }
    for (id = 0; id < 10; ++id) {
        result += tmp[id];
    }

    *hbms = result * RT;
}


/*get mean internal energy in molar units */
void CKUBML(double *  T, double *  x,  double *  ubml)
{
    int id; /*loop counter */
    double result = 0; 
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double uml[10]; /* temporary energy array */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    speciesInternalEnergy(uml, tc);

    /*perform dot product */
    for (id = 0; id < 10; ++id) {
        result += x[id]*uml[id];
    }

    *ubml = result * RT;
}


/*get mean internal energy in mass units */
AMREX_GPU_HOST_DEVICE void CKUBMS(double *  T, double *  y,  double *  ubms)
{
    double result = 0;
    double tT = *T; /*temporary temperature */
    double tc[] = { 0, tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double ums[10]; /* temporary energy array */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    speciesInternalEnergy(ums, tc);
    /*perform dot product + scaling by wt */
    result += y[0]*ums[0]*imw[0]; /*E */
    result += y[1]*ums[1]*imw[1]; /*O2 */
    result += y[2]*ums[2]*imw[2]; /*N2 */
    result += y[3]*ums[3]*imw[3]; /*O */
    result += y[4]*ums[4]*imw[4]; /*O2+ */
    result += y[5]*ums[5]*imw[5]; /*N2+ */
    result += y[6]*ums[6]*imw[6]; /*O4+ */
    result += y[7]*ums[7]*imw[7]; /*N4+ */
    result += y[8]*ums[8]*imw[8]; /*O2pN2 */
    result += y[9]*ums[9]*imw[9]; /*O2- */

    *ubms = result * RT;
}


/*get mixture entropy in molar units */
void CKSBML(double *  P, double *  T, double *  x,  double *  sbml)
{
    int id; /*loop counter */
    double result = 0; 
    /*Log of normalized pressure in cgs units dynes/cm^2 by Patm */
    double logPratio = log ( *P / 1013250.0 ); 
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double sor[10]; /* temporary storage */
    speciesEntropy(sor, tc);

    /*Compute Eq 42 */
    for (id = 0; id < 10; ++id) {
        result += x[id]*(sor[id]-log((x[id]+1e-100))-logPratio);
    }

    *sbml = result * 8.31446261815324e+07;
}


/*get mixture entropy in mass units */
void CKSBMS(double *  P, double *  T, double *  y,  double *  sbms)
{
    double result = 0; 
    /*Log of normalized pressure in cgs units dynes/cm^2 by Patm */
    double logPratio = log ( *P / 1013250.0 ); 
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double sor[10]; /* temporary storage */
    double x[10]; /* need a ytx conversion */
    double YOW = 0; /*See Eq 4, 6 in CK Manual */
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*E */
    YOW += y[1]*imw[1]; /*O2 */
    YOW += y[2]*imw[2]; /*N2 */
    YOW += y[3]*imw[3]; /*O */
    YOW += y[4]*imw[4]; /*O2+ */
    YOW += y[5]*imw[5]; /*N2+ */
    YOW += y[6]*imw[6]; /*O4+ */
    YOW += y[7]*imw[7]; /*N4+ */
    YOW += y[8]*imw[8]; /*O2pN2 */
    YOW += y[9]*imw[9]; /*O2- */
    /*Now compute y to x conversion */
    x[0] = y[0]/(0.000549*YOW); 
    x[1] = y[1]/(31.998800*YOW); 
    x[2] = y[2]/(28.013400*YOW); 
    x[3] = y[3]/(15.999400*YOW); 
    x[4] = y[4]/(31.998251*YOW); 
    x[5] = y[5]/(28.012851*YOW); 
    x[6] = y[6]/(63.997051*YOW); 
    x[7] = y[7]/(56.026251*YOW); 
    x[8] = y[8]/(63.997051*YOW); 
    x[9] = y[9]/(31.999349*YOW); 
    speciesEntropy(sor, tc);
    /*Perform computation in Eq 42 and 43 */
    result += x[0]*(sor[0]-log((x[0]+1e-100))-logPratio);
    result += x[1]*(sor[1]-log((x[1]+1e-100))-logPratio);
    result += x[2]*(sor[2]-log((x[2]+1e-100))-logPratio);
    result += x[3]*(sor[3]-log((x[3]+1e-100))-logPratio);
    result += x[4]*(sor[4]-log((x[4]+1e-100))-logPratio);
    result += x[5]*(sor[5]-log((x[5]+1e-100))-logPratio);
    result += x[6]*(sor[6]-log((x[6]+1e-100))-logPratio);
    result += x[7]*(sor[7]-log((x[7]+1e-100))-logPratio);
    result += x[8]*(sor[8]-log((x[8]+1e-100))-logPratio);
    result += x[9]*(sor[9]-log((x[9]+1e-100))-logPratio);
    /*Scale by R/W */
    *sbms = result * 8.31446261815324e+07 * YOW;
}


/*Returns mean gibbs free energy in molar units */
void CKGBML(double *  P, double *  T, double *  x,  double *  gbml)
{
    int id; /*loop counter */
    double result = 0; 
    /*Log of normalized pressure in cgs units dynes/cm^2 by Patm */
    double logPratio = log ( *P / 1013250.0 ); 
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    double gort[10]; /* temporary storage */
    /*Compute g/RT */
    gibbs(gort, tc);

    /*Compute Eq 44 */
    for (id = 0; id < 10; ++id) {
        result += x[id]*(gort[id]+log((x[id]+1e-100))+logPratio);
    }

    *gbml = result * RT;
}


/*Returns mixture gibbs free energy in mass units */
void CKGBMS(double *  P, double *  T, double *  y,  double *  gbms)
{
    double result = 0; 
    /*Log of normalized pressure in cgs units dynes/cm^2 by Patm */
    double logPratio = log ( *P / 1013250.0 ); 
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    double gort[10]; /* temporary storage */
    double x[10]; /* need a ytx conversion */
    double YOW = 0; /*To hold 1/molecularweight */
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*E */
    YOW += y[1]*imw[1]; /*O2 */
    YOW += y[2]*imw[2]; /*N2 */
    YOW += y[3]*imw[3]; /*O */
    YOW += y[4]*imw[4]; /*O2+ */
    YOW += y[5]*imw[5]; /*N2+ */
    YOW += y[6]*imw[6]; /*O4+ */
    YOW += y[7]*imw[7]; /*N4+ */
    YOW += y[8]*imw[8]; /*O2pN2 */
    YOW += y[9]*imw[9]; /*O2- */
    /*Now compute y to x conversion */
    x[0] = y[0]/(0.000549*YOW); 
    x[1] = y[1]/(31.998800*YOW); 
    x[2] = y[2]/(28.013400*YOW); 
    x[3] = y[3]/(15.999400*YOW); 
    x[4] = y[4]/(31.998251*YOW); 
    x[5] = y[5]/(28.012851*YOW); 
    x[6] = y[6]/(63.997051*YOW); 
    x[7] = y[7]/(56.026251*YOW); 
    x[8] = y[8]/(63.997051*YOW); 
    x[9] = y[9]/(31.999349*YOW); 
    gibbs(gort, tc);
    /*Perform computation in Eq 44 */
    result += x[0]*(gort[0]+log((x[0]+1e-100))+logPratio);
    result += x[1]*(gort[1]+log((x[1]+1e-100))+logPratio);
    result += x[2]*(gort[2]+log((x[2]+1e-100))+logPratio);
    result += x[3]*(gort[3]+log((x[3]+1e-100))+logPratio);
    result += x[4]*(gort[4]+log((x[4]+1e-100))+logPratio);
    result += x[5]*(gort[5]+log((x[5]+1e-100))+logPratio);
    result += x[6]*(gort[6]+log((x[6]+1e-100))+logPratio);
    result += x[7]*(gort[7]+log((x[7]+1e-100))+logPratio);
    result += x[8]*(gort[8]+log((x[8]+1e-100))+logPratio);
    result += x[9]*(gort[9]+log((x[9]+1e-100))+logPratio);
    /*Scale by RT/W */
    *gbms = result * RT * YOW;
}


/*Returns mean helmholtz free energy in molar units */
void CKABML(double *  P, double *  T, double *  x,  double *  abml)
{
    int id; /*loop counter */
    double result = 0; 
    /*Log of normalized pressure in cgs units dynes/cm^2 by Patm */
    double logPratio = log ( *P / 1013250.0 ); 
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    double aort[10]; /* temporary storage */
    /*Compute g/RT */
    helmholtz(aort, tc);

    /*Compute Eq 44 */
    for (id = 0; id < 10; ++id) {
        result += x[id]*(aort[id]+log((x[id]+1e-100))+logPratio);
    }

    *abml = result * RT;
}


/*Returns mixture helmholtz free energy in mass units */
void CKABMS(double *  P, double *  T, double *  y,  double *  abms)
{
    double result = 0; 
    /*Log of normalized pressure in cgs units dynes/cm^2 by Patm */
    double logPratio = log ( *P / 1013250.0 ); 
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double RT = 8.31446261815324e+07*tT; /*R*T */
    double aort[10]; /* temporary storage */
    double x[10]; /* need a ytx conversion */
    double YOW = 0; /*To hold 1/molecularweight */
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*E */
    YOW += y[1]*imw[1]; /*O2 */
    YOW += y[2]*imw[2]; /*N2 */
    YOW += y[3]*imw[3]; /*O */
    YOW += y[4]*imw[4]; /*O2+ */
    YOW += y[5]*imw[5]; /*N2+ */
    YOW += y[6]*imw[6]; /*O4+ */
    YOW += y[7]*imw[7]; /*N4+ */
    YOW += y[8]*imw[8]; /*O2pN2 */
    YOW += y[9]*imw[9]; /*O2- */
    /*Now compute y to x conversion */
    x[0] = y[0]/(0.000549*YOW); 
    x[1] = y[1]/(31.998800*YOW); 
    x[2] = y[2]/(28.013400*YOW); 
    x[3] = y[3]/(15.999400*YOW); 
    x[4] = y[4]/(31.998251*YOW); 
    x[5] = y[5]/(28.012851*YOW); 
    x[6] = y[6]/(63.997051*YOW); 
    x[7] = y[7]/(56.026251*YOW); 
    x[8] = y[8]/(63.997051*YOW); 
    x[9] = y[9]/(31.999349*YOW); 
    helmholtz(aort, tc);
    /*Perform computation in Eq 44 */
    result += x[0]*(aort[0]+log((x[0]+1e-100))+logPratio);
    result += x[1]*(aort[1]+log((x[1]+1e-100))+logPratio);
    result += x[2]*(aort[2]+log((x[2]+1e-100))+logPratio);
    result += x[3]*(aort[3]+log((x[3]+1e-100))+logPratio);
    result += x[4]*(aort[4]+log((x[4]+1e-100))+logPratio);
    result += x[5]*(aort[5]+log((x[5]+1e-100))+logPratio);
    result += x[6]*(aort[6]+log((x[6]+1e-100))+logPratio);
    result += x[7]*(aort[7]+log((x[7]+1e-100))+logPratio);
    result += x[8]*(aort[8]+log((x[8]+1e-100))+logPratio);
    result += x[9]*(aort[9]+log((x[9]+1e-100))+logPratio);
    /*Scale by RT/W */
    *abms = result * RT * YOW;
}


/*compute the production rate for each species */
AMREX_GPU_HOST_DEVICE void CKWC(double *  T, double *  C,  double *  wdot, double EoN)
{
    // units of mol/cm3-s
    int id; /*loop counter */

    /*convert to SI */
    for (id = 0; id < 10; ++id) {
        C[id] *= 1.0e6;
    }

    // TODO: remove after testing
    // for (id = 0; id < 10; id++){
    //     C[id] = 4.062176;
    // }

    /*convert to chemkin units */
    productionRate(wdot, C, *T, EoN);

    /*convert to chemkin units (cgs) */
    for (id = 0; id < 10; ++id) {
        C[id] *= 1.0e-6;
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given P, T, and mass fractions */
void CKWYP(double *  P, double *  T, double *  y,  double *  wdot)
{
    int id; /*loop counter */
    double c[10]; /*temporary storage */
    double YOW = 0; 
    double PWORT; 
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*E */
    YOW += y[1]*imw[1]; /*O2 */
    YOW += y[2]*imw[2]; /*N2 */
    YOW += y[3]*imw[3]; /*O */
    YOW += y[4]*imw[4]; /*O2+ */
    YOW += y[5]*imw[5]; /*N2+ */
    YOW += y[6]*imw[6]; /*O4+ */
    YOW += y[7]*imw[7]; /*N4+ */
    YOW += y[8]*imw[8]; /*O2pN2 */
    YOW += y[9]*imw[9]; /*O2- */
    /*PW/RT (see Eq. 7) */
    PWORT = (*P)/(YOW * 8.31446261815324e+07 * (*T)); 
    /*multiply by 1e6 so c goes to SI */
    PWORT *= 1e6; 
    /*Now compute conversion (and go to SI) */
    c[0] = PWORT * y[0]*imw[0]; 
    c[1] = PWORT * y[1]*imw[1]; 
    c[2] = PWORT * y[2]*imw[2]; 
    c[3] = PWORT * y[3]*imw[3]; 
    c[4] = PWORT * y[4]*imw[4]; 
    c[5] = PWORT * y[5]*imw[5]; 
    c[6] = PWORT * y[6]*imw[6]; 
    c[7] = PWORT * y[7]*imw[7]; 
    c[8] = PWORT * y[8]*imw[8]; 
    c[9] = PWORT * y[9]*imw[9]; 


    // ndeak - Temporary double for testing
    double EoN = 100.0;

    /*convert to chemkin units */
    productionRate(wdot, c, *T, EoN);

    /*convert to chemkin units */
    for (id = 0; id < 10; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given P, T, and mole fractions */
void CKWXP(double *  P, double *  T, double *  x,  double *  wdot)
{
    int id; /*loop counter */
    double c[10]; /*temporary storage */
    double PORT = 1e6 * (*P)/(8.31446261815324e+07 * (*T)); /*1e6 * P/RT so c goes to SI units */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 10; ++id) {
        c[id] = x[id]*PORT;
    }

    // ndeak - Temporary double for testing
    double EoN = 100.0;

    /*convert to chemkin units */
    productionRate(wdot, c, *T, EoN);

    /*convert to chemkin units */
    for (id = 0; id < 10; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given rho, T, and mass fractions */
AMREX_GPU_HOST_DEVICE void CKWYR(double *  rho, double *  T, double *  y,  double *  wdot)
{
    int id; /*loop counter */
    double c[10]; /*temporary storage */
    /*See Eq 8 with an extra 1e6 so c goes to SI */
    c[0] = 1e6 * (*rho) * y[0]*imw[0]; 
    c[1] = 1e6 * (*rho) * y[1]*imw[1]; 
    c[2] = 1e6 * (*rho) * y[2]*imw[2]; 
    c[3] = 1e6 * (*rho) * y[3]*imw[3]; 
    c[4] = 1e6 * (*rho) * y[4]*imw[4]; 
    c[5] = 1e6 * (*rho) * y[5]*imw[5]; 
    c[6] = 1e6 * (*rho) * y[6]*imw[6]; 
    c[7] = 1e6 * (*rho) * y[7]*imw[7]; 
    c[8] = 1e6 * (*rho) * y[8]*imw[8]; 
    c[9] = 1e6 * (*rho) * y[9]*imw[9]; 

    // Temporary double for testing
    double EoN = 100.0;

    /*call productionRate */
    productionRate(wdot, c, *T, EoN);

    /*convert to chemkin units */
    for (id = 0; id < 10; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the molar production rate of species */
/*Given rho, T, and mass fractions */
void VCKWYR(int *  np, double *  rho, double *  T,
	    double *  y,
	    double *  wdot)
{
#ifndef AMREX_USE_CUDA
    double c[10*(*np)]; /*temporary storage */
    /*See Eq 8 with an extra 1e6 so c goes to SI */
    for (int n=0; n<10; n++) {
        for (int i=0; i<(*np); i++) {
            c[n*(*np)+i] = 1.0e6 * rho[i] * y[n*(*np)+i] * imw[n];
        }
    }

    /*call productionRate */
    vproductionRate(*np, wdot, c, T);

    /*convert to chemkin units */
    for (int i=0; i<10*(*np); i++) {
        wdot[i] *= 1.0e-6;
    }
#endif
}


/*Returns the molar production rate of species */
/*Given rho, T, and mole fractions */
void CKWXR(double *  rho, double *  T, double *  x,  double *  wdot)
{
    int id; /*loop counter */
    double c[10]; /*temporary storage */
    double XW = 0; /*See Eq 4, 11 in CK Manual */
    double ROW; 
    /*Compute mean molecular wt first */
    XW += x[0]*0.000549; /*E */
    XW += x[1]*31.998800; /*O2 */
    XW += x[2]*28.013400; /*N2 */
    XW += x[3]*15.999400; /*O */
    XW += x[4]*31.998251; /*O2+ */
    XW += x[5]*28.012851; /*N2+ */
    XW += x[6]*63.997051; /*O4+ */
    XW += x[7]*56.026251; /*N4+ */
    XW += x[8]*63.997051; /*O2pN2 */
    XW += x[9]*31.999349; /*O2- */
    /*Extra 1e6 factor to take c to SI */
    ROW = 1e6*(*rho) / XW;

    /*Compute conversion, see Eq 11 */
    for (id = 0; id < 10; ++id) {
        c[id] = x[id]*ROW;
    }

    // ndeak - Temporary double for testing
    double EoN = 100.0;

    /*convert to chemkin units */
    productionRate(wdot, c, *T, EoN);

    /*convert to chemkin units */
    for (id = 0; id < 10; ++id) {
        wdot[id] *= 1.0e-6;
    }
}


/*Returns the rate of progress for each reaction */
void CKQC(double *  T, double *  C, double *  qdot)
{
    int id; /*loop counter */

    /*convert to SI */
    for (id = 0; id < 10; ++id) {
        C[id] *= 1.0e6;
    }

    /*convert to chemkin units */
    progressRate(qdot, C, *T);

    /*convert to chemkin units */
    for (id = 0; id < 10; ++id) {
        C[id] *= 1.0e-6;
    }

    for (id = 0; id < 22; ++id) {
        qdot[id] *= 1.0e-6;
    }
}


/*Returns the progress rates of each reactions */
/*Given P, T, and mole fractions */
void CKKFKR(double *  P, double *  T, double *  x, double *  q_f, double *  q_r)
{
    int id; /*loop counter */
    double c[10]; /*temporary storage */
    double PORT = 1e6 * (*P)/(8.31446261815324e+07 * (*T)); /*1e6 * P/RT so c goes to SI units */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 10; ++id) {
        c[id] = x[id]*PORT;
    }

    /*convert to chemkin units */
    progressRateFR(q_f, q_r, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 22; ++id) {
        q_f[id] *= 1.0e-6;
        q_r[id] *= 1.0e-6;
    }
}


/*Returns the progress rates of each reactions */
/*Given P, T, and mass fractions */
void CKQYP(double *  P, double *  T, double *  y, double *  qdot)
{
    int id; /*loop counter */
    double c[10]; /*temporary storage */
    double YOW = 0; 
    double PWORT; 
    /*Compute inverse of mean molecular wt first */
    YOW += y[0]*imw[0]; /*E */
    YOW += y[1]*imw[1]; /*O2 */
    YOW += y[2]*imw[2]; /*N2 */
    YOW += y[3]*imw[3]; /*O */
    YOW += y[4]*imw[4]; /*O2+ */
    YOW += y[5]*imw[5]; /*N2+ */
    YOW += y[6]*imw[6]; /*O4+ */
    YOW += y[7]*imw[7]; /*N4+ */
    YOW += y[8]*imw[8]; /*O2pN2 */
    YOW += y[9]*imw[9]; /*O2- */
    /*PW/RT (see Eq. 7) */
    PWORT = (*P)/(YOW * 8.31446261815324e+07 * (*T)); 
    /*multiply by 1e6 so c goes to SI */
    PWORT *= 1e6; 
    /*Now compute conversion (and go to SI) */
    c[0] = PWORT * y[0]*imw[0]; 
    c[1] = PWORT * y[1]*imw[1]; 
    c[2] = PWORT * y[2]*imw[2]; 
    c[3] = PWORT * y[3]*imw[3]; 
    c[4] = PWORT * y[4]*imw[4]; 
    c[5] = PWORT * y[5]*imw[5]; 
    c[6] = PWORT * y[6]*imw[6]; 
    c[7] = PWORT * y[7]*imw[7]; 
    c[8] = PWORT * y[8]*imw[8]; 
    c[9] = PWORT * y[9]*imw[9]; 

    /*convert to chemkin units */
    progressRate(qdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 22; ++id) {
        qdot[id] *= 1.0e-6;
    }
}


/*Returns the progress rates of each reactions */
/*Given P, T, and mole fractions */
void CKQXP(double *  P, double *  T, double *  x, double *  qdot)
{
    int id; /*loop counter */
    double c[10]; /*temporary storage */
    double PORT = 1e6 * (*P)/(8.31446261815324e+07 * (*T)); /*1e6 * P/RT so c goes to SI units */

    /*Compute conversion, see Eq 10 */
    for (id = 0; id < 10; ++id) {
        c[id] = x[id]*PORT;
    }

    /*convert to chemkin units */
    progressRate(qdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 22; ++id) {
        qdot[id] *= 1.0e-6;
    }
}


/*Returns the progress rates of each reactions */
/*Given rho, T, and mass fractions */
void CKQYR(double *  rho, double *  T, double *  y, double *  qdot)
{
    int id; /*loop counter */
    double c[10]; /*temporary storage */
    /*See Eq 8 with an extra 1e6 so c goes to SI */
    c[0] = 1e6 * (*rho) * y[0]*imw[0]; 
    c[1] = 1e6 * (*rho) * y[1]*imw[1]; 
    c[2] = 1e6 * (*rho) * y[2]*imw[2]; 
    c[3] = 1e6 * (*rho) * y[3]*imw[3]; 
    c[4] = 1e6 * (*rho) * y[4]*imw[4]; 
    c[5] = 1e6 * (*rho) * y[5]*imw[5]; 
    c[6] = 1e6 * (*rho) * y[6]*imw[6]; 
    c[7] = 1e6 * (*rho) * y[7]*imw[7]; 
    c[8] = 1e6 * (*rho) * y[8]*imw[8]; 
    c[9] = 1e6 * (*rho) * y[9]*imw[9]; 

    /*call progressRate */
    progressRate(qdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 22; ++id) {
        qdot[id] *= 1.0e-6;
    }
}


/*Returns the progress rates of each reactions */
/*Given rho, T, and mole fractions */
void CKQXR(double *  rho, double *  T, double *  x, double *  qdot)
{
    int id; /*loop counter */
    double c[10]; /*temporary storage */
    double XW = 0; /*See Eq 4, 11 in CK Manual */
    double ROW; 
    /*Compute mean molecular wt first */
    XW += x[0]*0.000549; /*E */
    XW += x[1]*31.998800; /*O2 */
    XW += x[2]*28.013400; /*N2 */
    XW += x[3]*15.999400; /*O */
    XW += x[4]*31.998251; /*O2+ */
    XW += x[5]*28.012851; /*N2+ */
    XW += x[6]*63.997051; /*O4+ */
    XW += x[7]*56.026251; /*N4+ */
    XW += x[8]*63.997051; /*O2pN2 */
    XW += x[9]*31.999349; /*O2- */
    /*Extra 1e6 factor to take c to SI */
    ROW = 1e6*(*rho) / XW;

    /*Compute conversion, see Eq 11 */
    for (id = 0; id < 10; ++id) {
        c[id] = x[id]*ROW;
    }

    /*convert to chemkin units */
    progressRate(qdot, c, *T);

    /*convert to chemkin units */
    for (id = 0; id < 22; ++id) {
        qdot[id] *= 1.0e-6;
    }
}


/*Returns the stoichiometric coefficients */
/*of the reaction mechanism. (Eq 50) */
void CKNU(int * kdim,  int * nuki)
{
    int id; /*loop counter */
    int kd = (*kdim); 
    /*Zero nuki */
    for (id = 0; id < 10 * kd; ++ id) {
         nuki[id] = 0; 
    }

    /*reaction 1: E + N2 => E + E + N2+ */
    nuki[ 0 * kd + 0 ] += -1.000000 ;
    nuki[ 2 * kd + 0 ] += -1.000000 ;
    nuki[ 0 * kd + 0 ] += +1.000000 ;
    nuki[ 0 * kd + 0 ] += +1.000000 ;
    nuki[ 5 * kd + 0 ] += +1.000000 ;

    /*reaction 2: E + O2 => E + E + O2+ */
    nuki[ 0 * kd + 1 ] += -1.000000 ;
    nuki[ 1 * kd + 1 ] += -1.000000 ;
    nuki[ 0 * kd + 1 ] += +1.000000 ;
    nuki[ 0 * kd + 1 ] += +1.000000 ;
    nuki[ 4 * kd + 1 ] += +1.000000 ;

    /*reaction 3: N2+ + N2 + N2 => N4+ + N2 */
    nuki[ 5 * kd + 2 ] += -1.000000 ;
    nuki[ 2 * kd + 2 ] += -1.000000 ;
    nuki[ 2 * kd + 2 ] += -1.000000 ;
    nuki[ 7 * kd + 2 ] += +1.000000 ;
    nuki[ 2 * kd + 2 ] += +1.000000 ;

    /*reaction 4: N2+ + N2 + O2 => N4+ + O2 */
    nuki[ 5 * kd + 3 ] += -1.000000 ;
    nuki[ 2 * kd + 3 ] += -1.000000 ;
    nuki[ 1 * kd + 3 ] += -1.000000 ;
    nuki[ 7 * kd + 3 ] += +1.000000 ;
    nuki[ 1 * kd + 3 ] += +1.000000 ;

    /*reaction 5: N4+ + O2 => O2+ + N2 + N2 */
    nuki[ 7 * kd + 4 ] += -1.000000 ;
    nuki[ 1 * kd + 4 ] += -1.000000 ;
    nuki[ 4 * kd + 4 ] += +1.000000 ;
    nuki[ 2 * kd + 4 ] += +1.000000 ;
    nuki[ 2 * kd + 4 ] += +1.000000 ;

    /*reaction 6: N2+ + O2 => O2+ + N2 */
    nuki[ 5 * kd + 5 ] += -1.000000 ;
    nuki[ 1 * kd + 5 ] += -1.000000 ;
    nuki[ 4 * kd + 5 ] += +1.000000 ;
    nuki[ 2 * kd + 5 ] += +1.000000 ;

    /*reaction 7: O2+ + N2 + N2 => O2pN2 + N2 */
    nuki[ 4 * kd + 6 ] += -1.000000 ;
    nuki[ 2 * kd + 6 ] += -1.000000 ;
    nuki[ 2 * kd + 6 ] += -1.000000 ;
    nuki[ 8 * kd + 6 ] += +1.000000 ;
    nuki[ 2 * kd + 6 ] += +1.000000 ;

    /*reaction 8: O2pN2 + N2 => O2+ + N2 + N2 */
    nuki[ 8 * kd + 7 ] += -1.000000 ;
    nuki[ 2 * kd + 7 ] += -1.000000 ;
    nuki[ 4 * kd + 7 ] += +1.000000 ;
    nuki[ 2 * kd + 7 ] += +1.000000 ;
    nuki[ 2 * kd + 7 ] += +1.000000 ;

    /*reaction 9: O2pN2 + O2 => O4+ + N2 */
    nuki[ 8 * kd + 8 ] += -1.000000 ;
    nuki[ 1 * kd + 8 ] += -1.000000 ;
    nuki[ 6 * kd + 8 ] += +1.000000 ;
    nuki[ 2 * kd + 8 ] += +1.000000 ;

    /*reaction 10: O2+ + O2 + N2 => O4+ + N2 */
    nuki[ 4 * kd + 9 ] += -1.000000 ;
    nuki[ 1 * kd + 9 ] += -1.000000 ;
    nuki[ 2 * kd + 9 ] += -1.000000 ;
    nuki[ 6 * kd + 9 ] += +1.000000 ;
    nuki[ 2 * kd + 9 ] += +1.000000 ;

    /*reaction 11: O2+ + O2 + O2 => O4+ + O2 */
    nuki[ 4 * kd + 10 ] += -1.000000 ;
    nuki[ 1 * kd + 10 ] += -1.000000 ;
    nuki[ 1 * kd + 10 ] += -1.000000 ;
    nuki[ 6 * kd + 10 ] += +1.000000 ;
    nuki[ 1 * kd + 10 ] += +1.000000 ;

    /*reaction 12: E + O4+ => O2 + O2 */
    nuki[ 0 * kd + 11 ] += -1.000000 ;
    nuki[ 6 * kd + 11 ] += -1.000000 ;
    nuki[ 1 * kd + 11 ] += +1.000000 ;
    nuki[ 1 * kd + 11 ] += +1.000000 ;

    /*reaction 13: E + O2+ => O + O */
    nuki[ 0 * kd + 12 ] += -1.000000 ;
    nuki[ 4 * kd + 12 ] += -1.000000 ;
    nuki[ 3 * kd + 12 ] += +1.000000 ;
    nuki[ 3 * kd + 12 ] += +1.000000 ;

    /*reaction 14: E + O2 + O2 => O2- + O2 */
    nuki[ 0 * kd + 13 ] += -1.000000 ;
    nuki[ 1 * kd + 13 ] += -1.000000 ;
    nuki[ 1 * kd + 13 ] += -1.000000 ;
    nuki[ 9 * kd + 13 ] += +1.000000 ;
    nuki[ 1 * kd + 13 ] += +1.000000 ;

    /*reaction 15: E + O2 + N2 => O2- + N2 */
    nuki[ 0 * kd + 14 ] += -1.000000 ;
    nuki[ 1 * kd + 14 ] += -1.000000 ;
    nuki[ 2 * kd + 14 ] += -1.000000 ;
    nuki[ 9 * kd + 14 ] += +1.000000 ;
    nuki[ 2 * kd + 14 ] += +1.000000 ;

    /*reaction 16: O2- + O4+ => O2 + O2 + O2 */
    nuki[ 9 * kd + 15 ] += -1.000000 ;
    nuki[ 6 * kd + 15 ] += -1.000000 ;
    nuki[ 1 * kd + 15 ] += +1.000000 ;
    nuki[ 1 * kd + 15 ] += +1.000000 ;
    nuki[ 1 * kd + 15 ] += +1.000000 ;

    /*reaction 17: O2- + O4+ + O2 => O2 + O2 + O2 + O2 */
    nuki[ 9 * kd + 16 ] += -1.000000 ;
    nuki[ 6 * kd + 16 ] += -1.000000 ;
    nuki[ 1 * kd + 16 ] += -1.000000 ;
    nuki[ 1 * kd + 16 ] += +1.000000 ;
    nuki[ 1 * kd + 16 ] += +1.000000 ;
    nuki[ 1 * kd + 16 ] += +1.000000 ;
    nuki[ 1 * kd + 16 ] += +1.000000 ;

    /*reaction 18: O2- + O4+ + N2 => O2 + O2 + O2 + N2 */
    nuki[ 9 * kd + 17 ] += -1.000000 ;
    nuki[ 6 * kd + 17 ] += -1.000000 ;
    nuki[ 2 * kd + 17 ] += -1.000000 ;
    nuki[ 1 * kd + 17 ] += +1.000000 ;
    nuki[ 1 * kd + 17 ] += +1.000000 ;
    nuki[ 1 * kd + 17 ] += +1.000000 ;
    nuki[ 2 * kd + 17 ] += +1.000000 ;

    /*reaction 19: O2- + O2+ + O2 => O2 + O2 + O2 */
    nuki[ 9 * kd + 18 ] += -1.000000 ;
    nuki[ 4 * kd + 18 ] += -1.000000 ;
    nuki[ 1 * kd + 18 ] += -1.000000 ;
    nuki[ 1 * kd + 18 ] += +1.000000 ;
    nuki[ 1 * kd + 18 ] += +1.000000 ;
    nuki[ 1 * kd + 18 ] += +1.000000 ;

    /*reaction 20: O2- + O2+ + N2 => O2 + O2 + N2 */
    nuki[ 9 * kd + 19 ] += -1.000000 ;
    nuki[ 4 * kd + 19 ] += -1.000000 ;
    nuki[ 2 * kd + 19 ] += -1.000000 ;
    nuki[ 1 * kd + 19 ] += +1.000000 ;
    nuki[ 1 * kd + 19 ] += +1.000000 ;
    nuki[ 2 * kd + 19 ] += +1.000000 ;

    /*reaction 21: O2- + O2 => E + O2 + O2 */
    nuki[ 9 * kd + 20 ] += -1.000000 ;
    nuki[ 1 * kd + 20 ] += -1.000000 ;
    nuki[ 0 * kd + 20 ] += +1.000000 ;
    nuki[ 1 * kd + 20 ] += +1.000000 ;
    nuki[ 1 * kd + 20 ] += +1.000000 ;

    /*reaction 22: O2- + N2 => E + O2 + N2 */
    nuki[ 9 * kd + 21 ] += -1.000000 ;
    nuki[ 2 * kd + 21 ] += -1.000000 ;
    nuki[ 0 * kd + 21 ] += +1.000000 ;
    nuki[ 1 * kd + 21 ] += +1.000000 ;
    nuki[ 2 * kd + 21 ] += +1.000000 ;
}


#ifndef AMREX_USE_CUDA
/*Returns a count of species in a reaction, and their indices */
/*and stoichiometric coefficients. (Eq 50) */
void CKINU(int * i, int * nspec, int * ki, int * nu)
{
    if (*i < 1) {
        /*Return max num species per reaction */
        *nspec = 7;
    } else {
        if (*i > 22) {
            *nspec = -1;
        } else {
            *nspec = kiv[*i-1].size();
            for (int j=0; j<*nspec; ++j) {
                ki[j] = kiv[*i-1][j] + 1;
                nu[j] = nuv[*i-1][j];
            }
        }
    }
}
#endif


/*Returns the elemental composition  */
/*of the speciesi (mdim is num of elements) */
void CKNCF(int * ncf)
{
    int id; /*loop counter */
    int kd = 3; 
    /*Zero ncf */
    for (id = 0; id < kd * 10; ++ id) {
         ncf[id] = 0; 
    }

    /*E */
    ncf[ 0 * kd + 2 ] = 1; /*E */

    /*O2 */
    ncf[ 1 * kd + 0 ] = 2; /*O */

    /*N2 */
    ncf[ 2 * kd + 1 ] = 2; /*N */

    /*O */
    ncf[ 3 * kd + 0 ] = 1; /*O */

    /*O2+ */
    ncf[ 4 * kd + 0 ] = 2; /*O */
    ncf[ 4 * kd + 2 ] = -1; /*E */

    /*N2+ */
    ncf[ 5 * kd + 1 ] = 2; /*N */
    ncf[ 5 * kd + 2 ] = -1; /*E */

    /*O4+ */
    ncf[ 6 * kd + 0 ] = 4; /*O */
    ncf[ 6 * kd + 2 ] = -1; /*E */

    /*N4+ */
    ncf[ 7 * kd + 1 ] = 4; /*N */
    ncf[ 7 * kd + 2 ] = -1; /*E */

    /*O2pN2 */
    ncf[ 8 * kd + 0 ] = 4; /*O */
    ncf[ 8 * kd + 2 ] = -1; /*E */

    /*O2- */
    ncf[ 9 * kd + 0 ] = 2; /*O */
    ncf[ 9 * kd + 2 ] = 1; /*E */

}


/*Returns the arrehenius coefficients  */
/*for all reactions */
void CKABE( double *  a, double *  b, double *  e)
{
    // (0):  E + N2 => E + E + N2+
    a[0] = 1;
    b[0] = 0;
    e[0] = 1;

    // (1):  E + O2 => E + E + O2+
    a[1] = 1;
    b[1] = 0;
    e[1] = 1;

    // (2):  N2+ + N2 + N2 => N4+ + N2
    a[2] = 1.8132242e+19;
    b[2] = 0;
    e[2] = 0;

    // (3):  N2+ + N2 + O2 => N4+ + O2
    a[3] = 1.8132242e+19;
    b[3] = 0;
    e[3] = 0;

    // (4):  N4+ + O2 => O2+ + N2 + N2
    a[4] = 150550000000000;
    b[4] = 0;
    e[4] = 0;

    // (5):  N2+ + O2 => O2+ + N2
    a[5] = 36132000000000;
    b[5] = 0;
    e[5] = 0;

    // (6):  O2+ + N2 + N2 => O2pN2 + N2
    a[6] = 3.26380356e+17;
    b[6] = 0;
    e[6] = 0;

    // (7):  O2pN2 + N2 => O2+ + N2 + N2
    a[7] = 258946000000000;
    b[7] = 0;
    e[7] = 0;

    // (8):  O2pN2 + O2 => O4+ + N2
    a[8] = 602200000000000;
    b[8] = 0;
    e[8] = 0;

    // (9):  O2+ + O2 + N2 => O4+ + N2
    a[9] = 8.70347616e+17;
    b[9] = 0;
    e[9] = 0;

    // (10):  O2+ + O2 + O2 => O4+ + O2
    a[10] = 8.70347616e+17;
    b[10] = 0;
    e[10] = 0;

    // (11):  E + O4+ => O2 + O2
    a[11] = 1;
    b[11] = 0;
    e[11] = 0;

    // (12):  E + O2+ => O + O
    a[12] = 1;
    b[12] = 0;
    e[12] = 0;

    // (13):  E + O2 + O2 => O2- + O2
    a[13] = 1;
    b[13] = 0;
    e[13] = 0;

    // (14):  E + O2 + N2 => O2- + N2
    a[14] = 1;
    b[14] = 0;
    e[14] = 0;

    // (15):  O2- + O4+ => O2 + O2 + O2
    a[15] = 60220000000000000;
    b[15] = 0;
    e[15] = 0;

    // (16):  O2- + O4+ + O2 => O2 + O2 + O2 + O2
    a[16] = 7.2528968000000003e+22;
    b[16] = 0;
    e[16] = 0;

    // (17):  O2- + O4+ + N2 => O2 + O2 + O2 + N2
    a[17] = 7.2528968000000003e+22;
    b[17] = 0;
    e[17] = 0;

    // (18):  O2- + O2+ + O2 => O2 + O2 + O2
    a[18] = 7.2528968000000003e+22;
    b[18] = 0;
    e[18] = 0;

    // (19):  O2- + O2+ + N2 => O2 + O2 + N2
    a[19] = 7.2528968000000003e+22;
    b[19] = 0;
    e[19] = 0;

    // (20):  O2- + O2 => E + O2 + O2
    a[20] = 1;
    b[20] = 0;
    e[20] = 0;

    // (21):  O2- + N2 => E + O2 + N2
    a[21] = 1;
    b[21] = 0;
    e[21] = 0;


    return;
}


/*Returns the equil constants for each reaction */
void CKEQC(double *  T, double *  C, double *  eqcon)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double gort[10]; /* temporary storage */

    /*compute the Gibbs free energy */
    gibbs(gort, tc);

    /*compute the equilibrium constants */
    equilibriumConstants(eqcon, gort, tT);

    /*reaction 1: E + N2 => E + E + N2+ */
    eqcon[0] *= 1e-06; 

    /*reaction 2: E + O2 => E + E + O2+ */
    eqcon[1] *= 1e-06; 

    /*reaction 3: N2+ + N2 + N2 => N4+ + N2 */
    eqcon[2] *= 1e+06; 

    /*reaction 4: N2+ + N2 + O2 => N4+ + O2 */
    eqcon[3] *= 1e+06; 

    /*reaction 5: N4+ + O2 => O2+ + N2 + N2 */
    eqcon[4] *= 1e-06; 

    /*reaction 6: N2+ + O2 => O2+ + N2 */
    /*eqcon[5] *= 1;  */

    /*reaction 7: O2+ + N2 + N2 => O2pN2 + N2 */
    eqcon[6] *= 1e+06; 

    /*reaction 8: O2pN2 + N2 => O2+ + N2 + N2 */
    eqcon[7] *= 1e-06; 

    /*reaction 9: O2pN2 + O2 => O4+ + N2 */
    /*eqcon[8] *= 1;  */

    /*reaction 10: O2+ + O2 + N2 => O4+ + N2 */
    eqcon[9] *= 1e+06; 

    /*reaction 11: O2+ + O2 + O2 => O4+ + O2 */
    eqcon[10] *= 1e+06; 

    /*reaction 12: E + O4+ => O2 + O2 */
    /*eqcon[11] *= 1;  */

    /*reaction 13: E + O2+ => O + O */
    /*eqcon[12] *= 1;  */

    /*reaction 14: E + O2 + O2 => O2- + O2 */
    eqcon[13] *= 1e+06; 

    /*reaction 15: E + O2 + N2 => O2- + N2 */
    eqcon[14] *= 1e+06; 

    /*reaction 16: O2- + O4+ => O2 + O2 + O2 */
    eqcon[15] *= 1e-06; 

    /*reaction 17: O2- + O4+ + O2 => O2 + O2 + O2 + O2 */
    eqcon[16] *= 1e-06; 

    /*reaction 18: O2- + O4+ + N2 => O2 + O2 + O2 + N2 */
    eqcon[17] *= 1e-06; 

    /*reaction 19: O2- + O2+ + O2 => O2 + O2 + O2 */
    /*eqcon[18] *= 1;  */

    /*reaction 20: O2- + O2+ + N2 => O2 + O2 + N2 */
    /*eqcon[19] *= 1;  */

    /*reaction 21: O2- + O2 => E + O2 + O2 */
    eqcon[20] *= 1e-06; 

    /*reaction 22: O2- + N2 => E + O2 + N2 */
    eqcon[21] *= 1e-06; 
}


/*Returns the equil constants for each reaction */
/*Given P, T, and mass fractions */
void CKEQYP(double *  P, double *  T, double *  y, double *  eqcon)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double gort[10]; /* temporary storage */

    /*compute the Gibbs free energy */
    gibbs(gort, tc);

    /*compute the equilibrium constants */
    equilibriumConstants(eqcon, gort, tT);

    /*reaction 1: E + N2 => E + E + N2+ */
    eqcon[0] *= 1e-06; 

    /*reaction 2: E + O2 => E + E + O2+ */
    eqcon[1] *= 1e-06; 

    /*reaction 3: N2+ + N2 + N2 => N4+ + N2 */
    eqcon[2] *= 1e+06; 

    /*reaction 4: N2+ + N2 + O2 => N4+ + O2 */
    eqcon[3] *= 1e+06; 

    /*reaction 5: N4+ + O2 => O2+ + N2 + N2 */
    eqcon[4] *= 1e-06; 

    /*reaction 6: N2+ + O2 => O2+ + N2 */
    /*eqcon[5] *= 1;  */

    /*reaction 7: O2+ + N2 + N2 => O2pN2 + N2 */
    eqcon[6] *= 1e+06; 

    /*reaction 8: O2pN2 + N2 => O2+ + N2 + N2 */
    eqcon[7] *= 1e-06; 

    /*reaction 9: O2pN2 + O2 => O4+ + N2 */
    /*eqcon[8] *= 1;  */

    /*reaction 10: O2+ + O2 + N2 => O4+ + N2 */
    eqcon[9] *= 1e+06; 

    /*reaction 11: O2+ + O2 + O2 => O4+ + O2 */
    eqcon[10] *= 1e+06; 

    /*reaction 12: E + O4+ => O2 + O2 */
    /*eqcon[11] *= 1;  */

    /*reaction 13: E + O2+ => O + O */
    /*eqcon[12] *= 1;  */

    /*reaction 14: E + O2 + O2 => O2- + O2 */
    eqcon[13] *= 1e+06; 

    /*reaction 15: E + O2 + N2 => O2- + N2 */
    eqcon[14] *= 1e+06; 

    /*reaction 16: O2- + O4+ => O2 + O2 + O2 */
    eqcon[15] *= 1e-06; 

    /*reaction 17: O2- + O4+ + O2 => O2 + O2 + O2 + O2 */
    eqcon[16] *= 1e-06; 

    /*reaction 18: O2- + O4+ + N2 => O2 + O2 + O2 + N2 */
    eqcon[17] *= 1e-06; 

    /*reaction 19: O2- + O2+ + O2 => O2 + O2 + O2 */
    /*eqcon[18] *= 1;  */

    /*reaction 20: O2- + O2+ + N2 => O2 + O2 + N2 */
    /*eqcon[19] *= 1;  */

    /*reaction 21: O2- + O2 => E + O2 + O2 */
    eqcon[20] *= 1e-06; 

    /*reaction 22: O2- + N2 => E + O2 + N2 */
    eqcon[21] *= 1e-06; 
}


/*Returns the equil constants for each reaction */
/*Given P, T, and mole fractions */
void CKEQXP(double *  P, double *  T, double *  x, double *  eqcon)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double gort[10]; /* temporary storage */

    /*compute the Gibbs free energy */
    gibbs(gort, tc);

    /*compute the equilibrium constants */
    equilibriumConstants(eqcon, gort, tT);

    /*reaction 1: E + N2 => E + E + N2+ */
    eqcon[0] *= 1e-06; 

    /*reaction 2: E + O2 => E + E + O2+ */
    eqcon[1] *= 1e-06; 

    /*reaction 3: N2+ + N2 + N2 => N4+ + N2 */
    eqcon[2] *= 1e+06; 

    /*reaction 4: N2+ + N2 + O2 => N4+ + O2 */
    eqcon[3] *= 1e+06; 

    /*reaction 5: N4+ + O2 => O2+ + N2 + N2 */
    eqcon[4] *= 1e-06; 

    /*reaction 6: N2+ + O2 => O2+ + N2 */
    /*eqcon[5] *= 1;  */

    /*reaction 7: O2+ + N2 + N2 => O2pN2 + N2 */
    eqcon[6] *= 1e+06; 

    /*reaction 8: O2pN2 + N2 => O2+ + N2 + N2 */
    eqcon[7] *= 1e-06; 

    /*reaction 9: O2pN2 + O2 => O4+ + N2 */
    /*eqcon[8] *= 1;  */

    /*reaction 10: O2+ + O2 + N2 => O4+ + N2 */
    eqcon[9] *= 1e+06; 

    /*reaction 11: O2+ + O2 + O2 => O4+ + O2 */
    eqcon[10] *= 1e+06; 

    /*reaction 12: E + O4+ => O2 + O2 */
    /*eqcon[11] *= 1;  */

    /*reaction 13: E + O2+ => O + O */
    /*eqcon[12] *= 1;  */

    /*reaction 14: E + O2 + O2 => O2- + O2 */
    eqcon[13] *= 1e+06; 

    /*reaction 15: E + O2 + N2 => O2- + N2 */
    eqcon[14] *= 1e+06; 

    /*reaction 16: O2- + O4+ => O2 + O2 + O2 */
    eqcon[15] *= 1e-06; 

    /*reaction 17: O2- + O4+ + O2 => O2 + O2 + O2 + O2 */
    eqcon[16] *= 1e-06; 

    /*reaction 18: O2- + O4+ + N2 => O2 + O2 + O2 + N2 */
    eqcon[17] *= 1e-06; 

    /*reaction 19: O2- + O2+ + O2 => O2 + O2 + O2 */
    /*eqcon[18] *= 1;  */

    /*reaction 20: O2- + O2+ + N2 => O2 + O2 + N2 */
    /*eqcon[19] *= 1;  */

    /*reaction 21: O2- + O2 => E + O2 + O2 */
    eqcon[20] *= 1e-06; 

    /*reaction 22: O2- + N2 => E + O2 + N2 */
    eqcon[21] *= 1e-06; 
}


/*Returns the equil constants for each reaction */
/*Given rho, T, and mass fractions */
void CKEQYR(double *  rho, double *  T, double *  y, double *  eqcon)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double gort[10]; /* temporary storage */

    /*compute the Gibbs free energy */
    gibbs(gort, tc);

    /*compute the equilibrium constants */
    equilibriumConstants(eqcon, gort, tT);

    /*reaction 1: E + N2 => E + E + N2+ */
    eqcon[0] *= 1e-06; 

    /*reaction 2: E + O2 => E + E + O2+ */
    eqcon[1] *= 1e-06; 

    /*reaction 3: N2+ + N2 + N2 => N4+ + N2 */
    eqcon[2] *= 1e+06; 

    /*reaction 4: N2+ + N2 + O2 => N4+ + O2 */
    eqcon[3] *= 1e+06; 

    /*reaction 5: N4+ + O2 => O2+ + N2 + N2 */
    eqcon[4] *= 1e-06; 

    /*reaction 6: N2+ + O2 => O2+ + N2 */
    /*eqcon[5] *= 1;  */

    /*reaction 7: O2+ + N2 + N2 => O2pN2 + N2 */
    eqcon[6] *= 1e+06; 

    /*reaction 8: O2pN2 + N2 => O2+ + N2 + N2 */
    eqcon[7] *= 1e-06; 

    /*reaction 9: O2pN2 + O2 => O4+ + N2 */
    /*eqcon[8] *= 1;  */

    /*reaction 10: O2+ + O2 + N2 => O4+ + N2 */
    eqcon[9] *= 1e+06; 

    /*reaction 11: O2+ + O2 + O2 => O4+ + O2 */
    eqcon[10] *= 1e+06; 

    /*reaction 12: E + O4+ => O2 + O2 */
    /*eqcon[11] *= 1;  */

    /*reaction 13: E + O2+ => O + O */
    /*eqcon[12] *= 1;  */

    /*reaction 14: E + O2 + O2 => O2- + O2 */
    eqcon[13] *= 1e+06; 

    /*reaction 15: E + O2 + N2 => O2- + N2 */
    eqcon[14] *= 1e+06; 

    /*reaction 16: O2- + O4+ => O2 + O2 + O2 */
    eqcon[15] *= 1e-06; 

    /*reaction 17: O2- + O4+ + O2 => O2 + O2 + O2 + O2 */
    eqcon[16] *= 1e-06; 

    /*reaction 18: O2- + O4+ + N2 => O2 + O2 + O2 + N2 */
    eqcon[17] *= 1e-06; 

    /*reaction 19: O2- + O2+ + O2 => O2 + O2 + O2 */
    /*eqcon[18] *= 1;  */

    /*reaction 20: O2- + O2+ + N2 => O2 + O2 + N2 */
    /*eqcon[19] *= 1;  */

    /*reaction 21: O2- + O2 => E + O2 + O2 */
    eqcon[20] *= 1e-06; 

    /*reaction 22: O2- + N2 => E + O2 + N2 */
    eqcon[21] *= 1e-06; 
}


/*Returns the equil constants for each reaction */
/*Given rho, T, and mole fractions */
void CKEQXR(double *  rho, double *  T, double *  x, double *  eqcon)
{
    double tT = *T; /*temporary temperature */
    double tc[] = { log(tT), tT, tT*tT, tT*tT*tT, tT*tT*tT*tT }; /*temperature cache */
    double gort[10]; /* temporary storage */

    /*compute the Gibbs free energy */
    gibbs(gort, tc);

    /*compute the equilibrium constants */
    equilibriumConstants(eqcon, gort, tT);

    /*reaction 1: E + N2 => E + E + N2+ */
    eqcon[0] *= 1e-06; 

    /*reaction 2: E + O2 => E + E + O2+ */
    eqcon[1] *= 1e-06; 

    /*reaction 3: N2+ + N2 + N2 => N4+ + N2 */
    eqcon[2] *= 1e+06; 

    /*reaction 4: N2+ + N2 + O2 => N4+ + O2 */
    eqcon[3] *= 1e+06; 

    /*reaction 5: N4+ + O2 => O2+ + N2 + N2 */
    eqcon[4] *= 1e-06; 

    /*reaction 6: N2+ + O2 => O2+ + N2 */
    /*eqcon[5] *= 1;  */

    /*reaction 7: O2+ + N2 + N2 => O2pN2 + N2 */
    eqcon[6] *= 1e+06; 

    /*reaction 8: O2pN2 + N2 => O2+ + N2 + N2 */
    eqcon[7] *= 1e-06; 

    /*reaction 9: O2pN2 + O2 => O4+ + N2 */
    /*eqcon[8] *= 1;  */

    /*reaction 10: O2+ + O2 + N2 => O4+ + N2 */
    eqcon[9] *= 1e+06; 

    /*reaction 11: O2+ + O2 + O2 => O4+ + O2 */
    eqcon[10] *= 1e+06; 

    /*reaction 12: E + O4+ => O2 + O2 */
    /*eqcon[11] *= 1;  */

    /*reaction 13: E + O2+ => O + O */
    /*eqcon[12] *= 1;  */

    /*reaction 14: E + O2 + O2 => O2- + O2 */
    eqcon[13] *= 1e+06; 

    /*reaction 15: E + O2 + N2 => O2- + N2 */
    eqcon[14] *= 1e+06; 

    /*reaction 16: O2- + O4+ => O2 + O2 + O2 */
    eqcon[15] *= 1e-06; 

    /*reaction 17: O2- + O4+ + O2 => O2 + O2 + O2 + O2 */
    eqcon[16] *= 1e-06; 

    /*reaction 18: O2- + O4+ + N2 => O2 + O2 + O2 + N2 */
    eqcon[17] *= 1e-06; 

    /*reaction 19: O2- + O2+ + O2 => O2 + O2 + O2 */
    /*eqcon[18] *= 1;  */

    /*reaction 20: O2- + O2+ + N2 => O2 + O2 + N2 */
    /*eqcon[19] *= 1;  */

    /*reaction 21: O2- + O2 => E + O2 + O2 */
    eqcon[20] *= 1e-06; 

    /*reaction 22: O2- + N2 => E + O2 + N2 */
    eqcon[21] *= 1e-06; 
}


/*Returns the electronic charges of the species */
void CKCHRG(int * kcharge)
{
    kcharge[0] = -1; /* E */
    kcharge[1] = 0; /* O2 */
    kcharge[2] = 0; /* N2 */
    kcharge[3] = 0; /* O */
    kcharge[4] = 1; /* O2+ */
    kcharge[5] = 1; /* N2+ */
    kcharge[6] = 1; /* O4+ */
    kcharge[7] = 1; /* N4+ */
    kcharge[8] = 1; /* O2pN2 */
    kcharge[9] = -1; /* O2- */
}

#ifdef AMREX_USE_CUDA
/*GPU version of productionRate: no more use of thermo namespace vectors */
/*compute the production rate for each species */
AMREX_GPU_HOST_DEVICE inline void  productionRate(double * wdot, double * sc, double T, double EoN)
{
    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double invT = 1.0 / tc[1];

    double qdot, q_f[22], q_r[22];
    comp_qfqr(q_f, q_r, sc, tc, invT);

    for (int i = 0; i < 10; ++i) {
        wdot[i] = 0.0;
    }

    qdot = q_f[0]-q_r[0];
    wdot[0] -= qdot;
    wdot[0] += qdot;
    wdot[0] += qdot;
    wdot[2] -= qdot;
    wdot[5] += qdot;

    qdot = q_f[1]-q_r[1];
    wdot[0] -= qdot;
    wdot[0] += qdot;
    wdot[0] += qdot;
    wdot[1] -= qdot;
    wdot[4] += qdot;

    qdot = q_f[2]-q_r[2];
    wdot[2] -= qdot;
    wdot[2] -= qdot;
    wdot[2] += qdot;
    wdot[5] -= qdot;
    wdot[7] += qdot;

    qdot = q_f[3]-q_r[3];
    wdot[1] -= qdot;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[5] -= qdot;
    wdot[7] += qdot;

    qdot = q_f[4]-q_r[4];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[2] += qdot;
    wdot[4] += qdot;
    wdot[7] -= qdot;

    qdot = q_f[5]-q_r[5];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[4] += qdot;
    wdot[5] -= qdot;

    qdot = q_f[6]-q_r[6];
    wdot[2] -= qdot;
    wdot[2] -= qdot;
    wdot[2] += qdot;
    wdot[4] -= qdot;
    wdot[8] += qdot;

    qdot = q_f[7]-q_r[7];
    wdot[2] -= qdot;
    wdot[2] += qdot;
    wdot[2] += qdot;
    wdot[4] += qdot;
    wdot[8] -= qdot;

    qdot = q_f[8]-q_r[8];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[6] += qdot;
    wdot[8] -= qdot;

    qdot = q_f[9]-q_r[9];
    wdot[1] -= qdot;
    wdot[2] -= qdot;
    wdot[2] += qdot;
    wdot[4] -= qdot;
    wdot[6] += qdot;

    qdot = q_f[10]-q_r[10];
    wdot[1] -= qdot;
    wdot[1] -= qdot;
    wdot[1] += qdot;
    wdot[4] -= qdot;
    wdot[6] += qdot;

    qdot = q_f[11]-q_r[11];
    wdot[0] -= qdot;
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[6] -= qdot;

    qdot = q_f[12]-q_r[12];
    wdot[0] -= qdot;
    wdot[3] += qdot;
    wdot[3] += qdot;
    wdot[4] -= qdot;

    qdot = q_f[13]-q_r[13];
    wdot[0] -= qdot;
    wdot[1] -= qdot;
    wdot[1] -= qdot;
    wdot[1] += qdot;
    wdot[9] += qdot;

    qdot = q_f[14]-q_r[14];
    wdot[0] -= qdot;
    wdot[1] -= qdot;
    wdot[2] -= qdot;
    wdot[2] += qdot;
    wdot[9] += qdot;

    qdot = q_f[15]-q_r[15];
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[6] -= qdot;
    wdot[9] -= qdot;

    qdot = q_f[16]-q_r[16];
    wdot[1] -= qdot;
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[6] -= qdot;
    wdot[9] -= qdot;

    qdot = q_f[17]-q_r[17];
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[2] += qdot;
    wdot[6] -= qdot;
    wdot[9] -= qdot;

    qdot = q_f[18]-q_r[18];
    wdot[1] -= qdot;
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[4] -= qdot;
    wdot[9] -= qdot;

    qdot = q_f[19]-q_r[19];
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[2] += qdot;
    wdot[4] -= qdot;
    wdot[9] -= qdot;

    qdot = q_f[20]-q_r[20];
    wdot[0] += qdot;
    wdot[1] -= qdot;
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[9] -= qdot;

    qdot = q_f[21]-q_r[21];
    wdot[0] += qdot;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[2] += qdot;
    wdot[9] -= qdot;

    return;
}

AMREX_GPU_HOST_DEVICE inline void comp_qfqr(double *  qf, double * qr, double * sc, double * tc, double invT)
{

    /*reaction 1: E + N2 => E + E + N2+ */
    qf[0] = sc[0]*sc[2];
    qr[0] = 0.0;

    /*reaction 2: E + O2 => E + E + O2+ */
    qf[1] = sc[0]*sc[1];
    qr[1] = 0.0;

    /*reaction 3: N2+ + N2 + N2 => N4+ + N2 */
    qf[2] = sc[2]*sc[2]*sc[5];
    qr[2] = 0.0;

    /*reaction 4: N2+ + N2 + O2 => N4+ + O2 */
    qf[3] = sc[1]*sc[2]*sc[5];
    qr[3] = 0.0;

    /*reaction 5: N4+ + O2 => O2+ + N2 + N2 */
    qf[4] = sc[1]*sc[7];
    qr[4] = 0.0;

    /*reaction 6: N2+ + O2 => O2+ + N2 */
    qf[5] = sc[1]*sc[5];
    qr[5] = 0.0;

    /*reaction 7: O2+ + N2 + N2 => O2pN2 + N2 */
    qf[6] = sc[2]*sc[2]*sc[4];
    qr[6] = 0.0;

    /*reaction 8: O2pN2 + N2 => O2+ + N2 + N2 */
    qf[7] = sc[2]*sc[8];
    qr[7] = 0.0;

    /*reaction 9: O2pN2 + O2 => O4+ + N2 */
    qf[8] = sc[1]*sc[8];
    qr[8] = 0.0;

    /*reaction 10: O2+ + O2 + N2 => O4+ + N2 */
    qf[9] = sc[1]*sc[2]*sc[4];
    qr[9] = 0.0;

    /*reaction 11: O2+ + O2 + O2 => O4+ + O2 */
    qf[10] = sc[1]*sc[1]*sc[4];
    qr[10] = 0.0;

    /*reaction 12: E + O4+ => O2 + O2 */
    qf[11] = sc[0]*sc[6];
    qr[11] = 0.0;

    /*reaction 13: E + O2+ => O + O */
    qf[12] = sc[0]*sc[4];
    qr[12] = 0.0;

    /*reaction 14: E + O2 + O2 => O2- + O2 */
    qf[13] = sc[0]*sc[1]*sc[1];
    qr[13] = 0.0;

    /*reaction 15: E + O2 + N2 => O2- + N2 */
    qf[14] = sc[0]*sc[1]*sc[2];
    qr[14] = 0.0;

    /*reaction 16: O2- + O4+ => O2 + O2 + O2 */
    qf[15] = sc[6]*sc[9];
    qr[15] = 0.0;

    /*reaction 17: O2- + O4+ + O2 => O2 + O2 + O2 + O2 */
    qf[16] = sc[1]*sc[6]*sc[9];
    qr[16] = 0.0;

    /*reaction 18: O2- + O4+ + N2 => O2 + O2 + O2 + N2 */
    qf[17] = sc[2]*sc[6]*sc[9];
    qr[17] = 0.0;

    /*reaction 19: O2- + O2+ + O2 => O2 + O2 + O2 */
    qf[18] = sc[1]*sc[4]*sc[9];
    qr[18] = 0.0;

    /*reaction 20: O2- + O2+ + N2 => O2 + O2 + N2 */
    qf[19] = sc[2]*sc[4]*sc[9];
    qr[19] = 0.0;

    /*reaction 21: O2- + O2 => E + O2 + O2 */
    qf[20] = sc[1]*sc[9];
    qr[20] = 0.0;

    /*reaction 22: O2- + N2 => E + O2 + N2 */
    qf[21] = sc[2]*sc[9];
    qr[21] = 0.0;

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int i = 0; i < 10; ++i) {
        mixture += sc[i];
    }

    /*compute the Gibbs free energy */
    double g_RT[10];
    gibbs(g_RT, tc);

    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 * invT;
    double refCinv = 1 / refC;

    /* Evaluate the kfs */
    double k_f, k_r, Corr;

    // (0):  E + N2 => E + E + N2+
    k_f = 1.0000000000000002e-06 * 1 
               * exp(0 * tc[0] - 0.50321666580471969 * (1) * invT);
    Corr  = 1.0;
    qf[0] *= Corr * k_f;
    qr[0] *= Corr * k_f / (exp(g_RT[0] - g_RT[0] - g_RT[0] + g_RT[2] - g_RT[5]) * refC);
    // (1):  E + O2 => E + E + O2+
    k_f = 1.0000000000000002e-06 * 1 
               * exp(0 * tc[0] - 0.50321666580471969 * (1) * invT);
    Corr  = 1.0;
    qf[1] *= Corr * k_f;
    qr[1] *= Corr * k_f / (exp(g_RT[0] - g_RT[0] - g_RT[0] + g_RT[1] - g_RT[4]) * refC);
    // (2):  N2+ + N2 + N2 => N4+ + N2
    k_f = 1.0000000000000002e-12 * 1.8132242e+19 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[2] *= Corr * k_f;
    qr[2] *= Corr * k_f / (exp(g_RT[2] + g_RT[2] - g_RT[2] + g_RT[5] - g_RT[7]) * refCinv);
    // (3):  N2+ + N2 + O2 => N4+ + O2
    k_f = 1.0000000000000002e-12 * 1.8132242e+19 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[3] *= Corr * k_f;
    qr[3] *= Corr * k_f / (exp(g_RT[1] - g_RT[1] + g_RT[2] + g_RT[5] - g_RT[7]) * refCinv);
    // (4):  N4+ + O2 => O2+ + N2 + N2
    k_f = 1.0000000000000002e-06 * 150550000000000 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[4] *= Corr * k_f;
    qr[4] *= Corr * k_f / (exp(g_RT[1] - g_RT[2] - g_RT[2] - g_RT[4] + g_RT[7]) * refC);
    // (5):  N2+ + O2 => O2+ + N2
    k_f = 1.0000000000000002e-06 * 36132000000000 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[5] *= Corr * k_f;
    qr[5] *= Corr * k_f / exp(g_RT[1] - g_RT[2] - g_RT[4] + g_RT[5]);
    // (6):  O2+ + N2 + N2 => O2pN2 + N2
    k_f = 1.0000000000000002e-12 * 3.26380356e+17 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[6] *= Corr * k_f;
    qr[6] *= Corr * k_f / (exp(g_RT[2] + g_RT[2] - g_RT[2] + g_RT[4] - g_RT[8]) * refCinv);
    // (7):  O2pN2 + N2 => O2+ + N2 + N2
    k_f = 1.0000000000000002e-06 * 258946000000000 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[7] *= Corr * k_f;
    qr[7] *= Corr * k_f / (exp(g_RT[2] - g_RT[2] - g_RT[2] - g_RT[4] + g_RT[8]) * refC);
    // (8):  O2pN2 + O2 => O4+ + N2
    k_f = 1.0000000000000002e-06 * 602200000000000 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[8] *= Corr * k_f;
    qr[8] *= Corr * k_f / exp(g_RT[1] - g_RT[2] - g_RT[6] + g_RT[8]);
    // (9):  O2+ + O2 + N2 => O4+ + N2
    k_f = 1.0000000000000002e-12 * 8.70347616e+17 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[9] *= Corr * k_f;
    qr[9] *= Corr * k_f / (exp(g_RT[1] + g_RT[2] - g_RT[2] + g_RT[4] - g_RT[6]) * refCinv);
    // (10):  O2+ + O2 + O2 => O4+ + O2
    k_f = 1.0000000000000002e-12 * 8.70347616e+17 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[10] *= Corr * k_f;
    qr[10] *= Corr * k_f / (exp(g_RT[1] + g_RT[1] - g_RT[1] + g_RT[4] - g_RT[6]) * refCinv);
    // (11):  E + O4+ => O2 + O2
    k_f = 1.0000000000000002e-06 * 1 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[11] *= Corr * k_f;
    qr[11] *= Corr * k_f / exp(g_RT[0] - g_RT[1] - g_RT[1] + g_RT[6]);
    // (12):  E + O2+ => O + O
    k_f = 1.0000000000000002e-06 * 1 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[12] *= Corr * k_f;
    qr[12] *= Corr * k_f / exp(g_RT[0] - g_RT[3] - g_RT[3] + g_RT[4]);
    // (13):  E + O2 + O2 => O2- + O2
    k_f = 1.0000000000000002e-12 * 1 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[13] *= Corr * k_f;
    qr[13] *= Corr * k_f / (exp(g_RT[0] + g_RT[1] + g_RT[1] - g_RT[1] - g_RT[9]) * refCinv);
    // (14):  E + O2 + N2 => O2- + N2
    k_f = 1.0000000000000002e-12 * 1 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[14] *= Corr * k_f;
    qr[14] *= Corr * k_f / (exp(g_RT[0] + g_RT[1] + g_RT[2] - g_RT[2] - g_RT[9]) * refCinv);
    // (15):  O2- + O4+ => O2 + O2 + O2
    k_f = 1.0000000000000002e-06 * 60220000000000000 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[15] *= Corr * k_f;
    qr[15] *= Corr * k_f / (exp(-g_RT[1] - g_RT[1] - g_RT[1] + g_RT[6] + g_RT[9]) * refC);
    // (16):  O2- + O4+ + O2 => O2 + O2 + O2 + O2
    k_f = 1.0000000000000002e-12 * 7.2528968000000003e+22 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[16] *= Corr * k_f;
    qr[16] *= Corr * k_f / (exp(g_RT[1] - g_RT[1] - g_RT[1] - g_RT[1] - g_RT[1] + g_RT[6] + g_RT[9]) * refC);
    // (17):  O2- + O4+ + N2 => O2 + O2 + O2 + N2
    k_f = 1.0000000000000002e-12 * 7.2528968000000003e+22 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[17] *= Corr * k_f;
    qr[17] *= Corr * k_f / (exp(-g_RT[1] - g_RT[1] - g_RT[1] + g_RT[2] - g_RT[2] + g_RT[6] + g_RT[9]) * refC);
    // (18):  O2- + O2+ + O2 => O2 + O2 + O2
    k_f = 1.0000000000000002e-12 * 7.2528968000000003e+22 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[18] *= Corr * k_f;
    qr[18] *= Corr * k_f / exp(g_RT[1] - g_RT[1] - g_RT[1] - g_RT[1] + g_RT[4] + g_RT[9]);
    // (19):  O2- + O2+ + N2 => O2 + O2 + N2
    k_f = 1.0000000000000002e-12 * 7.2528968000000003e+22 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[19] *= Corr * k_f;
    qr[19] *= Corr * k_f / exp(-g_RT[1] - g_RT[1] + g_RT[2] - g_RT[2] + g_RT[4] + g_RT[9]);
    // (20):  O2- + O2 => E + O2 + O2
    k_f = 1.0000000000000002e-06 * 1 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[20] *= Corr * k_f;
    qr[20] *= Corr * k_f / (exp(-g_RT[0] + g_RT[1] - g_RT[1] - g_RT[1] + g_RT[9]) * refC);
    // (21):  O2- + N2 => E + O2 + N2
    k_f = 1.0000000000000002e-06 * 1 
               * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    Corr  = 1.0;
    qf[21] *= Corr * k_f;
    qr[21] *= Corr * k_f / (exp(-g_RT[0] - g_RT[1] + g_RT[2] - g_RT[2] + g_RT[9]) * refC);


    return;
}
#endif


#ifndef AMREX_USE_CUDA
static double T_save = -1;
#ifdef _OPENMP
#pragma omp threadprivate(T_save)
#endif

static double k_f_save[22];
#ifdef _OPENMP
#pragma omp threadprivate(k_f_save)
#endif

static double Kc_save[22];
#ifdef _OPENMP
#pragma omp threadprivate(Kc_save)
#endif


/*compute the production rate for each species pointwise on CPU */
void productionRate(double *  wdot, double *  sc, double T, double EoN)
{
    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double invT = 1.0 / tc[1];
    double Te;

    if (T != T_save)
    {
        T_save = T;
        comp_k_f(tc,invT,k_f_save);
        comp_Kc(tc,invT,Kc_save);
    }

    /*calculate Te based on E/N */
    ExtrapTe(EoN, &Te);
    
    /*Calculate the electric field-dependent rate constants */
    plasmaFRates(tc, invT, k_f_save, EoN, Te);

    double qdot, q_f[22], q_r[22];
    comp_qfqr(q_f, q_r, sc, tc, invT);

    for (int i = 0; i < 10; ++i) {
        wdot[i] = 0.0;
    }

    qdot = q_f[0]-q_r[0];
    wdot[0] -= qdot;
    wdot[0] += qdot;
    wdot[0] += qdot;
    wdot[2] -= qdot;
    wdot[5] += qdot;

    qdot = q_f[1]-q_r[1];
    wdot[0] -= qdot;
    wdot[0] += qdot;
    wdot[0] += qdot;
    wdot[1] -= qdot;
    wdot[4] += qdot;

    qdot = q_f[2]-q_r[2];
    wdot[2] -= qdot;
    wdot[2] -= qdot;
    wdot[2] += qdot;
    wdot[5] -= qdot;
    wdot[7] += qdot;

    qdot = q_f[3]-q_r[3];
    wdot[1] -= qdot;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[5] -= qdot;
    wdot[7] += qdot;

    qdot = q_f[4]-q_r[4];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[2] += qdot;
    wdot[4] += qdot;
    wdot[7] -= qdot;

    qdot = q_f[5]-q_r[5];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[4] += qdot;
    wdot[5] -= qdot;

    qdot = q_f[6]-q_r[6];
    wdot[2] -= qdot;
    wdot[2] -= qdot;
    wdot[2] += qdot;
    wdot[4] -= qdot;
    wdot[8] += qdot;

    qdot = q_f[7]-q_r[7];
    wdot[2] -= qdot;
    wdot[2] += qdot;
    wdot[2] += qdot;
    wdot[4] += qdot;
    wdot[8] -= qdot;

    qdot = q_f[8]-q_r[8];
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[6] += qdot;
    wdot[8] -= qdot;

    qdot = q_f[9]-q_r[9];
    wdot[1] -= qdot;
    wdot[2] -= qdot;
    wdot[2] += qdot;
    wdot[4] -= qdot;
    wdot[6] += qdot;

    qdot = q_f[10]-q_r[10];
    wdot[1] -= qdot;
    wdot[1] -= qdot;
    wdot[1] += qdot;
    wdot[4] -= qdot;
    wdot[6] += qdot;

    qdot = q_f[11]-q_r[11];
    wdot[0] -= qdot;
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[6] -= qdot;

    qdot = q_f[12]-q_r[12];
    wdot[0] -= qdot;
    wdot[3] += qdot;
    wdot[3] += qdot;
    wdot[4] -= qdot;

    qdot = q_f[13]-q_r[13];
    wdot[0] -= qdot;
    wdot[1] -= qdot;
    wdot[1] -= qdot;
    wdot[1] += qdot;
    wdot[9] += qdot;

    qdot = q_f[14]-q_r[14];
    wdot[0] -= qdot;
    wdot[1] -= qdot;
    wdot[2] -= qdot;
    wdot[2] += qdot;
    wdot[9] += qdot;

    qdot = q_f[15]-q_r[15];
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[6] -= qdot;
    wdot[9] -= qdot;

    qdot = q_f[16]-q_r[16];
    wdot[1] -= qdot;
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[6] -= qdot;
    wdot[9] -= qdot;

    qdot = q_f[17]-q_r[17];
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[2] += qdot;
    wdot[6] -= qdot;
    wdot[9] -= qdot;

    qdot = q_f[18]-q_r[18];
    wdot[1] -= qdot;
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[4] -= qdot;
    wdot[9] -= qdot;

    qdot = q_f[19]-q_r[19];
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[2] += qdot;
    wdot[4] -= qdot;
    wdot[9] -= qdot;

    qdot = q_f[20]-q_r[20];
    wdot[0] += qdot;
    wdot[1] -= qdot;
    wdot[1] += qdot;
    wdot[1] += qdot;
    wdot[9] -= qdot;

    qdot = q_f[21]-q_r[21];
    wdot[0] += qdot;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[2] += qdot;
    wdot[9] -= qdot;

    // TODO remove after tests
    // for(int n = 0; n < 10; n++) wdot[n] = 0.0;

    // TODO: hardcoded for now, add to input file late
    // Creation of seed electron/ions (mol/m3-s)
    wdot[0] += (1.0e7/6.02214085774e23);
    wdot[1] -= (1.0e7/6.02214085774e23) * sc[1] / (sc[1] + sc[2]);
    wdot[2] -= (1.0e7/6.02214085774e23) * sc[2] / (sc[1] + sc[2]);
    wdot[4] += (1.0e7/6.02214085774e23) * sc[1] / (sc[1] + sc[2]);
    wdot[5] += (1.0e7/6.02214085774e23) * sc[2] / (sc[1] + sc[2]);

    // wdot[1] = 0.0;
    // wdot[2] = 0.0;

    // for(int n = 0; n<22; n++) printf("Rate %i = %.6e\n", n, k_f_save[n]);
    // for(int n = 0; n<10; n++) printf("Species %i production rate = %.6e (mol/m3-s)\n", wdot[n]);
    // exit(1);

    return;
}

void comp_k_f(double *  tc, double invT, double *  k_f)
{
    for (int i=0; i<22; ++i) {
        k_f[i] = prefactor_units[i] * fwd_A[i]
                    * exp(fwd_beta[i] * tc[0] - activation_units[i] * fwd_Ea[i] * invT);
    };
    return;
}

void plasmaFRates(double *  tc, double invT, double *  k_f, double EoN, double Te)
{
  // Calculates the forward rate constants (SI units - mol, m, s) for plasma reactions

  if(EoN > 1.0e-10){
    // E + N2 => E + E + N2+
    k_f[0] = 1.0e-6 * pow(10, -8.3 - 365.0/EoN) * 6.02214085774e23;

    // E + O2 => E + E + O2+   
    k_f[1] = 1.0e-6 * pow(10, -8.8 - 281.0/EoN) * 6.02214085774e23;
  }
  else{
    k_f[0] = 0.0;
    k_f[1] = 0.0;
  }

  // E + O4+ => O2 + O2
  k_f[11] = 1.0e-6 * 1.4e-6 * pow((300.0 / Te), 0.5) * 6.02214085774e23;

  // E + O2+ => O + O
  k_f[12] = 1.0e-6 * 2.0e-7 * (300.0 / Te) * 6.02214085774e23;

  // E + O2 + O2 => O2- + O2
  k_f[13] = 1.0e-12 * 1.4e-29 * (300.0/Te) * exp(-600.0/tc[1]) * exp(700.0*(Te-tc[1]) / (Te*tc[1])) * 6.02214085774e23 * 6.02214085774e23;

  // E + O2 + N2 => O2- + N2
  k_f[14] = 1.0e-12 * 1.07e-31 * pow((300.0/Te),2) * exp(-70.0/tc[1]) * exp(1500.0*(Te-tc[1]) / (Te*tc[1])) * 6.02214085774e23 * 6.02214085774e23;

  // O2- + M => E + O2 + M
  double expsum = -2.026484225560049e-05*tc[2] + 3.062335967570566e-02*tc[1] -2.470075295687302e+01;
  k_f[20] = pow(10, expsum);
  // Fitting for 300 K only
  expsum = -1.319420604234347e-09*EoN*EoN*EoN*EoN + 1.305060482084180e-06*EoN*EoN*EoN -4.818122893738965e-04*EoN*EoN + 8.343548556066513e-02*EoN -1.652720822318659e+01;
  k_f[20] += pow(10, expsum);
  k_f[20] *= 1.0e-6 * 6.02214085774e23;
  k_f[21] = k_f[20];

  return;
}

void ExtrapTe(double EoN, double * Te)
{
  
  double EN1, Te1, EN2, Te2;
  
  if(EoN <= ENData[0]){
    *Te = TeData[0] * 7736.34802879;     // Convert eV to K
  }
  else if(EoN >= ENData[Te_len-1]){
    *Te = TeData[Te_len-1] * 7736.34802879;
  }
  else{
    EN1 = ENData[0]; EN2 = ENData[1];
    Te1 = TeData[0]; Te2 = TeData[1];
    int cnt = 1;
    while(EN2 < EoN){
      cnt++;
      EN1 = EN2; Te1 = Te2;
      EN2 = ENData[cnt]; Te2 = TeData[cnt];
    }
    *Te = (Te1 + (Te2 - Te1) * ((EoN - EN1) / (EN2 - EN1))) * 7736.34802879;  // Linear extrapolation to get Te(E/N)
  }


  return;
}

void comp_Kc(double *  tc, double invT, double *  Kc)
{
    /*compute the Gibbs free energy */
    double g_RT[10];
    gibbs(g_RT, tc);

    Kc[0] = g_RT[0] - g_RT[0] - g_RT[0] + g_RT[2] - g_RT[5];
    Kc[1] = g_RT[0] - g_RT[0] - g_RT[0] + g_RT[1] - g_RT[4];
    Kc[2] = g_RT[2] + g_RT[2] - g_RT[2] + g_RT[5] - g_RT[7];
    Kc[3] = g_RT[1] - g_RT[1] + g_RT[2] + g_RT[5] - g_RT[7];
    Kc[4] = g_RT[1] - g_RT[2] - g_RT[2] - g_RT[4] + g_RT[7];
    Kc[5] = g_RT[1] - g_RT[2] - g_RT[4] + g_RT[5];
    Kc[6] = g_RT[2] + g_RT[2] - g_RT[2] + g_RT[4] - g_RT[8];
    Kc[7] = g_RT[2] - g_RT[2] - g_RT[2] - g_RT[4] + g_RT[8];
    Kc[8] = g_RT[1] - g_RT[2] - g_RT[6] + g_RT[8];
    Kc[9] = g_RT[1] + g_RT[2] - g_RT[2] + g_RT[4] - g_RT[6];
    Kc[10] = g_RT[1] + g_RT[1] - g_RT[1] + g_RT[4] - g_RT[6];
    Kc[11] = g_RT[0] - g_RT[1] - g_RT[1] + g_RT[6];
    Kc[12] = g_RT[0] - g_RT[3] - g_RT[3] + g_RT[4];
    Kc[13] = g_RT[0] + g_RT[1] + g_RT[1] - g_RT[1] - g_RT[9];
    Kc[14] = g_RT[0] + g_RT[1] + g_RT[2] - g_RT[2] - g_RT[9];
    Kc[15] = -g_RT[1] - g_RT[1] - g_RT[1] + g_RT[6] + g_RT[9];
    Kc[16] = g_RT[1] - g_RT[1] - g_RT[1] - g_RT[1] - g_RT[1] + g_RT[6] + g_RT[9];
    Kc[17] = -g_RT[1] - g_RT[1] - g_RT[1] + g_RT[2] - g_RT[2] + g_RT[6] + g_RT[9];
    Kc[18] = g_RT[1] - g_RT[1] - g_RT[1] - g_RT[1] + g_RT[4] + g_RT[9];
    Kc[19] = -g_RT[1] - g_RT[1] + g_RT[2] - g_RT[2] + g_RT[4] + g_RT[9];
    Kc[20] = -g_RT[0] + g_RT[1] - g_RT[1] - g_RT[1] + g_RT[9];
    Kc[21] = -g_RT[0] - g_RT[1] + g_RT[2] - g_RT[2] + g_RT[9];

    for (int i=0; i<22; ++i) {
        Kc[i] = exp(Kc[i]);
    };

    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 * invT;
    double refCinv = 1 / refC;

    Kc[0] *= refC;
    Kc[1] *= refC;
    Kc[2] *= refCinv;
    Kc[3] *= refCinv;
    Kc[4] *= refC;
    Kc[6] *= refCinv;
    Kc[7] *= refC;
    Kc[9] *= refCinv;
    Kc[10] *= refCinv;
    Kc[13] *= refCinv;
    Kc[14] *= refCinv;
    Kc[15] *= refC;
    Kc[16] *= refC;
    Kc[17] *= refC;
    Kc[20] *= refC;
    Kc[21] *= refC;

    return;
}

void comp_qfqr(double *  qf, double *  qr, double *  sc, double *  tc, double invT)
{

    /*reaction 1: E + N2 => E + E + N2+ */
    qf[0] = sc[0]*sc[2];
    qr[0] = 0.0;

    /*reaction 2: E + O2 => E + E + O2+ */
    qf[1] = sc[0]*sc[1];
    qr[1] = 0.0;

    /*reaction 3: N2+ + N2 + N2 => N4+ + N2 */
    qf[2] = sc[2]*sc[2]*sc[5];
    qr[2] = 0.0;

    /*reaction 4: N2+ + N2 + O2 => N4+ + O2 */
    qf[3] = sc[1]*sc[2]*sc[5];
    qr[3] = 0.0;

    /*reaction 5: N4+ + O2 => O2+ + N2 + N2 */
    qf[4] = sc[1]*sc[7];
    qr[4] = 0.0;

    /*reaction 6: N2+ + O2 => O2+ + N2 */
    qf[5] = sc[1]*sc[5];
    qr[5] = 0.0;

    /*reaction 7: O2+ + N2 + N2 => O2pN2 + N2 */
    qf[6] = sc[2]*sc[2]*sc[4];
    qr[6] = 0.0;

    /*reaction 8: O2pN2 + N2 => O2+ + N2 + N2 */
    qf[7] = sc[2]*sc[8];
    qr[7] = 0.0;

    /*reaction 9: O2pN2 + O2 => O4+ + N2 */
    qf[8] = sc[1]*sc[8];
    qr[8] = 0.0;

    /*reaction 10: O2+ + O2 + N2 => O4+ + N2 */
    qf[9] = sc[1]*sc[2]*sc[4];
    qr[9] = 0.0;

    /*reaction 11: O2+ + O2 + O2 => O4+ + O2 */
    qf[10] = sc[1]*sc[1]*sc[4];
    qr[10] = 0.0;

    /*reaction 12: E + O4+ => O2 + O2 */
    qf[11] = sc[0]*sc[6];
    qr[11] = 0.0;

    /*reaction 13: E + O2+ => O + O */
    qf[12] = sc[0]*sc[4];
    qr[12] = 0.0;

    /*reaction 14: E + O2 + O2 => O2- + O2 */
    qf[13] = sc[0]*sc[1]*sc[1];
    qr[13] = 0.0;

    /*reaction 15: E + O2 + N2 => O2- + N2 */
    qf[14] = sc[0]*sc[1]*sc[2];
    qr[14] = 0.0;

    /*reaction 16: O2- + O4+ => O2 + O2 + O2 */
    qf[15] = sc[6]*sc[9];
    qr[15] = 0.0;

    /*reaction 17: O2- + O4+ + O2 => O2 + O2 + O2 + O2 */
    qf[16] = sc[1]*sc[6]*sc[9];
    qr[16] = 0.0;

    /*reaction 18: O2- + O4+ + N2 => O2 + O2 + O2 + N2 */
    qf[17] = sc[2]*sc[6]*sc[9];
    qr[17] = 0.0;

    /*reaction 19: O2- + O2+ + O2 => O2 + O2 + O2 */
    qf[18] = sc[1]*sc[4]*sc[9];
    qr[18] = 0.0;

    /*reaction 20: O2- + O2+ + N2 => O2 + O2 + N2 */
    qf[19] = sc[2]*sc[4]*sc[9];
    qr[19] = 0.0;

    /*reaction 21: O2- + O2 => E + O2 + O2 */
    qf[20] = sc[1]*sc[9];
    qr[20] = 0.0;

    /*reaction 22: O2- + N2 => E + O2 + N2 */
    qf[21] = sc[2]*sc[9];
    qr[21] = 0.0;

    double T = tc[1];

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int i = 0; i < 10; ++i) {
        mixture += sc[i];
    }

    double Corr[22];
    for (int i = 0; i < 22; ++i) {
        Corr[i] = 1.0;
    }

    for (int i=0; i<22; i++)
    {
        qf[i] *= Corr[i] * k_f_save[i];
        qr[i] *= Corr[i] * k_f_save[i] / Kc_save[i];
    }

    return;
}
#endif


#ifndef AMREX_USE_CUDA
/*compute the production rate for each species */
void vproductionRate(int npt, double *  wdot, double *  sc, double *  T)
{
    double k_f_s[22*npt], Kc_s[22*npt], mixture[npt], g_RT[10*npt];
    double tc[5*npt], invT[npt];

    for (int i=0; i<npt; i++) {
        tc[0*npt+i] = log(T[i]);
        tc[1*npt+i] = T[i];
        tc[2*npt+i] = T[i]*T[i];
        tc[3*npt+i] = T[i]*T[i]*T[i];
        tc[4*npt+i] = T[i]*T[i]*T[i]*T[i];
        invT[i] = 1.0 / T[i];
    }

    for (int i=0; i<npt; i++) {
        mixture[i] = 0.0;
    }

    for (int n=0; n<10; n++) {
        for (int i=0; i<npt; i++) {
            mixture[i] += sc[n*npt+i];
            wdot[n*npt+i] = 0.0;
        }
    }

    vcomp_k_f(npt, k_f_s, tc, invT);

    vcomp_gibbs(npt, g_RT, tc);

    vcomp_Kc(npt, Kc_s, g_RT, invT);

    vcomp_wdot(npt, wdot, mixture, sc, k_f_s, Kc_s, tc, invT, T);
}

void vcomp_k_f(int npt, double *  k_f_s, double *  tc, double *  invT)
{
    for (int i=0; i<npt; i++) {
        k_f_s[0*npt+i] = prefactor_units[0] * fwd_A[0] * exp(fwd_beta[0] * tc[i] - activation_units[0] * fwd_Ea[0] * invT[i]);
        k_f_s[1*npt+i] = prefactor_units[1] * fwd_A[1] * exp(fwd_beta[1] * tc[i] - activation_units[1] * fwd_Ea[1] * invT[i]);
        k_f_s[2*npt+i] = prefactor_units[2] * fwd_A[2] * exp(fwd_beta[2] * tc[i] - activation_units[2] * fwd_Ea[2] * invT[i]);
        k_f_s[3*npt+i] = prefactor_units[3] * fwd_A[3] * exp(fwd_beta[3] * tc[i] - activation_units[3] * fwd_Ea[3] * invT[i]);
        k_f_s[4*npt+i] = prefactor_units[4] * fwd_A[4] * exp(fwd_beta[4] * tc[i] - activation_units[4] * fwd_Ea[4] * invT[i]);
        k_f_s[5*npt+i] = prefactor_units[5] * fwd_A[5] * exp(fwd_beta[5] * tc[i] - activation_units[5] * fwd_Ea[5] * invT[i]);
        k_f_s[6*npt+i] = prefactor_units[6] * fwd_A[6] * exp(fwd_beta[6] * tc[i] - activation_units[6] * fwd_Ea[6] * invT[i]);
        k_f_s[7*npt+i] = prefactor_units[7] * fwd_A[7] * exp(fwd_beta[7] * tc[i] - activation_units[7] * fwd_Ea[7] * invT[i]);
        k_f_s[8*npt+i] = prefactor_units[8] * fwd_A[8] * exp(fwd_beta[8] * tc[i] - activation_units[8] * fwd_Ea[8] * invT[i]);
        k_f_s[9*npt+i] = prefactor_units[9] * fwd_A[9] * exp(fwd_beta[9] * tc[i] - activation_units[9] * fwd_Ea[9] * invT[i]);
        k_f_s[10*npt+i] = prefactor_units[10] * fwd_A[10] * exp(fwd_beta[10] * tc[i] - activation_units[10] * fwd_Ea[10] * invT[i]);
        k_f_s[11*npt+i] = prefactor_units[11] * fwd_A[11] * exp(fwd_beta[11] * tc[i] - activation_units[11] * fwd_Ea[11] * invT[i]);
        k_f_s[12*npt+i] = prefactor_units[12] * fwd_A[12] * exp(fwd_beta[12] * tc[i] - activation_units[12] * fwd_Ea[12] * invT[i]);
        k_f_s[13*npt+i] = prefactor_units[13] * fwd_A[13] * exp(fwd_beta[13] * tc[i] - activation_units[13] * fwd_Ea[13] * invT[i]);
        k_f_s[14*npt+i] = prefactor_units[14] * fwd_A[14] * exp(fwd_beta[14] * tc[i] - activation_units[14] * fwd_Ea[14] * invT[i]);
        k_f_s[15*npt+i] = prefactor_units[15] * fwd_A[15] * exp(fwd_beta[15] * tc[i] - activation_units[15] * fwd_Ea[15] * invT[i]);
        k_f_s[16*npt+i] = prefactor_units[16] * fwd_A[16] * exp(fwd_beta[16] * tc[i] - activation_units[16] * fwd_Ea[16] * invT[i]);
        k_f_s[17*npt+i] = prefactor_units[17] * fwd_A[17] * exp(fwd_beta[17] * tc[i] - activation_units[17] * fwd_Ea[17] * invT[i]);
        k_f_s[18*npt+i] = prefactor_units[18] * fwd_A[18] * exp(fwd_beta[18] * tc[i] - activation_units[18] * fwd_Ea[18] * invT[i]);
        k_f_s[19*npt+i] = prefactor_units[19] * fwd_A[19] * exp(fwd_beta[19] * tc[i] - activation_units[19] * fwd_Ea[19] * invT[i]);
        k_f_s[20*npt+i] = prefactor_units[20] * fwd_A[20] * exp(fwd_beta[20] * tc[i] - activation_units[20] * fwd_Ea[20] * invT[i]);
        k_f_s[21*npt+i] = prefactor_units[21] * fwd_A[21] * exp(fwd_beta[21] * tc[i] - activation_units[21] * fwd_Ea[21] * invT[i]);
    }
}

void vcomp_gibbs(int npt, double *  g_RT, double *  tc)
{
    /*compute the Gibbs free energy */
    for (int i=0; i<npt; i++) {
        double tg[5], g[10];
        tg[0] = tc[0*npt+i];
        tg[1] = tc[1*npt+i];
        tg[2] = tc[2*npt+i];
        tg[3] = tc[3*npt+i];
        tg[4] = tc[4*npt+i];

        gibbs(g, tg);

        g_RT[0*npt+i] = g[0];
        g_RT[1*npt+i] = g[1];
        g_RT[2*npt+i] = g[2];
        g_RT[3*npt+i] = g[3];
        g_RT[4*npt+i] = g[4];
        g_RT[5*npt+i] = g[5];
        g_RT[6*npt+i] = g[6];
        g_RT[7*npt+i] = g[7];
        g_RT[8*npt+i] = g[8];
        g_RT[9*npt+i] = g[9];
    }
}

void vcomp_Kc(int npt, double *  Kc_s, double *  g_RT, double *  invT)
{
    for (int i=0; i<npt; i++) {
        /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
        double refC = (101325. / 8.31451) * invT[i];
        double refCinv = 1.0 / refC;

        Kc_s[0*npt+i] = refC * exp((g_RT[0*npt+i] + g_RT[2*npt+i]) - (g_RT[0*npt+i] + g_RT[0*npt+i] + g_RT[5*npt+i]));
        Kc_s[1*npt+i] = refC * exp((g_RT[0*npt+i] + g_RT[1*npt+i]) - (g_RT[0*npt+i] + g_RT[0*npt+i] + g_RT[4*npt+i]));
        Kc_s[2*npt+i] = refCinv * exp((g_RT[2*npt+i] + g_RT[2*npt+i] + g_RT[5*npt+i]) - (g_RT[2*npt+i] + g_RT[7*npt+i]));
        Kc_s[3*npt+i] = refCinv * exp((g_RT[1*npt+i] + g_RT[2*npt+i] + g_RT[5*npt+i]) - (g_RT[1*npt+i] + g_RT[7*npt+i]));
        Kc_s[4*npt+i] = refC * exp((g_RT[1*npt+i] + g_RT[7*npt+i]) - (g_RT[2*npt+i] + g_RT[2*npt+i] + g_RT[4*npt+i]));
        Kc_s[5*npt+i] = exp((g_RT[1*npt+i] + g_RT[5*npt+i]) - (g_RT[2*npt+i] + g_RT[4*npt+i]));
        Kc_s[6*npt+i] = refCinv * exp((g_RT[2*npt+i] + g_RT[2*npt+i] + g_RT[4*npt+i]) - (g_RT[2*npt+i] + g_RT[8*npt+i]));
        Kc_s[7*npt+i] = refC * exp((g_RT[2*npt+i] + g_RT[8*npt+i]) - (g_RT[2*npt+i] + g_RT[2*npt+i] + g_RT[4*npt+i]));
        Kc_s[8*npt+i] = exp((g_RT[1*npt+i] + g_RT[8*npt+i]) - (g_RT[2*npt+i] + g_RT[6*npt+i]));
        Kc_s[9*npt+i] = refCinv * exp((g_RT[1*npt+i] + g_RT[2*npt+i] + g_RT[4*npt+i]) - (g_RT[2*npt+i] + g_RT[6*npt+i]));
        Kc_s[10*npt+i] = refCinv * exp((g_RT[1*npt+i] + g_RT[1*npt+i] + g_RT[4*npt+i]) - (g_RT[1*npt+i] + g_RT[6*npt+i]));
        Kc_s[11*npt+i] = exp((g_RT[0*npt+i] + g_RT[6*npt+i]) - (g_RT[1*npt+i] + g_RT[1*npt+i]));
        Kc_s[12*npt+i] = exp((g_RT[0*npt+i] + g_RT[4*npt+i]) - (g_RT[3*npt+i] + g_RT[3*npt+i]));
        Kc_s[13*npt+i] = refCinv * exp((g_RT[0*npt+i] + g_RT[1*npt+i] + g_RT[1*npt+i]) - (g_RT[1*npt+i] + g_RT[9*npt+i]));
        Kc_s[14*npt+i] = refCinv * exp((g_RT[0*npt+i] + g_RT[1*npt+i] + g_RT[2*npt+i]) - (g_RT[2*npt+i] + g_RT[9*npt+i]));
        Kc_s[15*npt+i] = refC * exp((g_RT[6*npt+i] + g_RT[9*npt+i]) - (g_RT[1*npt+i] + g_RT[1*npt+i] + g_RT[1*npt+i]));
        Kc_s[16*npt+i] = refC * exp((g_RT[1*npt+i] + g_RT[6*npt+i] + g_RT[9*npt+i]) - (g_RT[1*npt+i] + g_RT[1*npt+i] + g_RT[1*npt+i] + g_RT[1*npt+i]));
        Kc_s[17*npt+i] = refC * exp((g_RT[2*npt+i] + g_RT[6*npt+i] + g_RT[9*npt+i]) - (g_RT[1*npt+i] + g_RT[1*npt+i] + g_RT[1*npt+i] + g_RT[2*npt+i]));
        Kc_s[18*npt+i] = exp((g_RT[1*npt+i] + g_RT[4*npt+i] + g_RT[9*npt+i]) - (g_RT[1*npt+i] + g_RT[1*npt+i] + g_RT[1*npt+i]));
        Kc_s[19*npt+i] = exp((g_RT[2*npt+i] + g_RT[4*npt+i] + g_RT[9*npt+i]) - (g_RT[1*npt+i] + g_RT[1*npt+i] + g_RT[2*npt+i]));
        Kc_s[20*npt+i] = refC * exp((g_RT[1*npt+i] + g_RT[9*npt+i]) - (g_RT[0*npt+i] + g_RT[1*npt+i] + g_RT[1*npt+i]));
        Kc_s[21*npt+i] = refC * exp((g_RT[2*npt+i] + g_RT[9*npt+i]) - (g_RT[0*npt+i] + g_RT[1*npt+i] + g_RT[2*npt+i]));
    }
}

void vcomp_wdot(int npt, double *  wdot, double *  mixture, double *  sc,
		double *  k_f_s, double *  Kc_s,
		double *  tc, double *  invT, double *  T)
{
    for (int i=0; i<npt; i++) {
        double qdot, q_f, q_r, phi_f, phi_r, k_f, k_r, Kc;
        double alpha;

        /*reaction 1: E + N2 => E + E + N2+ */
        phi_f = sc[0*npt+i]*sc[2*npt+i];
        k_f = k_f_s[0*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[0*npt+i] -= qdot;
        wdot[0*npt+i] += qdot;
        wdot[0*npt+i] += qdot;
        wdot[2*npt+i] -= qdot;
        wdot[5*npt+i] += qdot;

        /*reaction 2: E + O2 => E + E + O2+ */
        phi_f = sc[0*npt+i]*sc[1*npt+i];
        k_f = k_f_s[1*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[0*npt+i] -= qdot;
        wdot[0*npt+i] += qdot;
        wdot[0*npt+i] += qdot;
        wdot[1*npt+i] -= qdot;
        wdot[4*npt+i] += qdot;

        /*reaction 3: N2+ + N2 + N2 => N4+ + N2 */
        phi_f = sc[2*npt+i]*sc[2*npt+i]*sc[5*npt+i];
        k_f = k_f_s[2*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[2*npt+i] -= qdot;
        wdot[2*npt+i] -= qdot;
        wdot[2*npt+i] += qdot;
        wdot[5*npt+i] -= qdot;
        wdot[7*npt+i] += qdot;

        /*reaction 4: N2+ + N2 + O2 => N4+ + O2 */
        phi_f = sc[1*npt+i]*sc[2*npt+i]*sc[5*npt+i];
        k_f = k_f_s[3*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[1*npt+i] -= qdot;
        wdot[1*npt+i] += qdot;
        wdot[2*npt+i] -= qdot;
        wdot[5*npt+i] -= qdot;
        wdot[7*npt+i] += qdot;

        /*reaction 5: N4+ + O2 => O2+ + N2 + N2 */
        phi_f = sc[1*npt+i]*sc[7*npt+i];
        k_f = k_f_s[4*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[1*npt+i] -= qdot;
        wdot[2*npt+i] += qdot;
        wdot[2*npt+i] += qdot;
        wdot[4*npt+i] += qdot;
        wdot[7*npt+i] -= qdot;

        /*reaction 6: N2+ + O2 => O2+ + N2 */
        phi_f = sc[1*npt+i]*sc[5*npt+i];
        k_f = k_f_s[5*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[1*npt+i] -= qdot;
        wdot[2*npt+i] += qdot;
        wdot[4*npt+i] += qdot;
        wdot[5*npt+i] -= qdot;

        /*reaction 7: O2+ + N2 + N2 => O2pN2 + N2 */
        phi_f = sc[2*npt+i]*sc[2*npt+i]*sc[4*npt+i];
        k_f = k_f_s[6*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[2*npt+i] -= qdot;
        wdot[2*npt+i] -= qdot;
        wdot[2*npt+i] += qdot;
        wdot[4*npt+i] -= qdot;
        wdot[8*npt+i] += qdot;

        /*reaction 8: O2pN2 + N2 => O2+ + N2 + N2 */
        phi_f = sc[2*npt+i]*sc[8*npt+i];
        k_f = k_f_s[7*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[2*npt+i] -= qdot;
        wdot[2*npt+i] += qdot;
        wdot[2*npt+i] += qdot;
        wdot[4*npt+i] += qdot;
        wdot[8*npt+i] -= qdot;

        /*reaction 9: O2pN2 + O2 => O4+ + N2 */
        phi_f = sc[1*npt+i]*sc[8*npt+i];
        k_f = k_f_s[8*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[1*npt+i] -= qdot;
        wdot[2*npt+i] += qdot;
        wdot[6*npt+i] += qdot;
        wdot[8*npt+i] -= qdot;

        /*reaction 10: O2+ + O2 + N2 => O4+ + N2 */
        phi_f = sc[1*npt+i]*sc[2*npt+i]*sc[4*npt+i];
        k_f = k_f_s[9*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[1*npt+i] -= qdot;
        wdot[2*npt+i] -= qdot;
        wdot[2*npt+i] += qdot;
        wdot[4*npt+i] -= qdot;
        wdot[6*npt+i] += qdot;

        /*reaction 11: O2+ + O2 + O2 => O4+ + O2 */
        phi_f = sc[1*npt+i]*sc[1*npt+i]*sc[4*npt+i];
        k_f = k_f_s[10*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[1*npt+i] -= qdot;
        wdot[1*npt+i] -= qdot;
        wdot[1*npt+i] += qdot;
        wdot[4*npt+i] -= qdot;
        wdot[6*npt+i] += qdot;

        /*reaction 12: E + O4+ => O2 + O2 */
        phi_f = sc[0*npt+i]*sc[6*npt+i];
        k_f = k_f_s[11*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[0*npt+i] -= qdot;
        wdot[1*npt+i] += qdot;
        wdot[1*npt+i] += qdot;
        wdot[6*npt+i] -= qdot;

        /*reaction 13: E + O2+ => O + O */
        phi_f = sc[0*npt+i]*sc[4*npt+i];
        k_f = k_f_s[12*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[0*npt+i] -= qdot;
        wdot[3*npt+i] += qdot;
        wdot[3*npt+i] += qdot;
        wdot[4*npt+i] -= qdot;

        /*reaction 14: E + O2 + O2 => O2- + O2 */
        phi_f = sc[0*npt+i]*sc[1*npt+i]*sc[1*npt+i];
        k_f = k_f_s[13*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[0*npt+i] -= qdot;
        wdot[1*npt+i] -= qdot;
        wdot[1*npt+i] -= qdot;
        wdot[1*npt+i] += qdot;
        wdot[9*npt+i] += qdot;

        /*reaction 15: E + O2 + N2 => O2- + N2 */
        phi_f = sc[0*npt+i]*sc[1*npt+i]*sc[2*npt+i];
        k_f = k_f_s[14*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[0*npt+i] -= qdot;
        wdot[1*npt+i] -= qdot;
        wdot[2*npt+i] -= qdot;
        wdot[2*npt+i] += qdot;
        wdot[9*npt+i] += qdot;

        /*reaction 16: O2- + O4+ => O2 + O2 + O2 */
        phi_f = sc[6*npt+i]*sc[9*npt+i];
        k_f = k_f_s[15*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[1*npt+i] += qdot;
        wdot[1*npt+i] += qdot;
        wdot[1*npt+i] += qdot;
        wdot[6*npt+i] -= qdot;
        wdot[9*npt+i] -= qdot;

        /*reaction 17: O2- + O4+ + O2 => O2 + O2 + O2 + O2 */
        phi_f = sc[1*npt+i]*sc[6*npt+i]*sc[9*npt+i];
        k_f = k_f_s[16*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[1*npt+i] -= qdot;
        wdot[1*npt+i] += qdot;
        wdot[1*npt+i] += qdot;
        wdot[1*npt+i] += qdot;
        wdot[1*npt+i] += qdot;
        wdot[6*npt+i] -= qdot;
        wdot[9*npt+i] -= qdot;

        /*reaction 18: O2- + O4+ + N2 => O2 + O2 + O2 + N2 */
        phi_f = sc[2*npt+i]*sc[6*npt+i]*sc[9*npt+i];
        k_f = k_f_s[17*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[1*npt+i] += qdot;
        wdot[1*npt+i] += qdot;
        wdot[1*npt+i] += qdot;
        wdot[2*npt+i] -= qdot;
        wdot[2*npt+i] += qdot;
        wdot[6*npt+i] -= qdot;
        wdot[9*npt+i] -= qdot;

        /*reaction 19: O2- + O2+ + O2 => O2 + O2 + O2 */
        phi_f = sc[1*npt+i]*sc[4*npt+i]*sc[9*npt+i];
        k_f = k_f_s[18*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[1*npt+i] -= qdot;
        wdot[1*npt+i] += qdot;
        wdot[1*npt+i] += qdot;
        wdot[1*npt+i] += qdot;
        wdot[4*npt+i] -= qdot;
        wdot[9*npt+i] -= qdot;

        /*reaction 20: O2- + O2+ + N2 => O2 + O2 + N2 */
        phi_f = sc[2*npt+i]*sc[4*npt+i]*sc[9*npt+i];
        k_f = k_f_s[19*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[1*npt+i] += qdot;
        wdot[1*npt+i] += qdot;
        wdot[2*npt+i] -= qdot;
        wdot[2*npt+i] += qdot;
        wdot[4*npt+i] -= qdot;
        wdot[9*npt+i] -= qdot;

        /*reaction 21: O2- + O2 => E + O2 + O2 */
        phi_f = sc[1*npt+i]*sc[9*npt+i];
        k_f = k_f_s[20*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[0*npt+i] += qdot;
        wdot[1*npt+i] -= qdot;
        wdot[1*npt+i] += qdot;
        wdot[1*npt+i] += qdot;
        wdot[9*npt+i] -= qdot;

        /*reaction 22: O2- + N2 => E + O2 + N2 */
        phi_f = sc[2*npt+i]*sc[9*npt+i];
        k_f = k_f_s[21*npt+i];
        q_f = phi_f * k_f;
        q_r = 0.0;
        qdot = q_f - q_r;
        wdot[0*npt+i] += qdot;
        wdot[1*npt+i] += qdot;
        wdot[2*npt+i] -= qdot;
        wdot[2*npt+i] += qdot;
        wdot[9*npt+i] -= qdot;
    }
}
#endif

/*compute an approx to the reaction Jacobian (for preconditioning) */
AMREX_GPU_HOST_DEVICE void DWDOT_SIMPLIFIED(double *  J, double *  sc, double *  Tp, int * HP)
{
    double c[10];

    for (int k=0; k<10; k++) {
        c[k] = 1.e6 * sc[k];
    }

    aJacobian_precond(J, c, *Tp, *HP);

    /* dwdot[k]/dT */
    /* dTdot/d[X] */
    for (int k=0; k<10; k++) {
        J[110+k] *= 1.e-6;
        J[k*11+10] *= 1.e6;
    }

    return;
}

/*compute the reaction Jacobian */
AMREX_GPU_HOST_DEVICE void DWDOT(double *  J, double *  sc, double *  Tp, int * consP)
{
    double c[10];

    for (int k=0; k<10; k++) {
        c[k] = 1.e6 * sc[k];
    }

    aJacobian(J, c, *Tp, *consP);

    /* dwdot[k]/dT */
    /* dTdot/d[X] */
    for (int k=0; k<10; k++) {
        J[110+k] *= 1.e-6;
        J[k*11+10] *= 1.e6;
    }

    return;
}

/*compute the sparsity pattern of the chemistry Jacobian */
AMREX_GPU_HOST_DEVICE void SPARSITY_INFO( int * nJdata, int * consP, int NCELLS)
{
    double c[10];
    double J[121];

    for (int k=0; k<10; k++) {
        c[k] = 1.0/ 10.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<11; k++) {
        for (int l=0; l<11; l++) {
            if(J[ 11 * k + l] != 0.0){
                nJdata_tmp = nJdata_tmp + 1;
            }
        }
    }

    *nJdata = NCELLS * nJdata_tmp;

    return;
}



/*compute the sparsity pattern of the system Jacobian */
AMREX_GPU_HOST_DEVICE void SPARSITY_INFO_SYST( int * nJdata, int * consP, int NCELLS)
{
    double c[10];
    double J[121];

    for (int k=0; k<10; k++) {
        c[k] = 1.0/ 10.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<11; k++) {
        for (int l=0; l<11; l++) {
            if(k == l){
                nJdata_tmp = nJdata_tmp + 1;
            } else {
                if(J[ 11 * k + l] != 0.0){
                    nJdata_tmp = nJdata_tmp + 1;
                }
            }
        }
    }

    *nJdata = NCELLS * nJdata_tmp;

    return;
}



/*compute the sparsity pattern of the simplified (for preconditioning) system Jacobian */
AMREX_GPU_HOST_DEVICE void SPARSITY_INFO_SYST_SIMPLIFIED( int * nJdata, int * consP)
{
    double c[10];
    double J[121];

    for (int k=0; k<10; k++) {
        c[k] = 1.0/ 10.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    int nJdata_tmp = 0;
    for (int k=0; k<11; k++) {
        for (int l=0; l<11; l++) {
            if(k == l){
                nJdata_tmp = nJdata_tmp + 1;
            } else {
                if(J[ 11 * k + l] != 0.0){
                    nJdata_tmp = nJdata_tmp + 1;
                }
            }
        }
    }

    nJdata[0] = nJdata_tmp;

    return;
}


/*compute the sparsity pattern of the chemistry Jacobian in CSC format -- base 0 */
AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_CSC(int *  rowVals, int *  colPtrs, int * consP, int NCELLS)
{
    double c[10];
    double J[121];
    int offset_row;
    int offset_col;

    for (int k=0; k<10; k++) {
        c[k] = 1.0/ 10.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    colPtrs[0] = 0;
    int nJdata_tmp = 0;
    for (int nc=0; nc<NCELLS; nc++) {
        offset_row = nc * 11;
        offset_col = nc * 11;
        for (int k=0; k<11; k++) {
            for (int l=0; l<11; l++) {
                if(J[11*k + l] != 0.0) {
                    rowVals[nJdata_tmp] = l + offset_row; 
                    nJdata_tmp = nJdata_tmp + 1; 
                }
            }
            colPtrs[offset_col + (k + 1)] = nJdata_tmp;
        }
    }

    return;
}

/*compute the sparsity pattern of the chemistry Jacobian in CSR format -- base 0 */
AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_CSR(int * colVals, int * rowPtrs, int * consP, int NCELLS, int base)
{
    double c[10];
    double J[121];
    int offset;

    for (int k=0; k<10; k++) {
        c[k] = 1.0/ 10.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtrs[0] = 1;
        int nJdata_tmp = 1;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 11;
            for (int l=0; l<11; l++) {
                for (int k=0; k<11; k++) {
                    if(J[11*k + l] != 0.0) {
                        colVals[nJdata_tmp-1] = k+1 + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    }
                }
                rowPtrs[offset + (l + 1)] = nJdata_tmp;
            }
        }
    } else {
        rowPtrs[0] = 0;
        int nJdata_tmp = 0;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 11;
            for (int l=0; l<11; l++) {
                for (int k=0; k<11; k++) {
                    if(J[11*k + l] != 0.0) {
                        colVals[nJdata_tmp] = k + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    }
                }
                rowPtrs[offset + (l + 1)] = nJdata_tmp;
            }
        }
    }

    return;
}

/*compute the sparsity pattern of the system Jacobian */
/*CSR format BASE is user choice */
AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_SYST_CSR(int * colVals, int * rowPtr, int * consP, int NCELLS, int base)
{
    double c[10];
    double J[121];
    int offset;

    for (int k=0; k<10; k++) {
        c[k] = 1.0/ 10.000000 ;
    }

    aJacobian(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtr[0] = 1;
        int nJdata_tmp = 1;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 11;
            for (int l=0; l<11; l++) {
                for (int k=0; k<11; k++) {
                    if (k == l) {
                        colVals[nJdata_tmp-1] = l+1 + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    } else {
                        if(J[11*k + l] != 0.0) {
                            colVals[nJdata_tmp-1] = k+1 + offset; 
                            nJdata_tmp = nJdata_tmp + 1; 
                        }
                    }
                }
                rowPtr[offset + (l + 1)] = nJdata_tmp;
            }
        }
    } else {
        rowPtr[0] = 0;
        int nJdata_tmp = 0;
        for (int nc=0; nc<NCELLS; nc++) {
            offset = nc * 11;
            for (int l=0; l<11; l++) {
                for (int k=0; k<11; k++) {
                    if (k == l) {
                        colVals[nJdata_tmp] = l + offset; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    } else {
                        if(J[11*k + l] != 0.0) {
                            colVals[nJdata_tmp] = k + offset; 
                            nJdata_tmp = nJdata_tmp + 1; 
                        }
                    }
                }
                rowPtr[offset + (l + 1)] = nJdata_tmp;
            }
        }
    }

    return;
}

/*compute the sparsity pattern of the simplified (for precond) system Jacobian on CPU */
/*BASE 0 */
AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSC(int * rowVals, int * colPtrs, int * indx, int * consP)
{
    double c[10];
    double J[121];

    for (int k=0; k<10; k++) {
        c[k] = 1.0/ 10.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    colPtrs[0] = 0;
    int nJdata_tmp = 0;
    for (int k=0; k<11; k++) {
        for (int l=0; l<11; l++) {
            if (k == l) {
                rowVals[nJdata_tmp] = l; 
                indx[nJdata_tmp] = 11*k + l;
                nJdata_tmp = nJdata_tmp + 1; 
            } else {
                if(J[11*k + l] != 0.0) {
                    rowVals[nJdata_tmp] = l; 
                    indx[nJdata_tmp] = 11*k + l;
                    nJdata_tmp = nJdata_tmp + 1; 
                }
            }
        }
        colPtrs[k+1] = nJdata_tmp;
    }

    return;
}

/*compute the sparsity pattern of the simplified (for precond) system Jacobian */
/*CSR format BASE is under choice */
AMREX_GPU_HOST_DEVICE void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(int * colVals, int * rowPtr, int * consP, int base)
{
    double c[10];
    double J[121];

    for (int k=0; k<10; k++) {
        c[k] = 1.0/ 10.000000 ;
    }

    aJacobian_precond(J, c, 1500.0, *consP);

    if (base == 1) {
        rowPtr[0] = 1;
        int nJdata_tmp = 1;
        for (int l=0; l<11; l++) {
            for (int k=0; k<11; k++) {
                if (k == l) {
                    colVals[nJdata_tmp-1] = l+1; 
                    nJdata_tmp = nJdata_tmp + 1; 
                } else {
                    if(J[11*k + l] != 0.0) {
                        colVals[nJdata_tmp-1] = k+1; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    }
                }
            }
            rowPtr[l+1] = nJdata_tmp;
        }
    } else {
        rowPtr[0] = 0;
        int nJdata_tmp = 0;
        for (int l=0; l<11; l++) {
            for (int k=0; k<11; k++) {
                if (k == l) {
                    colVals[nJdata_tmp] = l; 
                    nJdata_tmp = nJdata_tmp + 1; 
                } else {
                    if(J[11*k + l] != 0.0) {
                        colVals[nJdata_tmp] = k; 
                        nJdata_tmp = nJdata_tmp + 1; 
                    }
                }
            }
            rowPtr[l+1] = nJdata_tmp;
        }
    }

    return;
}


#ifdef AMREX_USE_CUDA
/*compute the reaction Jacobian on GPU */
AMREX_GPU_HOST_DEVICE
void aJacobian(double * J, double * sc, double T, int consP)
{


    for (int i=0; i<121; i++) {
        J[i] = 0.0;
    }

    double wdot[10];
    for (int k=0; k<10; k++) {
        wdot[k] = 0.0;
    }

    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double invT = 1.0 / tc[1];
    double invT2 = invT * invT;

    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 / T;
    double refCinv = 1.0 / refC;

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int k = 0; k < 10; ++k) {
        mixture += sc[k];
    }

    /*compute the Gibbs free energy */
    double g_RT[10];
    gibbs(g_RT, tc);

    /*compute the species enthalpy */
    double h_RT[10];
    speciesEnthalpy(h_RT, tc);

    double phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
    double dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
    double dqdci, dcdc_fac, dqdc[10];
    double Pr, fPr, F, k_0, logPr;
    double logFcent, troe_c, troe_n, troePr_den, troePr, troe;
    double Fcent1, Fcent2, Fcent3, Fcent;
    double dlogFdc, dlogFdn, dlogFdcn_fac;
    double dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
    const double ln10 = log(10.0);
    const double log10e = 1.0/log(10.0);
    /*reaction 1: E + N2 => E + E + N2+ */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[0]*sc[2];
    k_f = 1.0000000000000002e-06 * 1
                * exp(0 * tc[0] - 0.50321666580471969 * (1) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  1  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] += q; /* E */
    wdot[2] -= q; /* N2 */
    wdot[5] += q; /* N2+ */
    /* d()/d[E] */
    dqdci =  + k_f*sc[2];
    J[0] += dqdci;                /* dwdot[E]/d[E] */
    J[2] -= dqdci;                /* dwdot[N2]/d[E] */
    J[5] += dqdci;                /* dwdot[N2+]/d[E] */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[0];
    J[22] += dqdci;               /* dwdot[E]/d[N2] */
    J[24] -= dqdci;               /* dwdot[N2]/d[N2] */
    J[27] += dqdci;               /* dwdot[N2+]/d[N2] */
    /* d()/dT */
    J[110] += dqdT;               /* dwdot[E]/dT */
    J[112] -= dqdT;               /* dwdot[N2]/dT */
    J[115] += dqdT;               /* dwdot[N2+]/dT */

    /*reaction 2: E + O2 => E + E + O2+ */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[0]*sc[1];
    k_f = 1.0000000000000002e-06 * 1
                * exp(0 * tc[0] - 0.50321666580471969 * (1) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  1  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] += q; /* E */
    wdot[1] -= q; /* O2 */
    wdot[4] += q; /* O2+ */
    /* d()/d[E] */
    dqdci =  + k_f*sc[1];
    J[0] += dqdci;                /* dwdot[E]/d[E] */
    J[1] -= dqdci;                /* dwdot[O2]/d[E] */
    J[4] += dqdci;                /* dwdot[O2+]/d[E] */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[0];
    J[11] += dqdci;               /* dwdot[E]/d[O2] */
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[15] += dqdci;               /* dwdot[O2+]/d[O2] */
    /* d()/dT */
    J[110] += dqdT;               /* dwdot[E]/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[114] += dqdT;               /* dwdot[O2+]/dT */

    /*reaction 3: N2+ + N2 + N2 => N4+ + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = pow(sc[2], 2.000000)*sc[5];
    k_f = 1.0000000000000002e-12 * 1.8132242e+19
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[2] -= q; /* N2 */
    wdot[5] -= q; /* N2+ */
    wdot[7] += q; /* N4+ */
    /* d()/d[N2] */
    dqdci =  + k_f*2.000000*sc[2]*sc[5];
    J[24] -= dqdci;               /* dwdot[N2]/d[N2] */
    J[27] -= dqdci;               /* dwdot[N2+]/d[N2] */
    J[29] += dqdci;               /* dwdot[N4+]/d[N2] */
    /* d()/d[N2+] */
    dqdci =  + k_f*pow(sc[2], 2.000000);
    J[57] -= dqdci;               /* dwdot[N2]/d[N2+] */
    J[60] -= dqdci;               /* dwdot[N2+]/d[N2+] */
    J[62] += dqdci;               /* dwdot[N4+]/d[N2+] */
    /* d()/dT */
    J[112] -= dqdT;               /* dwdot[N2]/dT */
    J[115] -= dqdT;               /* dwdot[N2+]/dT */
    J[117] += dqdT;               /* dwdot[N4+]/dT */

    /*reaction 4: N2+ + N2 + O2 => N4+ + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[2]*sc[5];
    k_f = 1.0000000000000002e-12 * 1.8132242e+19
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[2] -= q; /* N2 */
    wdot[5] -= q; /* N2+ */
    wdot[7] += q; /* N4+ */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[2]*sc[5];
    J[13] -= dqdci;               /* dwdot[N2]/d[O2] */
    J[16] -= dqdci;               /* dwdot[N2+]/d[O2] */
    J[18] += dqdci;               /* dwdot[N4+]/d[O2] */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[1]*sc[5];
    J[24] -= dqdci;               /* dwdot[N2]/d[N2] */
    J[27] -= dqdci;               /* dwdot[N2+]/d[N2] */
    J[29] += dqdci;               /* dwdot[N4+]/d[N2] */
    /* d()/d[N2+] */
    dqdci =  + k_f*sc[1]*sc[2];
    J[57] -= dqdci;               /* dwdot[N2]/d[N2+] */
    J[60] -= dqdci;               /* dwdot[N2+]/d[N2+] */
    J[62] += dqdci;               /* dwdot[N4+]/d[N2+] */
    /* d()/dT */
    J[112] -= dqdT;               /* dwdot[N2]/dT */
    J[115] -= dqdT;               /* dwdot[N2+]/dT */
    J[117] += dqdT;               /* dwdot[N4+]/dT */

    /*reaction 5: N4+ + O2 => O2+ + N2 + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[7];
    k_f = 1.0000000000000002e-06 * 150550000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] -= q; /* O2 */
    wdot[2] += 2 * q; /* N2 */
    wdot[4] += q; /* O2+ */
    wdot[7] -= q; /* N4+ */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[7];
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[13] += 2 * dqdci;           /* dwdot[N2]/d[O2] */
    J[15] += dqdci;               /* dwdot[O2+]/d[O2] */
    J[18] -= dqdci;               /* dwdot[N4+]/d[O2] */
    /* d()/d[N4+] */
    dqdci =  + k_f*sc[1];
    J[78] -= dqdci;               /* dwdot[O2]/d[N4+] */
    J[79] += 2 * dqdci;           /* dwdot[N2]/d[N4+] */
    J[81] += dqdci;               /* dwdot[O2+]/d[N4+] */
    J[84] -= dqdci;               /* dwdot[N4+]/d[N4+] */
    /* d()/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[112] += 2 * dqdT;           /* dwdot[N2]/dT */
    J[114] += dqdT;               /* dwdot[O2+]/dT */
    J[117] -= dqdT;               /* dwdot[N4+]/dT */

    /*reaction 6: N2+ + O2 => O2+ + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[5];
    k_f = 1.0000000000000002e-06 * 36132000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] -= q; /* O2 */
    wdot[2] += q; /* N2 */
    wdot[4] += q; /* O2+ */
    wdot[5] -= q; /* N2+ */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[5];
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[13] += dqdci;               /* dwdot[N2]/d[O2] */
    J[15] += dqdci;               /* dwdot[O2+]/d[O2] */
    J[16] -= dqdci;               /* dwdot[N2+]/d[O2] */
    /* d()/d[N2+] */
    dqdci =  + k_f*sc[1];
    J[56] -= dqdci;               /* dwdot[O2]/d[N2+] */
    J[57] += dqdci;               /* dwdot[N2]/d[N2+] */
    J[59] += dqdci;               /* dwdot[O2+]/d[N2+] */
    J[60] -= dqdci;               /* dwdot[N2+]/d[N2+] */
    /* d()/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[112] += dqdT;               /* dwdot[N2]/dT */
    J[114] += dqdT;               /* dwdot[O2+]/dT */
    J[115] -= dqdT;               /* dwdot[N2+]/dT */

    /*reaction 7: O2+ + N2 + N2 => O2pN2 + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = pow(sc[2], 2.000000)*sc[4];
    k_f = 1.0000000000000002e-12 * 3.26380356e+17
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[2] -= q; /* N2 */
    wdot[4] -= q; /* O2+ */
    wdot[8] += q; /* O2pN2 */
    /* d()/d[N2] */
    dqdci =  + k_f*2.000000*sc[2]*sc[4];
    J[24] -= dqdci;               /* dwdot[N2]/d[N2] */
    J[26] -= dqdci;               /* dwdot[O2+]/d[N2] */
    J[30] += dqdci;               /* dwdot[O2pN2]/d[N2] */
    /* d()/d[O2+] */
    dqdci =  + k_f*pow(sc[2], 2.000000);
    J[46] -= dqdci;               /* dwdot[N2]/d[O2+] */
    J[48] -= dqdci;               /* dwdot[O2+]/d[O2+] */
    J[52] += dqdci;               /* dwdot[O2pN2]/d[O2+] */
    /* d()/dT */
    J[112] -= dqdT;               /* dwdot[N2]/dT */
    J[114] -= dqdT;               /* dwdot[O2+]/dT */
    J[118] += dqdT;               /* dwdot[O2pN2]/dT */

    /*reaction 8: O2pN2 + N2 => O2+ + N2 + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[2]*sc[8];
    k_f = 1.0000000000000002e-06 * 258946000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[2] += q; /* N2 */
    wdot[4] += q; /* O2+ */
    wdot[8] -= q; /* O2pN2 */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[8];
    J[24] += dqdci;               /* dwdot[N2]/d[N2] */
    J[26] += dqdci;               /* dwdot[O2+]/d[N2] */
    J[30] -= dqdci;               /* dwdot[O2pN2]/d[N2] */
    /* d()/d[O2pN2] */
    dqdci =  + k_f*sc[2];
    J[90] += dqdci;               /* dwdot[N2]/d[O2pN2] */
    J[92] += dqdci;               /* dwdot[O2+]/d[O2pN2] */
    J[96] -= dqdci;               /* dwdot[O2pN2]/d[O2pN2] */
    /* d()/dT */
    J[112] += dqdT;               /* dwdot[N2]/dT */
    J[114] += dqdT;               /* dwdot[O2+]/dT */
    J[118] -= dqdT;               /* dwdot[O2pN2]/dT */

    /*reaction 9: O2pN2 + O2 => O4+ + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[8];
    k_f = 1.0000000000000002e-06 * 602200000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] -= q; /* O2 */
    wdot[2] += q; /* N2 */
    wdot[6] += q; /* O4+ */
    wdot[8] -= q; /* O2pN2 */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[8];
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[13] += dqdci;               /* dwdot[N2]/d[O2] */
    J[17] += dqdci;               /* dwdot[O4+]/d[O2] */
    J[19] -= dqdci;               /* dwdot[O2pN2]/d[O2] */
    /* d()/d[O2pN2] */
    dqdci =  + k_f*sc[1];
    J[89] -= dqdci;               /* dwdot[O2]/d[O2pN2] */
    J[90] += dqdci;               /* dwdot[N2]/d[O2pN2] */
    J[94] += dqdci;               /* dwdot[O4+]/d[O2pN2] */
    J[96] -= dqdci;               /* dwdot[O2pN2]/d[O2pN2] */
    /* d()/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[112] += dqdT;               /* dwdot[N2]/dT */
    J[116] += dqdT;               /* dwdot[O4+]/dT */
    J[118] -= dqdT;               /* dwdot[O2pN2]/dT */

    /*reaction 10: O2+ + O2 + N2 => O4+ + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[2]*sc[4];
    k_f = 1.0000000000000002e-12 * 8.70347616e+17
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] -= q; /* O2 */
    wdot[4] -= q; /* O2+ */
    wdot[6] += q; /* O4+ */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[2]*sc[4];
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[15] -= dqdci;               /* dwdot[O2+]/d[O2] */
    J[17] += dqdci;               /* dwdot[O4+]/d[O2] */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[1]*sc[4];
    J[23] -= dqdci;               /* dwdot[O2]/d[N2] */
    J[26] -= dqdci;               /* dwdot[O2+]/d[N2] */
    J[28] += dqdci;               /* dwdot[O4+]/d[N2] */
    /* d()/d[O2+] */
    dqdci =  + k_f*sc[1]*sc[2];
    J[45] -= dqdci;               /* dwdot[O2]/d[O2+] */
    J[48] -= dqdci;               /* dwdot[O2+]/d[O2+] */
    J[50] += dqdci;               /* dwdot[O4+]/d[O2+] */
    /* d()/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[114] -= dqdT;               /* dwdot[O2+]/dT */
    J[116] += dqdT;               /* dwdot[O4+]/dT */

    /*reaction 11: O2+ + O2 + O2 => O4+ + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = pow(sc[1], 2.000000)*sc[4];
    k_f = 1.0000000000000002e-12 * 8.70347616e+17
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] -= q; /* O2 */
    wdot[4] -= q; /* O2+ */
    wdot[6] += q; /* O4+ */
    /* d()/d[O2] */
    dqdci =  + k_f*2.000000*sc[1]*sc[4];
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[15] -= dqdci;               /* dwdot[O2+]/d[O2] */
    J[17] += dqdci;               /* dwdot[O4+]/d[O2] */
    /* d()/d[O2+] */
    dqdci =  + k_f*pow(sc[1], 2.000000);
    J[45] -= dqdci;               /* dwdot[O2]/d[O2+] */
    J[48] -= dqdci;               /* dwdot[O2+]/d[O2+] */
    J[50] += dqdci;               /* dwdot[O4+]/d[O2+] */
    /* d()/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[114] -= dqdT;               /* dwdot[O2+]/dT */
    J[116] += dqdT;               /* dwdot[O4+]/dT */

    /*reaction 12: E + O4+ => O2 + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[0]*sc[6];
    k_f = 1.0000000000000002e-06 * 1
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] -= q; /* E */
    wdot[1] += 2 * q; /* O2 */
    wdot[6] -= q; /* O4+ */
    /* d()/d[E] */
    dqdci =  + k_f*sc[6];
    J[0] -= dqdci;                /* dwdot[E]/d[E] */
    J[1] += 2 * dqdci;            /* dwdot[O2]/d[E] */
    J[6] -= dqdci;                /* dwdot[O4+]/d[E] */
    /* d()/d[O4+] */
    dqdci =  + k_f*sc[0];
    J[66] -= dqdci;               /* dwdot[E]/d[O4+] */
    J[67] += 2 * dqdci;           /* dwdot[O2]/d[O4+] */
    J[72] -= dqdci;               /* dwdot[O4+]/d[O4+] */
    /* d()/dT */
    J[110] -= dqdT;               /* dwdot[E]/dT */
    J[111] += 2 * dqdT;           /* dwdot[O2]/dT */
    J[116] -= dqdT;               /* dwdot[O4+]/dT */

    /*reaction 13: E + O2+ => O + O */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[0]*sc[4];
    k_f = 1.0000000000000002e-06 * 1
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] -= q; /* E */
    wdot[3] += 2 * q; /* O */
    wdot[4] -= q; /* O2+ */
    /* d()/d[E] */
    dqdci =  + k_f*sc[4];
    J[0] -= dqdci;                /* dwdot[E]/d[E] */
    J[3] += 2 * dqdci;            /* dwdot[O]/d[E] */
    J[4] -= dqdci;                /* dwdot[O2+]/d[E] */
    /* d()/d[O2+] */
    dqdci =  + k_f*sc[0];
    J[44] -= dqdci;               /* dwdot[E]/d[O2+] */
    J[47] += 2 * dqdci;           /* dwdot[O]/d[O2+] */
    J[48] -= dqdci;               /* dwdot[O2+]/d[O2+] */
    /* d()/dT */
    J[110] -= dqdT;               /* dwdot[E]/dT */
    J[113] += 2 * dqdT;           /* dwdot[O]/dT */
    J[114] -= dqdT;               /* dwdot[O2+]/dT */

    /*reaction 14: E + O2 + O2 => O2- + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[0]*pow(sc[1], 2.000000);
    k_f = 1.0000000000000002e-12 * 1
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] -= q; /* E */
    wdot[1] -= q; /* O2 */
    wdot[9] += q; /* O2- */
    /* d()/d[E] */
    dqdci =  + k_f*pow(sc[1], 2.000000);
    J[0] -= dqdci;                /* dwdot[E]/d[E] */
    J[1] -= dqdci;                /* dwdot[O2]/d[E] */
    J[9] += dqdci;                /* dwdot[O2-]/d[E] */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[0]*2.000000*sc[1];
    J[11] -= dqdci;               /* dwdot[E]/d[O2] */
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[20] += dqdci;               /* dwdot[O2-]/d[O2] */
    /* d()/dT */
    J[110] -= dqdT;               /* dwdot[E]/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[119] += dqdT;               /* dwdot[O2-]/dT */

    /*reaction 15: E + O2 + N2 => O2- + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[0]*sc[1]*sc[2];
    k_f = 1.0000000000000002e-12 * 1
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] -= q; /* E */
    wdot[1] -= q; /* O2 */
    wdot[9] += q; /* O2- */
    /* d()/d[E] */
    dqdci =  + k_f*sc[1]*sc[2];
    J[0] -= dqdci;                /* dwdot[E]/d[E] */
    J[1] -= dqdci;                /* dwdot[O2]/d[E] */
    J[9] += dqdci;                /* dwdot[O2-]/d[E] */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[0]*sc[2];
    J[11] -= dqdci;               /* dwdot[E]/d[O2] */
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[20] += dqdci;               /* dwdot[O2-]/d[O2] */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[0]*sc[1];
    J[22] -= dqdci;               /* dwdot[E]/d[N2] */
    J[23] -= dqdci;               /* dwdot[O2]/d[N2] */
    J[31] += dqdci;               /* dwdot[O2-]/d[N2] */
    /* d()/dT */
    J[110] -= dqdT;               /* dwdot[E]/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[119] += dqdT;               /* dwdot[O2-]/dT */

    /*reaction 16: O2- + O4+ => O2 + O2 + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[6]*sc[9];
    k_f = 1.0000000000000002e-06 * 60220000000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] += 3 * q; /* O2 */
    wdot[6] -= q; /* O4+ */
    wdot[9] -= q; /* O2- */
    /* d()/d[O4+] */
    dqdci =  + k_f*sc[9];
    J[67] += 3 * dqdci;           /* dwdot[O2]/d[O4+] */
    J[72] -= dqdci;               /* dwdot[O4+]/d[O4+] */
    J[75] -= dqdci;               /* dwdot[O2-]/d[O4+] */
    /* d()/d[O2-] */
    dqdci =  + k_f*sc[6];
    J[100] += 3 * dqdci;          /* dwdot[O2]/d[O2-] */
    J[105] -= dqdci;              /* dwdot[O4+]/d[O2-] */
    J[108] -= dqdci;              /* dwdot[O2-]/d[O2-] */
    /* d()/dT */
    J[111] += 3 * dqdT;           /* dwdot[O2]/dT */
    J[116] -= dqdT;               /* dwdot[O4+]/dT */
    J[119] -= dqdT;               /* dwdot[O2-]/dT */

    /*reaction 17: O2- + O4+ + O2 => O2 + O2 + O2 + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[6]*sc[9];
    k_f = 1.0000000000000002e-12 * 7.2528968000000003e+22
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] += 3 * q; /* O2 */
    wdot[6] -= q; /* O4+ */
    wdot[9] -= q; /* O2- */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[6]*sc[9];
    J[12] += 3 * dqdci;           /* dwdot[O2]/d[O2] */
    J[17] -= dqdci;               /* dwdot[O4+]/d[O2] */
    J[20] -= dqdci;               /* dwdot[O2-]/d[O2] */
    /* d()/d[O4+] */
    dqdci =  + k_f*sc[1]*sc[9];
    J[67] += 3 * dqdci;           /* dwdot[O2]/d[O4+] */
    J[72] -= dqdci;               /* dwdot[O4+]/d[O4+] */
    J[75] -= dqdci;               /* dwdot[O2-]/d[O4+] */
    /* d()/d[O2-] */
    dqdci =  + k_f*sc[1]*sc[6];
    J[100] += 3 * dqdci;          /* dwdot[O2]/d[O2-] */
    J[105] -= dqdci;              /* dwdot[O4+]/d[O2-] */
    J[108] -= dqdci;              /* dwdot[O2-]/d[O2-] */
    /* d()/dT */
    J[111] += 3 * dqdT;           /* dwdot[O2]/dT */
    J[116] -= dqdT;               /* dwdot[O4+]/dT */
    J[119] -= dqdT;               /* dwdot[O2-]/dT */

    /*reaction 18: O2- + O4+ + N2 => O2 + O2 + O2 + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[2]*sc[6]*sc[9];
    k_f = 1.0000000000000002e-12 * 7.2528968000000003e+22
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] += 3 * q; /* O2 */
    wdot[6] -= q; /* O4+ */
    wdot[9] -= q; /* O2- */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[6]*sc[9];
    J[23] += 3 * dqdci;           /* dwdot[O2]/d[N2] */
    J[28] -= dqdci;               /* dwdot[O4+]/d[N2] */
    J[31] -= dqdci;               /* dwdot[O2-]/d[N2] */
    /* d()/d[O4+] */
    dqdci =  + k_f*sc[2]*sc[9];
    J[67] += 3 * dqdci;           /* dwdot[O2]/d[O4+] */
    J[72] -= dqdci;               /* dwdot[O4+]/d[O4+] */
    J[75] -= dqdci;               /* dwdot[O2-]/d[O4+] */
    /* d()/d[O2-] */
    dqdci =  + k_f*sc[2]*sc[6];
    J[100] += 3 * dqdci;          /* dwdot[O2]/d[O2-] */
    J[105] -= dqdci;              /* dwdot[O4+]/d[O2-] */
    J[108] -= dqdci;              /* dwdot[O2-]/d[O2-] */
    /* d()/dT */
    J[111] += 3 * dqdT;           /* dwdot[O2]/dT */
    J[116] -= dqdT;               /* dwdot[O4+]/dT */
    J[119] -= dqdT;               /* dwdot[O2-]/dT */

    /*reaction 19: O2- + O2+ + O2 => O2 + O2 + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[4]*sc[9];
    k_f = 1.0000000000000002e-12 * 7.2528968000000003e+22
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] += 2 * q; /* O2 */
    wdot[4] -= q; /* O2+ */
    wdot[9] -= q; /* O2- */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[4]*sc[9];
    J[12] += 2 * dqdci;           /* dwdot[O2]/d[O2] */
    J[15] -= dqdci;               /* dwdot[O2+]/d[O2] */
    J[20] -= dqdci;               /* dwdot[O2-]/d[O2] */
    /* d()/d[O2+] */
    dqdci =  + k_f*sc[1]*sc[9];
    J[45] += 2 * dqdci;           /* dwdot[O2]/d[O2+] */
    J[48] -= dqdci;               /* dwdot[O2+]/d[O2+] */
    J[53] -= dqdci;               /* dwdot[O2-]/d[O2+] */
    /* d()/d[O2-] */
    dqdci =  + k_f*sc[1]*sc[4];
    J[100] += 2 * dqdci;          /* dwdot[O2]/d[O2-] */
    J[103] -= dqdci;              /* dwdot[O2+]/d[O2-] */
    J[108] -= dqdci;              /* dwdot[O2-]/d[O2-] */
    /* d()/dT */
    J[111] += 2 * dqdT;           /* dwdot[O2]/dT */
    J[114] -= dqdT;               /* dwdot[O2+]/dT */
    J[119] -= dqdT;               /* dwdot[O2-]/dT */

    /*reaction 20: O2- + O2+ + N2 => O2 + O2 + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[2]*sc[4]*sc[9];
    k_f = 1.0000000000000002e-12 * 7.2528968000000003e+22
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] += 2 * q; /* O2 */
    wdot[4] -= q; /* O2+ */
    wdot[9] -= q; /* O2- */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[4]*sc[9];
    J[23] += 2 * dqdci;           /* dwdot[O2]/d[N2] */
    J[26] -= dqdci;               /* dwdot[O2+]/d[N2] */
    J[31] -= dqdci;               /* dwdot[O2-]/d[N2] */
    /* d()/d[O2+] */
    dqdci =  + k_f*sc[2]*sc[9];
    J[45] += 2 * dqdci;           /* dwdot[O2]/d[O2+] */
    J[48] -= dqdci;               /* dwdot[O2+]/d[O2+] */
    J[53] -= dqdci;               /* dwdot[O2-]/d[O2+] */
    /* d()/d[O2-] */
    dqdci =  + k_f*sc[2]*sc[4];
    J[100] += 2 * dqdci;          /* dwdot[O2]/d[O2-] */
    J[103] -= dqdci;              /* dwdot[O2+]/d[O2-] */
    J[108] -= dqdci;              /* dwdot[O2-]/d[O2-] */
    /* d()/dT */
    J[111] += 2 * dqdT;           /* dwdot[O2]/dT */
    J[114] -= dqdT;               /* dwdot[O2+]/dT */
    J[119] -= dqdT;               /* dwdot[O2-]/dT */

    /*reaction 21: O2- + O2 => E + O2 + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[9];
    k_f = 1.0000000000000002e-06 * 1
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] += q; /* E */
    wdot[1] += q; /* O2 */
    wdot[9] -= q; /* O2- */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[9];
    J[11] += dqdci;               /* dwdot[E]/d[O2] */
    J[12] += dqdci;               /* dwdot[O2]/d[O2] */
    J[20] -= dqdci;               /* dwdot[O2-]/d[O2] */
    /* d()/d[O2-] */
    dqdci =  + k_f*sc[1];
    J[99] += dqdci;               /* dwdot[E]/d[O2-] */
    J[100] += dqdci;              /* dwdot[O2]/d[O2-] */
    J[108] -= dqdci;              /* dwdot[O2-]/d[O2-] */
    /* d()/dT */
    J[110] += dqdT;               /* dwdot[E]/dT */
    J[111] += dqdT;               /* dwdot[O2]/dT */
    J[119] -= dqdT;               /* dwdot[O2-]/dT */

    /*reaction 22: O2- + N2 => E + O2 + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[2]*sc[9];
    k_f = 1.0000000000000002e-06 * 1
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  0  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] += q; /* E */
    wdot[1] += q; /* O2 */
    wdot[9] -= q; /* O2- */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[9];
    J[22] += dqdci;               /* dwdot[E]/d[N2] */
    J[23] += dqdci;               /* dwdot[O2]/d[N2] */
    J[31] -= dqdci;               /* dwdot[O2-]/d[N2] */
    /* d()/d[O2-] */
    dqdci =  + k_f*sc[2];
    J[99] += dqdci;               /* dwdot[E]/d[O2-] */
    J[100] += dqdci;              /* dwdot[O2]/d[O2-] */
    J[108] -= dqdci;              /* dwdot[O2-]/d[O2-] */
    /* d()/dT */
    J[110] += dqdT;               /* dwdot[E]/dT */
    J[111] += dqdT;               /* dwdot[O2]/dT */
    J[119] -= dqdT;               /* dwdot[O2-]/dT */

    double c_R[10], dcRdT[10], e_RT[10];
    double * eh_RT;
    if (consP) {
        cp_R(c_R, tc);
        dcvpRdT(dcRdT, tc);
        eh_RT = &h_RT[0];
    }
    else {
        cv_R(c_R, tc);
        dcvpRdT(dcRdT, tc);
        speciesInternalEnergy(e_RT, tc);
        eh_RT = &e_RT[0];
    }

    double cmix = 0.0, ehmix = 0.0, dcmixdT=0.0, dehmixdT=0.0;
    for (int k = 0; k < 10; ++k) {
        cmix += c_R[k]*sc[k];
        dcmixdT += dcRdT[k]*sc[k];
        ehmix += eh_RT[k]*wdot[k];
        dehmixdT += invT*(c_R[k]-eh_RT[k])*wdot[k] + eh_RT[k]*J[110+k];
    }

    double cmixinv = 1.0/cmix;
    double tmp1 = ehmix*cmixinv;
    double tmp3 = cmixinv*T;
    double tmp2 = tmp1*tmp3;
    double dehmixdc;
    /* dTdot/d[X] */
    for (int k = 0; k < 10; ++k) {
        dehmixdc = 0.0;
        for (int m = 0; m < 10; ++m) {
            dehmixdc += eh_RT[m]*J[k*11+m];
        }
        J[k*11+10] = tmp2*c_R[k] - tmp3*dehmixdc;
    }
    /* dTdot/dT */
    J[120] = -tmp1 + tmp2*dcmixdT - tmp3*dehmixdT;

return;
}
#endif


#ifndef AMREX_USE_CUDA
/*compute the reaction Jacobian on CPU */
void aJacobian(double *  J, double *  sc, double T, int consP)
{
    for (int i=0; i<121; i++) {
        J[i] = 0.0;
    }

    double wdot[10];
    for (int k=0; k<10; k++) {
        wdot[k] = 0.0;
    }

    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double invT = 1.0 / tc[1];
    double invT2 = invT * invT;

    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 / T;
    double refCinv = 1.0 / refC;

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int k = 0; k < 10; ++k) {
        mixture += sc[k];
    }

    /*compute the Gibbs free energy */
    double g_RT[10];
    gibbs(g_RT, tc);

    /*compute the species enthalpy */
    double h_RT[10];
    speciesEnthalpy(h_RT, tc);

    double phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
    double dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
    double dqdci, dcdc_fac, dqdc[10];
    double Pr, fPr, F, k_0, logPr;
    double logFcent, troe_c, troe_n, troePr_den, troePr, troe;
    double Fcent1, Fcent2, Fcent3, Fcent;
    double dlogFdc, dlogFdn, dlogFdcn_fac;
    double dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
    const double ln10 = log(10.0);
    const double log10e = 1.0/log(10.0);
    /*reaction 1: E + N2 => E + E + N2+ */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[0]*sc[2];
    k_f = prefactor_units[0] * fwd_A[0]
                * exp(fwd_beta[0] * tc[0] - activation_units[0] * fwd_Ea[0] * invT);
    dlnkfdT = fwd_beta[0] * invT + activation_units[0] * fwd_Ea[0] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] += q; /* E */
    wdot[2] -= q; /* N2 */
    wdot[5] += q; /* N2+ */
    /* d()/d[E] */
    dqdci =  + k_f*sc[2];
    J[0] += dqdci;                /* dwdot[E]/d[E] */
    J[2] -= dqdci;                /* dwdot[N2]/d[E] */
    J[5] += dqdci;                /* dwdot[N2+]/d[E] */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[0];
    J[22] += dqdci;               /* dwdot[E]/d[N2] */
    J[24] -= dqdci;               /* dwdot[N2]/d[N2] */
    J[27] += dqdci;               /* dwdot[N2+]/d[N2] */
    /* d()/dT */
    J[110] += dqdT;               /* dwdot[E]/dT */
    J[112] -= dqdT;               /* dwdot[N2]/dT */
    J[115] += dqdT;               /* dwdot[N2+]/dT */

    /*reaction 2: E + O2 => E + E + O2+ */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[0]*sc[1];
    k_f = prefactor_units[1] * fwd_A[1]
                * exp(fwd_beta[1] * tc[0] - activation_units[1] * fwd_Ea[1] * invT);
    dlnkfdT = fwd_beta[1] * invT + activation_units[1] * fwd_Ea[1] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] += q; /* E */
    wdot[1] -= q; /* O2 */
    wdot[4] += q; /* O2+ */
    /* d()/d[E] */
    dqdci =  + k_f*sc[1];
    J[0] += dqdci;                /* dwdot[E]/d[E] */
    J[1] -= dqdci;                /* dwdot[O2]/d[E] */
    J[4] += dqdci;                /* dwdot[O2+]/d[E] */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[0];
    J[11] += dqdci;               /* dwdot[E]/d[O2] */
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[15] += dqdci;               /* dwdot[O2+]/d[O2] */
    /* d()/dT */
    J[110] += dqdT;               /* dwdot[E]/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[114] += dqdT;               /* dwdot[O2+]/dT */

    /*reaction 3: N2+ + N2 + N2 => N4+ + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = pow(sc[2], 2.000000)*sc[5];
    k_f = prefactor_units[2] * fwd_A[2]
                * exp(fwd_beta[2] * tc[0] - activation_units[2] * fwd_Ea[2] * invT);
    dlnkfdT = fwd_beta[2] * invT + activation_units[2] * fwd_Ea[2] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[2] -= q; /* N2 */
    wdot[5] -= q; /* N2+ */
    wdot[7] += q; /* N4+ */
    /* d()/d[N2] */
    dqdci =  + k_f*2.000000*sc[2]*sc[5];
    J[24] -= dqdci;               /* dwdot[N2]/d[N2] */
    J[27] -= dqdci;               /* dwdot[N2+]/d[N2] */
    J[29] += dqdci;               /* dwdot[N4+]/d[N2] */
    /* d()/d[N2+] */
    dqdci =  + k_f*pow(sc[2], 2.000000);
    J[57] -= dqdci;               /* dwdot[N2]/d[N2+] */
    J[60] -= dqdci;               /* dwdot[N2+]/d[N2+] */
    J[62] += dqdci;               /* dwdot[N4+]/d[N2+] */
    /* d()/dT */
    J[112] -= dqdT;               /* dwdot[N2]/dT */
    J[115] -= dqdT;               /* dwdot[N2+]/dT */
    J[117] += dqdT;               /* dwdot[N4+]/dT */

    /*reaction 4: N2+ + N2 + O2 => N4+ + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[2]*sc[5];
    k_f = prefactor_units[3] * fwd_A[3]
                * exp(fwd_beta[3] * tc[0] - activation_units[3] * fwd_Ea[3] * invT);
    dlnkfdT = fwd_beta[3] * invT + activation_units[3] * fwd_Ea[3] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[2] -= q; /* N2 */
    wdot[5] -= q; /* N2+ */
    wdot[7] += q; /* N4+ */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[2]*sc[5];
    J[13] -= dqdci;               /* dwdot[N2]/d[O2] */
    J[16] -= dqdci;               /* dwdot[N2+]/d[O2] */
    J[18] += dqdci;               /* dwdot[N4+]/d[O2] */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[1]*sc[5];
    J[24] -= dqdci;               /* dwdot[N2]/d[N2] */
    J[27] -= dqdci;               /* dwdot[N2+]/d[N2] */
    J[29] += dqdci;               /* dwdot[N4+]/d[N2] */
    /* d()/d[N2+] */
    dqdci =  + k_f*sc[1]*sc[2];
    J[57] -= dqdci;               /* dwdot[N2]/d[N2+] */
    J[60] -= dqdci;               /* dwdot[N2+]/d[N2+] */
    J[62] += dqdci;               /* dwdot[N4+]/d[N2+] */
    /* d()/dT */
    J[112] -= dqdT;               /* dwdot[N2]/dT */
    J[115] -= dqdT;               /* dwdot[N2+]/dT */
    J[117] += dqdT;               /* dwdot[N4+]/dT */

    /*reaction 5: N4+ + O2 => O2+ + N2 + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[7];
    k_f = prefactor_units[4] * fwd_A[4]
                * exp(fwd_beta[4] * tc[0] - activation_units[4] * fwd_Ea[4] * invT);
    dlnkfdT = fwd_beta[4] * invT + activation_units[4] * fwd_Ea[4] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] -= q; /* O2 */
    wdot[2] += 2 * q; /* N2 */
    wdot[4] += q; /* O2+ */
    wdot[7] -= q; /* N4+ */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[7];
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[13] += 2 * dqdci;           /* dwdot[N2]/d[O2] */
    J[15] += dqdci;               /* dwdot[O2+]/d[O2] */
    J[18] -= dqdci;               /* dwdot[N4+]/d[O2] */
    /* d()/d[N4+] */
    dqdci =  + k_f*sc[1];
    J[78] -= dqdci;               /* dwdot[O2]/d[N4+] */
    J[79] += 2 * dqdci;           /* dwdot[N2]/d[N4+] */
    J[81] += dqdci;               /* dwdot[O2+]/d[N4+] */
    J[84] -= dqdci;               /* dwdot[N4+]/d[N4+] */
    /* d()/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[112] += 2 * dqdT;           /* dwdot[N2]/dT */
    J[114] += dqdT;               /* dwdot[O2+]/dT */
    J[117] -= dqdT;               /* dwdot[N4+]/dT */

    /*reaction 6: N2+ + O2 => O2+ + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[5];
    k_f = prefactor_units[5] * fwd_A[5]
                * exp(fwd_beta[5] * tc[0] - activation_units[5] * fwd_Ea[5] * invT);
    dlnkfdT = fwd_beta[5] * invT + activation_units[5] * fwd_Ea[5] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] -= q; /* O2 */
    wdot[2] += q; /* N2 */
    wdot[4] += q; /* O2+ */
    wdot[5] -= q; /* N2+ */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[5];
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[13] += dqdci;               /* dwdot[N2]/d[O2] */
    J[15] += dqdci;               /* dwdot[O2+]/d[O2] */
    J[16] -= dqdci;               /* dwdot[N2+]/d[O2] */
    /* d()/d[N2+] */
    dqdci =  + k_f*sc[1];
    J[56] -= dqdci;               /* dwdot[O2]/d[N2+] */
    J[57] += dqdci;               /* dwdot[N2]/d[N2+] */
    J[59] += dqdci;               /* dwdot[O2+]/d[N2+] */
    J[60] -= dqdci;               /* dwdot[N2+]/d[N2+] */
    /* d()/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[112] += dqdT;               /* dwdot[N2]/dT */
    J[114] += dqdT;               /* dwdot[O2+]/dT */
    J[115] -= dqdT;               /* dwdot[N2+]/dT */

    /*reaction 7: O2+ + N2 + N2 => O2pN2 + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = pow(sc[2], 2.000000)*sc[4];
    k_f = prefactor_units[6] * fwd_A[6]
                * exp(fwd_beta[6] * tc[0] - activation_units[6] * fwd_Ea[6] * invT);
    dlnkfdT = fwd_beta[6] * invT + activation_units[6] * fwd_Ea[6] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[2] -= q; /* N2 */
    wdot[4] -= q; /* O2+ */
    wdot[8] += q; /* O2pN2 */
    /* d()/d[N2] */
    dqdci =  + k_f*2.000000*sc[2]*sc[4];
    J[24] -= dqdci;               /* dwdot[N2]/d[N2] */
    J[26] -= dqdci;               /* dwdot[O2+]/d[N2] */
    J[30] += dqdci;               /* dwdot[O2pN2]/d[N2] */
    /* d()/d[O2+] */
    dqdci =  + k_f*pow(sc[2], 2.000000);
    J[46] -= dqdci;               /* dwdot[N2]/d[O2+] */
    J[48] -= dqdci;               /* dwdot[O2+]/d[O2+] */
    J[52] += dqdci;               /* dwdot[O2pN2]/d[O2+] */
    /* d()/dT */
    J[112] -= dqdT;               /* dwdot[N2]/dT */
    J[114] -= dqdT;               /* dwdot[O2+]/dT */
    J[118] += dqdT;               /* dwdot[O2pN2]/dT */

    /*reaction 8: O2pN2 + N2 => O2+ + N2 + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[2]*sc[8];
    k_f = prefactor_units[7] * fwd_A[7]
                * exp(fwd_beta[7] * tc[0] - activation_units[7] * fwd_Ea[7] * invT);
    dlnkfdT = fwd_beta[7] * invT + activation_units[7] * fwd_Ea[7] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[2] += q; /* N2 */
    wdot[4] += q; /* O2+ */
    wdot[8] -= q; /* O2pN2 */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[8];
    J[24] += dqdci;               /* dwdot[N2]/d[N2] */
    J[26] += dqdci;               /* dwdot[O2+]/d[N2] */
    J[30] -= dqdci;               /* dwdot[O2pN2]/d[N2] */
    /* d()/d[O2pN2] */
    dqdci =  + k_f*sc[2];
    J[90] += dqdci;               /* dwdot[N2]/d[O2pN2] */
    J[92] += dqdci;               /* dwdot[O2+]/d[O2pN2] */
    J[96] -= dqdci;               /* dwdot[O2pN2]/d[O2pN2] */
    /* d()/dT */
    J[112] += dqdT;               /* dwdot[N2]/dT */
    J[114] += dqdT;               /* dwdot[O2+]/dT */
    J[118] -= dqdT;               /* dwdot[O2pN2]/dT */

    /*reaction 9: O2pN2 + O2 => O4+ + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[8];
    k_f = prefactor_units[8] * fwd_A[8]
                * exp(fwd_beta[8] * tc[0] - activation_units[8] * fwd_Ea[8] * invT);
    dlnkfdT = fwd_beta[8] * invT + activation_units[8] * fwd_Ea[8] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] -= q; /* O2 */
    wdot[2] += q; /* N2 */
    wdot[6] += q; /* O4+ */
    wdot[8] -= q; /* O2pN2 */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[8];
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[13] += dqdci;               /* dwdot[N2]/d[O2] */
    J[17] += dqdci;               /* dwdot[O4+]/d[O2] */
    J[19] -= dqdci;               /* dwdot[O2pN2]/d[O2] */
    /* d()/d[O2pN2] */
    dqdci =  + k_f*sc[1];
    J[89] -= dqdci;               /* dwdot[O2]/d[O2pN2] */
    J[90] += dqdci;               /* dwdot[N2]/d[O2pN2] */
    J[94] += dqdci;               /* dwdot[O4+]/d[O2pN2] */
    J[96] -= dqdci;               /* dwdot[O2pN2]/d[O2pN2] */
    /* d()/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[112] += dqdT;               /* dwdot[N2]/dT */
    J[116] += dqdT;               /* dwdot[O4+]/dT */
    J[118] -= dqdT;               /* dwdot[O2pN2]/dT */

    /*reaction 10: O2+ + O2 + N2 => O4+ + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[2]*sc[4];
    k_f = prefactor_units[9] * fwd_A[9]
                * exp(fwd_beta[9] * tc[0] - activation_units[9] * fwd_Ea[9] * invT);
    dlnkfdT = fwd_beta[9] * invT + activation_units[9] * fwd_Ea[9] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] -= q; /* O2 */
    wdot[4] -= q; /* O2+ */
    wdot[6] += q; /* O4+ */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[2]*sc[4];
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[15] -= dqdci;               /* dwdot[O2+]/d[O2] */
    J[17] += dqdci;               /* dwdot[O4+]/d[O2] */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[1]*sc[4];
    J[23] -= dqdci;               /* dwdot[O2]/d[N2] */
    J[26] -= dqdci;               /* dwdot[O2+]/d[N2] */
    J[28] += dqdci;               /* dwdot[O4+]/d[N2] */
    /* d()/d[O2+] */
    dqdci =  + k_f*sc[1]*sc[2];
    J[45] -= dqdci;               /* dwdot[O2]/d[O2+] */
    J[48] -= dqdci;               /* dwdot[O2+]/d[O2+] */
    J[50] += dqdci;               /* dwdot[O4+]/d[O2+] */
    /* d()/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[114] -= dqdT;               /* dwdot[O2+]/dT */
    J[116] += dqdT;               /* dwdot[O4+]/dT */

    /*reaction 11: O2+ + O2 + O2 => O4+ + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = pow(sc[1], 2.000000)*sc[4];
    k_f = prefactor_units[10] * fwd_A[10]
                * exp(fwd_beta[10] * tc[0] - activation_units[10] * fwd_Ea[10] * invT);
    dlnkfdT = fwd_beta[10] * invT + activation_units[10] * fwd_Ea[10] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] -= q; /* O2 */
    wdot[4] -= q; /* O2+ */
    wdot[6] += q; /* O4+ */
    /* d()/d[O2] */
    dqdci =  + k_f*2.000000*sc[1]*sc[4];
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[15] -= dqdci;               /* dwdot[O2+]/d[O2] */
    J[17] += dqdci;               /* dwdot[O4+]/d[O2] */
    /* d()/d[O2+] */
    dqdci =  + k_f*pow(sc[1], 2.000000);
    J[45] -= dqdci;               /* dwdot[O2]/d[O2+] */
    J[48] -= dqdci;               /* dwdot[O2+]/d[O2+] */
    J[50] += dqdci;               /* dwdot[O4+]/d[O2+] */
    /* d()/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[114] -= dqdT;               /* dwdot[O2+]/dT */
    J[116] += dqdT;               /* dwdot[O4+]/dT */

    /*reaction 12: E + O4+ => O2 + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[0]*sc[6];
    k_f = prefactor_units[11] * fwd_A[11]
                * exp(fwd_beta[11] * tc[0] - activation_units[11] * fwd_Ea[11] * invT);
    dlnkfdT = fwd_beta[11] * invT + activation_units[11] * fwd_Ea[11] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] -= q; /* E */
    wdot[1] += 2 * q; /* O2 */
    wdot[6] -= q; /* O4+ */
    /* d()/d[E] */
    dqdci =  + k_f*sc[6];
    J[0] -= dqdci;                /* dwdot[E]/d[E] */
    J[1] += 2 * dqdci;            /* dwdot[O2]/d[E] */
    J[6] -= dqdci;                /* dwdot[O4+]/d[E] */
    /* d()/d[O4+] */
    dqdci =  + k_f*sc[0];
    J[66] -= dqdci;               /* dwdot[E]/d[O4+] */
    J[67] += 2 * dqdci;           /* dwdot[O2]/d[O4+] */
    J[72] -= dqdci;               /* dwdot[O4+]/d[O4+] */
    /* d()/dT */
    J[110] -= dqdT;               /* dwdot[E]/dT */
    J[111] += 2 * dqdT;           /* dwdot[O2]/dT */
    J[116] -= dqdT;               /* dwdot[O4+]/dT */

    /*reaction 13: E + O2+ => O + O */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[0]*sc[4];
    k_f = prefactor_units[12] * fwd_A[12]
                * exp(fwd_beta[12] * tc[0] - activation_units[12] * fwd_Ea[12] * invT);
    dlnkfdT = fwd_beta[12] * invT + activation_units[12] * fwd_Ea[12] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] -= q; /* E */
    wdot[3] += 2 * q; /* O */
    wdot[4] -= q; /* O2+ */
    /* d()/d[E] */
    dqdci =  + k_f*sc[4];
    J[0] -= dqdci;                /* dwdot[E]/d[E] */
    J[3] += 2 * dqdci;            /* dwdot[O]/d[E] */
    J[4] -= dqdci;                /* dwdot[O2+]/d[E] */
    /* d()/d[O2+] */
    dqdci =  + k_f*sc[0];
    J[44] -= dqdci;               /* dwdot[E]/d[O2+] */
    J[47] += 2 * dqdci;           /* dwdot[O]/d[O2+] */
    J[48] -= dqdci;               /* dwdot[O2+]/d[O2+] */
    /* d()/dT */
    J[110] -= dqdT;               /* dwdot[E]/dT */
    J[113] += 2 * dqdT;           /* dwdot[O]/dT */
    J[114] -= dqdT;               /* dwdot[O2+]/dT */

    /*reaction 14: E + O2 + O2 => O2- + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[0]*pow(sc[1], 2.000000);
    k_f = prefactor_units[13] * fwd_A[13]
                * exp(fwd_beta[13] * tc[0] - activation_units[13] * fwd_Ea[13] * invT);
    dlnkfdT = fwd_beta[13] * invT + activation_units[13] * fwd_Ea[13] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] -= q; /* E */
    wdot[1] -= q; /* O2 */
    wdot[9] += q; /* O2- */
    /* d()/d[E] */
    dqdci =  + k_f*pow(sc[1], 2.000000);
    J[0] -= dqdci;                /* dwdot[E]/d[E] */
    J[1] -= dqdci;                /* dwdot[O2]/d[E] */
    J[9] += dqdci;                /* dwdot[O2-]/d[E] */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[0]*2.000000*sc[1];
    J[11] -= dqdci;               /* dwdot[E]/d[O2] */
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[20] += dqdci;               /* dwdot[O2-]/d[O2] */
    /* d()/dT */
    J[110] -= dqdT;               /* dwdot[E]/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[119] += dqdT;               /* dwdot[O2-]/dT */

    /*reaction 15: E + O2 + N2 => O2- + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[0]*sc[1]*sc[2];
    k_f = prefactor_units[14] * fwd_A[14]
                * exp(fwd_beta[14] * tc[0] - activation_units[14] * fwd_Ea[14] * invT);
    dlnkfdT = fwd_beta[14] * invT + activation_units[14] * fwd_Ea[14] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] -= q; /* E */
    wdot[1] -= q; /* O2 */
    wdot[9] += q; /* O2- */
    /* d()/d[E] */
    dqdci =  + k_f*sc[1]*sc[2];
    J[0] -= dqdci;                /* dwdot[E]/d[E] */
    J[1] -= dqdci;                /* dwdot[O2]/d[E] */
    J[9] += dqdci;                /* dwdot[O2-]/d[E] */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[0]*sc[2];
    J[11] -= dqdci;               /* dwdot[E]/d[O2] */
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[20] += dqdci;               /* dwdot[O2-]/d[O2] */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[0]*sc[1];
    J[22] -= dqdci;               /* dwdot[E]/d[N2] */
    J[23] -= dqdci;               /* dwdot[O2]/d[N2] */
    J[31] += dqdci;               /* dwdot[O2-]/d[N2] */
    /* d()/dT */
    J[110] -= dqdT;               /* dwdot[E]/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[119] += dqdT;               /* dwdot[O2-]/dT */

    /*reaction 16: O2- + O4+ => O2 + O2 + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[6]*sc[9];
    k_f = prefactor_units[15] * fwd_A[15]
                * exp(fwd_beta[15] * tc[0] - activation_units[15] * fwd_Ea[15] * invT);
    dlnkfdT = fwd_beta[15] * invT + activation_units[15] * fwd_Ea[15] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] += 3 * q; /* O2 */
    wdot[6] -= q; /* O4+ */
    wdot[9] -= q; /* O2- */
    /* d()/d[O4+] */
    dqdci =  + k_f*sc[9];
    J[67] += 3 * dqdci;           /* dwdot[O2]/d[O4+] */
    J[72] -= dqdci;               /* dwdot[O4+]/d[O4+] */
    J[75] -= dqdci;               /* dwdot[O2-]/d[O4+] */
    /* d()/d[O2-] */
    dqdci =  + k_f*sc[6];
    J[100] += 3 * dqdci;          /* dwdot[O2]/d[O2-] */
    J[105] -= dqdci;              /* dwdot[O4+]/d[O2-] */
    J[108] -= dqdci;              /* dwdot[O2-]/d[O2-] */
    /* d()/dT */
    J[111] += 3 * dqdT;           /* dwdot[O2]/dT */
    J[116] -= dqdT;               /* dwdot[O4+]/dT */
    J[119] -= dqdT;               /* dwdot[O2-]/dT */

    /*reaction 17: O2- + O4+ + O2 => O2 + O2 + O2 + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[6]*sc[9];
    k_f = prefactor_units[16] * fwd_A[16]
                * exp(fwd_beta[16] * tc[0] - activation_units[16] * fwd_Ea[16] * invT);
    dlnkfdT = fwd_beta[16] * invT + activation_units[16] * fwd_Ea[16] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] += 3 * q; /* O2 */
    wdot[6] -= q; /* O4+ */
    wdot[9] -= q; /* O2- */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[6]*sc[9];
    J[12] += 3 * dqdci;           /* dwdot[O2]/d[O2] */
    J[17] -= dqdci;               /* dwdot[O4+]/d[O2] */
    J[20] -= dqdci;               /* dwdot[O2-]/d[O2] */
    /* d()/d[O4+] */
    dqdci =  + k_f*sc[1]*sc[9];
    J[67] += 3 * dqdci;           /* dwdot[O2]/d[O4+] */
    J[72] -= dqdci;               /* dwdot[O4+]/d[O4+] */
    J[75] -= dqdci;               /* dwdot[O2-]/d[O4+] */
    /* d()/d[O2-] */
    dqdci =  + k_f*sc[1]*sc[6];
    J[100] += 3 * dqdci;          /* dwdot[O2]/d[O2-] */
    J[105] -= dqdci;              /* dwdot[O4+]/d[O2-] */
    J[108] -= dqdci;              /* dwdot[O2-]/d[O2-] */
    /* d()/dT */
    J[111] += 3 * dqdT;           /* dwdot[O2]/dT */
    J[116] -= dqdT;               /* dwdot[O4+]/dT */
    J[119] -= dqdT;               /* dwdot[O2-]/dT */

    /*reaction 18: O2- + O4+ + N2 => O2 + O2 + O2 + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[2]*sc[6]*sc[9];
    k_f = prefactor_units[17] * fwd_A[17]
                * exp(fwd_beta[17] * tc[0] - activation_units[17] * fwd_Ea[17] * invT);
    dlnkfdT = fwd_beta[17] * invT + activation_units[17] * fwd_Ea[17] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] += 3 * q; /* O2 */
    wdot[6] -= q; /* O4+ */
    wdot[9] -= q; /* O2- */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[6]*sc[9];
    J[23] += 3 * dqdci;           /* dwdot[O2]/d[N2] */
    J[28] -= dqdci;               /* dwdot[O4+]/d[N2] */
    J[31] -= dqdci;               /* dwdot[O2-]/d[N2] */
    /* d()/d[O4+] */
    dqdci =  + k_f*sc[2]*sc[9];
    J[67] += 3 * dqdci;           /* dwdot[O2]/d[O4+] */
    J[72] -= dqdci;               /* dwdot[O4+]/d[O4+] */
    J[75] -= dqdci;               /* dwdot[O2-]/d[O4+] */
    /* d()/d[O2-] */
    dqdci =  + k_f*sc[2]*sc[6];
    J[100] += 3 * dqdci;          /* dwdot[O2]/d[O2-] */
    J[105] -= dqdci;              /* dwdot[O4+]/d[O2-] */
    J[108] -= dqdci;              /* dwdot[O2-]/d[O2-] */
    /* d()/dT */
    J[111] += 3 * dqdT;           /* dwdot[O2]/dT */
    J[116] -= dqdT;               /* dwdot[O4+]/dT */
    J[119] -= dqdT;               /* dwdot[O2-]/dT */

    /*reaction 19: O2- + O2+ + O2 => O2 + O2 + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[4]*sc[9];
    k_f = prefactor_units[18] * fwd_A[18]
                * exp(fwd_beta[18] * tc[0] - activation_units[18] * fwd_Ea[18] * invT);
    dlnkfdT = fwd_beta[18] * invT + activation_units[18] * fwd_Ea[18] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] += 2 * q; /* O2 */
    wdot[4] -= q; /* O2+ */
    wdot[9] -= q; /* O2- */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[4]*sc[9];
    J[12] += 2 * dqdci;           /* dwdot[O2]/d[O2] */
    J[15] -= dqdci;               /* dwdot[O2+]/d[O2] */
    J[20] -= dqdci;               /* dwdot[O2-]/d[O2] */
    /* d()/d[O2+] */
    dqdci =  + k_f*sc[1]*sc[9];
    J[45] += 2 * dqdci;           /* dwdot[O2]/d[O2+] */
    J[48] -= dqdci;               /* dwdot[O2+]/d[O2+] */
    J[53] -= dqdci;               /* dwdot[O2-]/d[O2+] */
    /* d()/d[O2-] */
    dqdci =  + k_f*sc[1]*sc[4];
    J[100] += 2 * dqdci;          /* dwdot[O2]/d[O2-] */
    J[103] -= dqdci;              /* dwdot[O2+]/d[O2-] */
    J[108] -= dqdci;              /* dwdot[O2-]/d[O2-] */
    /* d()/dT */
    J[111] += 2 * dqdT;           /* dwdot[O2]/dT */
    J[114] -= dqdT;               /* dwdot[O2+]/dT */
    J[119] -= dqdT;               /* dwdot[O2-]/dT */

    /*reaction 20: O2- + O2+ + N2 => O2 + O2 + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[2]*sc[4]*sc[9];
    k_f = prefactor_units[19] * fwd_A[19]
                * exp(fwd_beta[19] * tc[0] - activation_units[19] * fwd_Ea[19] * invT);
    dlnkfdT = fwd_beta[19] * invT + activation_units[19] * fwd_Ea[19] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] += 2 * q; /* O2 */
    wdot[4] -= q; /* O2+ */
    wdot[9] -= q; /* O2- */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[4]*sc[9];
    J[23] += 2 * dqdci;           /* dwdot[O2]/d[N2] */
    J[26] -= dqdci;               /* dwdot[O2+]/d[N2] */
    J[31] -= dqdci;               /* dwdot[O2-]/d[N2] */
    /* d()/d[O2+] */
    dqdci =  + k_f*sc[2]*sc[9];
    J[45] += 2 * dqdci;           /* dwdot[O2]/d[O2+] */
    J[48] -= dqdci;               /* dwdot[O2+]/d[O2+] */
    J[53] -= dqdci;               /* dwdot[O2-]/d[O2+] */
    /* d()/d[O2-] */
    dqdci =  + k_f*sc[2]*sc[4];
    J[100] += 2 * dqdci;          /* dwdot[O2]/d[O2-] */
    J[103] -= dqdci;              /* dwdot[O2+]/d[O2-] */
    J[108] -= dqdci;              /* dwdot[O2-]/d[O2-] */
    /* d()/dT */
    J[111] += 2 * dqdT;           /* dwdot[O2]/dT */
    J[114] -= dqdT;               /* dwdot[O2+]/dT */
    J[119] -= dqdT;               /* dwdot[O2-]/dT */

    /*reaction 21: O2- + O2 => E + O2 + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[9];
    k_f = prefactor_units[20] * fwd_A[20]
                * exp(fwd_beta[20] * tc[0] - activation_units[20] * fwd_Ea[20] * invT);
    dlnkfdT = fwd_beta[20] * invT + activation_units[20] * fwd_Ea[20] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] += q; /* E */
    wdot[1] += q; /* O2 */
    wdot[9] -= q; /* O2- */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[9];
    J[11] += dqdci;               /* dwdot[E]/d[O2] */
    J[12] += dqdci;               /* dwdot[O2]/d[O2] */
    J[20] -= dqdci;               /* dwdot[O2-]/d[O2] */
    /* d()/d[O2-] */
    dqdci =  + k_f*sc[1];
    J[99] += dqdci;               /* dwdot[E]/d[O2-] */
    J[100] += dqdci;              /* dwdot[O2]/d[O2-] */
    J[108] -= dqdci;              /* dwdot[O2-]/d[O2-] */
    /* d()/dT */
    J[110] += dqdT;               /* dwdot[E]/dT */
    J[111] += dqdT;               /* dwdot[O2]/dT */
    J[119] -= dqdT;               /* dwdot[O2-]/dT */

    /*reaction 22: O2- + N2 => E + O2 + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[2]*sc[9];
    k_f = prefactor_units[21] * fwd_A[21]
                * exp(fwd_beta[21] * tc[0] - activation_units[21] * fwd_Ea[21] * invT);
    dlnkfdT = fwd_beta[21] * invT + activation_units[21] * fwd_Ea[21] * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] += q; /* E */
    wdot[1] += q; /* O2 */
    wdot[9] -= q; /* O2- */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[9];
    J[22] += dqdci;               /* dwdot[E]/d[N2] */
    J[23] += dqdci;               /* dwdot[O2]/d[N2] */
    J[31] -= dqdci;               /* dwdot[O2-]/d[N2] */
    /* d()/d[O2-] */
    dqdci =  + k_f*sc[2];
    J[99] += dqdci;               /* dwdot[E]/d[O2-] */
    J[100] += dqdci;              /* dwdot[O2]/d[O2-] */
    J[108] -= dqdci;              /* dwdot[O2-]/d[O2-] */
    /* d()/dT */
    J[110] += dqdT;               /* dwdot[E]/dT */
    J[111] += dqdT;               /* dwdot[O2]/dT */
    J[119] -= dqdT;               /* dwdot[O2-]/dT */

    double c_R[10], dcRdT[10], e_RT[10];
    double * eh_RT;
    if (consP) {
        cp_R(c_R, tc);
        dcvpRdT(dcRdT, tc);
        eh_RT = &h_RT[0];
    }
    else {
        cv_R(c_R, tc);
        dcvpRdT(dcRdT, tc);
        speciesInternalEnergy(e_RT, tc);
        eh_RT = &e_RT[0];
    }

    double cmix = 0.0, ehmix = 0.0, dcmixdT=0.0, dehmixdT=0.0;
    for (int k = 0; k < 10; ++k) {
        cmix += c_R[k]*sc[k];
        dcmixdT += dcRdT[k]*sc[k];
        ehmix += eh_RT[k]*wdot[k];
        dehmixdT += invT*(c_R[k]-eh_RT[k])*wdot[k] + eh_RT[k]*J[110+k];
    }

    double cmixinv = 1.0/cmix;
    double tmp1 = ehmix*cmixinv;
    double tmp3 = cmixinv*T;
    double tmp2 = tmp1*tmp3;
    double dehmixdc;
    /* dTdot/d[X] */
    for (int k = 0; k < 10; ++k) {
        dehmixdc = 0.0;
        for (int m = 0; m < 10; ++m) {
            dehmixdc += eh_RT[m]*J[k*11+m];
        }
        J[k*11+10] = tmp2*c_R[k] - tmp3*dehmixdc;
    }
    /* dTdot/dT */
    J[120] = -tmp1 + tmp2*dcmixdT - tmp3*dehmixdT;
}
#endif


/*compute an approx to the reaction Jacobian */
AMREX_GPU_HOST_DEVICE void aJacobian_precond(double *  J, double *  sc, double T, int HP)
{
    for (int i=0; i<121; i++) {
        J[i] = 0.0;
    }

    double wdot[10];
    for (int k=0; k<10; k++) {
        wdot[k] = 0.0;
    }

    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double invT = 1.0 / tc[1];
    double invT2 = invT * invT;

    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 / T;
    double refCinv = 1.0 / refC;

    /*compute the mixture concentration */
    double mixture = 0.0;
    for (int k = 0; k < 10; ++k) {
        mixture += sc[k];
    }

    /*compute the Gibbs free energy */
    double g_RT[10];
    gibbs(g_RT, tc);

    /*compute the species enthalpy */
    double h_RT[10];
    speciesEnthalpy(h_RT, tc);

    double phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
    double dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
    double dqdci, dcdc_fac, dqdc[10];
    double Pr, fPr, F, k_0, logPr;
    double logFcent, troe_c, troe_n, troePr_den, troePr, troe;
    double Fcent1, Fcent2, Fcent3, Fcent;
    double dlogFdc, dlogFdn, dlogFdcn_fac;
    double dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
    const double ln10 = log(10.0);
    const double log10e = 1.0/log(10.0);
    /*reaction 1: E + N2 => E + E + N2+ */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[0]*sc[2];
    k_f = 1.0000000000000002e-06 * 1
                * exp(0 * tc[0] - 0.50321666580471969 * (1) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (1)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] += q; /* E */
    wdot[2] -= q; /* N2 */
    wdot[5] += q; /* N2+ */
    /* d()/d[E] */
    dqdci =  + k_f*sc[2];
    J[0] += dqdci;                /* dwdot[E]/d[E] */
    J[2] -= dqdci;                /* dwdot[N2]/d[E] */
    J[5] += dqdci;                /* dwdot[N2+]/d[E] */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[0];
    J[22] += dqdci;               /* dwdot[E]/d[N2] */
    J[24] -= dqdci;               /* dwdot[N2]/d[N2] */
    J[27] += dqdci;               /* dwdot[N2+]/d[N2] */
    /* d()/dT */
    J[110] += dqdT;               /* dwdot[E]/dT */
    J[112] -= dqdT;               /* dwdot[N2]/dT */
    J[115] += dqdT;               /* dwdot[N2+]/dT */

    /*reaction 2: E + O2 => E + E + O2+ */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[0]*sc[1];
    k_f = 1.0000000000000002e-06 * 1
                * exp(0 * tc[0] - 0.50321666580471969 * (1) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (1)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] += q; /* E */
    wdot[1] -= q; /* O2 */
    wdot[4] += q; /* O2+ */
    /* d()/d[E] */
    dqdci =  + k_f*sc[1];
    J[0] += dqdci;                /* dwdot[E]/d[E] */
    J[1] -= dqdci;                /* dwdot[O2]/d[E] */
    J[4] += dqdci;                /* dwdot[O2+]/d[E] */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[0];
    J[11] += dqdci;               /* dwdot[E]/d[O2] */
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[15] += dqdci;               /* dwdot[O2+]/d[O2] */
    /* d()/dT */
    J[110] += dqdT;               /* dwdot[E]/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[114] += dqdT;               /* dwdot[O2+]/dT */

    /*reaction 3: N2+ + N2 + N2 => N4+ + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = pow(sc[2], 2.000000)*sc[5];
    k_f = 1.0000000000000002e-12 * 1.8132242e+19
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[2] -= q; /* N2 */
    wdot[5] -= q; /* N2+ */
    wdot[7] += q; /* N4+ */
    /* d()/d[N2] */
    dqdci =  + k_f*2.000000*sc[2]*sc[5];
    J[24] -= dqdci;               /* dwdot[N2]/d[N2] */
    J[27] -= dqdci;               /* dwdot[N2+]/d[N2] */
    J[29] += dqdci;               /* dwdot[N4+]/d[N2] */
    /* d()/d[N2+] */
    dqdci =  + k_f*pow(sc[2], 2.000000);
    J[57] -= dqdci;               /* dwdot[N2]/d[N2+] */
    J[60] -= dqdci;               /* dwdot[N2+]/d[N2+] */
    J[62] += dqdci;               /* dwdot[N4+]/d[N2+] */
    /* d()/dT */
    J[112] -= dqdT;               /* dwdot[N2]/dT */
    J[115] -= dqdT;               /* dwdot[N2+]/dT */
    J[117] += dqdT;               /* dwdot[N4+]/dT */

    /*reaction 4: N2+ + N2 + O2 => N4+ + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[2]*sc[5];
    k_f = 1.0000000000000002e-12 * 1.8132242e+19
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[2] -= q; /* N2 */
    wdot[5] -= q; /* N2+ */
    wdot[7] += q; /* N4+ */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[2]*sc[5];
    J[13] -= dqdci;               /* dwdot[N2]/d[O2] */
    J[16] -= dqdci;               /* dwdot[N2+]/d[O2] */
    J[18] += dqdci;               /* dwdot[N4+]/d[O2] */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[1]*sc[5];
    J[24] -= dqdci;               /* dwdot[N2]/d[N2] */
    J[27] -= dqdci;               /* dwdot[N2+]/d[N2] */
    J[29] += dqdci;               /* dwdot[N4+]/d[N2] */
    /* d()/d[N2+] */
    dqdci =  + k_f*sc[1]*sc[2];
    J[57] -= dqdci;               /* dwdot[N2]/d[N2+] */
    J[60] -= dqdci;               /* dwdot[N2+]/d[N2+] */
    J[62] += dqdci;               /* dwdot[N4+]/d[N2+] */
    /* d()/dT */
    J[112] -= dqdT;               /* dwdot[N2]/dT */
    J[115] -= dqdT;               /* dwdot[N2+]/dT */
    J[117] += dqdT;               /* dwdot[N4+]/dT */

    /*reaction 5: N4+ + O2 => O2+ + N2 + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[7];
    k_f = 1.0000000000000002e-06 * 150550000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] -= q; /* O2 */
    wdot[2] += 2 * q; /* N2 */
    wdot[4] += q; /* O2+ */
    wdot[7] -= q; /* N4+ */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[7];
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[13] += 2 * dqdci;           /* dwdot[N2]/d[O2] */
    J[15] += dqdci;               /* dwdot[O2+]/d[O2] */
    J[18] -= dqdci;               /* dwdot[N4+]/d[O2] */
    /* d()/d[N4+] */
    dqdci =  + k_f*sc[1];
    J[78] -= dqdci;               /* dwdot[O2]/d[N4+] */
    J[79] += 2 * dqdci;           /* dwdot[N2]/d[N4+] */
    J[81] += dqdci;               /* dwdot[O2+]/d[N4+] */
    J[84] -= dqdci;               /* dwdot[N4+]/d[N4+] */
    /* d()/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[112] += 2 * dqdT;           /* dwdot[N2]/dT */
    J[114] += dqdT;               /* dwdot[O2+]/dT */
    J[117] -= dqdT;               /* dwdot[N4+]/dT */

    /*reaction 6: N2+ + O2 => O2+ + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[5];
    k_f = 1.0000000000000002e-06 * 36132000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] -= q; /* O2 */
    wdot[2] += q; /* N2 */
    wdot[4] += q; /* O2+ */
    wdot[5] -= q; /* N2+ */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[5];
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[13] += dqdci;               /* dwdot[N2]/d[O2] */
    J[15] += dqdci;               /* dwdot[O2+]/d[O2] */
    J[16] -= dqdci;               /* dwdot[N2+]/d[O2] */
    /* d()/d[N2+] */
    dqdci =  + k_f*sc[1];
    J[56] -= dqdci;               /* dwdot[O2]/d[N2+] */
    J[57] += dqdci;               /* dwdot[N2]/d[N2+] */
    J[59] += dqdci;               /* dwdot[O2+]/d[N2+] */
    J[60] -= dqdci;               /* dwdot[N2+]/d[N2+] */
    /* d()/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[112] += dqdT;               /* dwdot[N2]/dT */
    J[114] += dqdT;               /* dwdot[O2+]/dT */
    J[115] -= dqdT;               /* dwdot[N2+]/dT */

    /*reaction 7: O2+ + N2 + N2 => O2pN2 + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = pow(sc[2], 2.000000)*sc[4];
    k_f = 1.0000000000000002e-12 * 3.26380356e+17
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[2] -= q; /* N2 */
    wdot[4] -= q; /* O2+ */
    wdot[8] += q; /* O2pN2 */
    /* d()/d[N2] */
    dqdci =  + k_f*2.000000*sc[2]*sc[4];
    J[24] -= dqdci;               /* dwdot[N2]/d[N2] */
    J[26] -= dqdci;               /* dwdot[O2+]/d[N2] */
    J[30] += dqdci;               /* dwdot[O2pN2]/d[N2] */
    /* d()/d[O2+] */
    dqdci =  + k_f*pow(sc[2], 2.000000);
    J[46] -= dqdci;               /* dwdot[N2]/d[O2+] */
    J[48] -= dqdci;               /* dwdot[O2+]/d[O2+] */
    J[52] += dqdci;               /* dwdot[O2pN2]/d[O2+] */
    /* d()/dT */
    J[112] -= dqdT;               /* dwdot[N2]/dT */
    J[114] -= dqdT;               /* dwdot[O2+]/dT */
    J[118] += dqdT;               /* dwdot[O2pN2]/dT */

    /*reaction 8: O2pN2 + N2 => O2+ + N2 + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[2]*sc[8];
    k_f = 1.0000000000000002e-06 * 258946000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[2] += q; /* N2 */
    wdot[4] += q; /* O2+ */
    wdot[8] -= q; /* O2pN2 */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[8];
    J[24] += dqdci;               /* dwdot[N2]/d[N2] */
    J[26] += dqdci;               /* dwdot[O2+]/d[N2] */
    J[30] -= dqdci;               /* dwdot[O2pN2]/d[N2] */
    /* d()/d[O2pN2] */
    dqdci =  + k_f*sc[2];
    J[90] += dqdci;               /* dwdot[N2]/d[O2pN2] */
    J[92] += dqdci;               /* dwdot[O2+]/d[O2pN2] */
    J[96] -= dqdci;               /* dwdot[O2pN2]/d[O2pN2] */
    /* d()/dT */
    J[112] += dqdT;               /* dwdot[N2]/dT */
    J[114] += dqdT;               /* dwdot[O2+]/dT */
    J[118] -= dqdT;               /* dwdot[O2pN2]/dT */

    /*reaction 9: O2pN2 + O2 => O4+ + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[8];
    k_f = 1.0000000000000002e-06 * 602200000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] -= q; /* O2 */
    wdot[2] += q; /* N2 */
    wdot[6] += q; /* O4+ */
    wdot[8] -= q; /* O2pN2 */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[8];
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[13] += dqdci;               /* dwdot[N2]/d[O2] */
    J[17] += dqdci;               /* dwdot[O4+]/d[O2] */
    J[19] -= dqdci;               /* dwdot[O2pN2]/d[O2] */
    /* d()/d[O2pN2] */
    dqdci =  + k_f*sc[1];
    J[89] -= dqdci;               /* dwdot[O2]/d[O2pN2] */
    J[90] += dqdci;               /* dwdot[N2]/d[O2pN2] */
    J[94] += dqdci;               /* dwdot[O4+]/d[O2pN2] */
    J[96] -= dqdci;               /* dwdot[O2pN2]/d[O2pN2] */
    /* d()/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[112] += dqdT;               /* dwdot[N2]/dT */
    J[116] += dqdT;               /* dwdot[O4+]/dT */
    J[118] -= dqdT;               /* dwdot[O2pN2]/dT */

    /*reaction 10: O2+ + O2 + N2 => O4+ + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[2]*sc[4];
    k_f = 1.0000000000000002e-12 * 8.70347616e+17
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] -= q; /* O2 */
    wdot[4] -= q; /* O2+ */
    wdot[6] += q; /* O4+ */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[2]*sc[4];
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[15] -= dqdci;               /* dwdot[O2+]/d[O2] */
    J[17] += dqdci;               /* dwdot[O4+]/d[O2] */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[1]*sc[4];
    J[23] -= dqdci;               /* dwdot[O2]/d[N2] */
    J[26] -= dqdci;               /* dwdot[O2+]/d[N2] */
    J[28] += dqdci;               /* dwdot[O4+]/d[N2] */
    /* d()/d[O2+] */
    dqdci =  + k_f*sc[1]*sc[2];
    J[45] -= dqdci;               /* dwdot[O2]/d[O2+] */
    J[48] -= dqdci;               /* dwdot[O2+]/d[O2+] */
    J[50] += dqdci;               /* dwdot[O4+]/d[O2+] */
    /* d()/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[114] -= dqdT;               /* dwdot[O2+]/dT */
    J[116] += dqdT;               /* dwdot[O4+]/dT */

    /*reaction 11: O2+ + O2 + O2 => O4+ + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = pow(sc[1], 2.000000)*sc[4];
    k_f = 1.0000000000000002e-12 * 8.70347616e+17
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] -= q; /* O2 */
    wdot[4] -= q; /* O2+ */
    wdot[6] += q; /* O4+ */
    /* d()/d[O2] */
    dqdci =  + k_f*2.000000*sc[1]*sc[4];
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[15] -= dqdci;               /* dwdot[O2+]/d[O2] */
    J[17] += dqdci;               /* dwdot[O4+]/d[O2] */
    /* d()/d[O2+] */
    dqdci =  + k_f*pow(sc[1], 2.000000);
    J[45] -= dqdci;               /* dwdot[O2]/d[O2+] */
    J[48] -= dqdci;               /* dwdot[O2+]/d[O2+] */
    J[50] += dqdci;               /* dwdot[O4+]/d[O2+] */
    /* d()/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[114] -= dqdT;               /* dwdot[O2+]/dT */
    J[116] += dqdT;               /* dwdot[O4+]/dT */

    /*reaction 12: E + O4+ => O2 + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[0]*sc[6];
    k_f = 1.0000000000000002e-06 * 1
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] -= q; /* E */
    wdot[1] += 2 * q; /* O2 */
    wdot[6] -= q; /* O4+ */
    /* d()/d[E] */
    dqdci =  + k_f*sc[6];
    J[0] -= dqdci;                /* dwdot[E]/d[E] */
    J[1] += 2 * dqdci;            /* dwdot[O2]/d[E] */
    J[6] -= dqdci;                /* dwdot[O4+]/d[E] */
    /* d()/d[O4+] */
    dqdci =  + k_f*sc[0];
    J[66] -= dqdci;               /* dwdot[E]/d[O4+] */
    J[67] += 2 * dqdci;           /* dwdot[O2]/d[O4+] */
    J[72] -= dqdci;               /* dwdot[O4+]/d[O4+] */
    /* d()/dT */
    J[110] -= dqdT;               /* dwdot[E]/dT */
    J[111] += 2 * dqdT;           /* dwdot[O2]/dT */
    J[116] -= dqdT;               /* dwdot[O4+]/dT */

    /*reaction 13: E + O2+ => O + O */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[0]*sc[4];
    k_f = 1.0000000000000002e-06 * 1
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] -= q; /* E */
    wdot[3] += 2 * q; /* O */
    wdot[4] -= q; /* O2+ */
    /* d()/d[E] */
    dqdci =  + k_f*sc[4];
    J[0] -= dqdci;                /* dwdot[E]/d[E] */
    J[3] += 2 * dqdci;            /* dwdot[O]/d[E] */
    J[4] -= dqdci;                /* dwdot[O2+]/d[E] */
    /* d()/d[O2+] */
    dqdci =  + k_f*sc[0];
    J[44] -= dqdci;               /* dwdot[E]/d[O2+] */
    J[47] += 2 * dqdci;           /* dwdot[O]/d[O2+] */
    J[48] -= dqdci;               /* dwdot[O2+]/d[O2+] */
    /* d()/dT */
    J[110] -= dqdT;               /* dwdot[E]/dT */
    J[113] += 2 * dqdT;           /* dwdot[O]/dT */
    J[114] -= dqdT;               /* dwdot[O2+]/dT */

    /*reaction 14: E + O2 + O2 => O2- + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[0]*pow(sc[1], 2.000000);
    k_f = 1.0000000000000002e-12 * 1
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] -= q; /* E */
    wdot[1] -= q; /* O2 */
    wdot[9] += q; /* O2- */
    /* d()/d[E] */
    dqdci =  + k_f*pow(sc[1], 2.000000);
    J[0] -= dqdci;                /* dwdot[E]/d[E] */
    J[1] -= dqdci;                /* dwdot[O2]/d[E] */
    J[9] += dqdci;                /* dwdot[O2-]/d[E] */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[0]*2.000000*sc[1];
    J[11] -= dqdci;               /* dwdot[E]/d[O2] */
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[20] += dqdci;               /* dwdot[O2-]/d[O2] */
    /* d()/dT */
    J[110] -= dqdT;               /* dwdot[E]/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[119] += dqdT;               /* dwdot[O2-]/dT */

    /*reaction 15: E + O2 + N2 => O2- + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[0]*sc[1]*sc[2];
    k_f = 1.0000000000000002e-12 * 1
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] -= q; /* E */
    wdot[1] -= q; /* O2 */
    wdot[9] += q; /* O2- */
    /* d()/d[E] */
    dqdci =  + k_f*sc[1]*sc[2];
    J[0] -= dqdci;                /* dwdot[E]/d[E] */
    J[1] -= dqdci;                /* dwdot[O2]/d[E] */
    J[9] += dqdci;                /* dwdot[O2-]/d[E] */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[0]*sc[2];
    J[11] -= dqdci;               /* dwdot[E]/d[O2] */
    J[12] -= dqdci;               /* dwdot[O2]/d[O2] */
    J[20] += dqdci;               /* dwdot[O2-]/d[O2] */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[0]*sc[1];
    J[22] -= dqdci;               /* dwdot[E]/d[N2] */
    J[23] -= dqdci;               /* dwdot[O2]/d[N2] */
    J[31] += dqdci;               /* dwdot[O2-]/d[N2] */
    /* d()/dT */
    J[110] -= dqdT;               /* dwdot[E]/dT */
    J[111] -= dqdT;               /* dwdot[O2]/dT */
    J[119] += dqdT;               /* dwdot[O2-]/dT */

    /*reaction 16: O2- + O4+ => O2 + O2 + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[6]*sc[9];
    k_f = 1.0000000000000002e-06 * 60220000000000000
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] += 3 * q; /* O2 */
    wdot[6] -= q; /* O4+ */
    wdot[9] -= q; /* O2- */
    /* d()/d[O4+] */
    dqdci =  + k_f*sc[9];
    J[67] += 3 * dqdci;           /* dwdot[O2]/d[O4+] */
    J[72] -= dqdci;               /* dwdot[O4+]/d[O4+] */
    J[75] -= dqdci;               /* dwdot[O2-]/d[O4+] */
    /* d()/d[O2-] */
    dqdci =  + k_f*sc[6];
    J[100] += 3 * dqdci;          /* dwdot[O2]/d[O2-] */
    J[105] -= dqdci;              /* dwdot[O4+]/d[O2-] */
    J[108] -= dqdci;              /* dwdot[O2-]/d[O2-] */
    /* d()/dT */
    J[111] += 3 * dqdT;           /* dwdot[O2]/dT */
    J[116] -= dqdT;               /* dwdot[O4+]/dT */
    J[119] -= dqdT;               /* dwdot[O2-]/dT */

    /*reaction 17: O2- + O4+ + O2 => O2 + O2 + O2 + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[6]*sc[9];
    k_f = 1.0000000000000002e-12 * 7.2528968000000003e+22
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] += 3 * q; /* O2 */
    wdot[6] -= q; /* O4+ */
    wdot[9] -= q; /* O2- */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[6]*sc[9];
    J[12] += 3 * dqdci;           /* dwdot[O2]/d[O2] */
    J[17] -= dqdci;               /* dwdot[O4+]/d[O2] */
    J[20] -= dqdci;               /* dwdot[O2-]/d[O2] */
    /* d()/d[O4+] */
    dqdci =  + k_f*sc[1]*sc[9];
    J[67] += 3 * dqdci;           /* dwdot[O2]/d[O4+] */
    J[72] -= dqdci;               /* dwdot[O4+]/d[O4+] */
    J[75] -= dqdci;               /* dwdot[O2-]/d[O4+] */
    /* d()/d[O2-] */
    dqdci =  + k_f*sc[1]*sc[6];
    J[100] += 3 * dqdci;          /* dwdot[O2]/d[O2-] */
    J[105] -= dqdci;              /* dwdot[O4+]/d[O2-] */
    J[108] -= dqdci;              /* dwdot[O2-]/d[O2-] */
    /* d()/dT */
    J[111] += 3 * dqdT;           /* dwdot[O2]/dT */
    J[116] -= dqdT;               /* dwdot[O4+]/dT */
    J[119] -= dqdT;               /* dwdot[O2-]/dT */

    /*reaction 18: O2- + O4+ + N2 => O2 + O2 + O2 + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[2]*sc[6]*sc[9];
    k_f = 1.0000000000000002e-12 * 7.2528968000000003e+22
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] += 3 * q; /* O2 */
    wdot[6] -= q; /* O4+ */
    wdot[9] -= q; /* O2- */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[6]*sc[9];
    J[23] += 3 * dqdci;           /* dwdot[O2]/d[N2] */
    J[28] -= dqdci;               /* dwdot[O4+]/d[N2] */
    J[31] -= dqdci;               /* dwdot[O2-]/d[N2] */
    /* d()/d[O4+] */
    dqdci =  + k_f*sc[2]*sc[9];
    J[67] += 3 * dqdci;           /* dwdot[O2]/d[O4+] */
    J[72] -= dqdci;               /* dwdot[O4+]/d[O4+] */
    J[75] -= dqdci;               /* dwdot[O2-]/d[O4+] */
    /* d()/d[O2-] */
    dqdci =  + k_f*sc[2]*sc[6];
    J[100] += 3 * dqdci;          /* dwdot[O2]/d[O2-] */
    J[105] -= dqdci;              /* dwdot[O4+]/d[O2-] */
    J[108] -= dqdci;              /* dwdot[O2-]/d[O2-] */
    /* d()/dT */
    J[111] += 3 * dqdT;           /* dwdot[O2]/dT */
    J[116] -= dqdT;               /* dwdot[O4+]/dT */
    J[119] -= dqdT;               /* dwdot[O2-]/dT */

    /*reaction 19: O2- + O2+ + O2 => O2 + O2 + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[4]*sc[9];
    k_f = 1.0000000000000002e-12 * 7.2528968000000003e+22
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] += 2 * q; /* O2 */
    wdot[4] -= q; /* O2+ */
    wdot[9] -= q; /* O2- */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[4]*sc[9];
    J[12] += 2 * dqdci;           /* dwdot[O2]/d[O2] */
    J[15] -= dqdci;               /* dwdot[O2+]/d[O2] */
    J[20] -= dqdci;               /* dwdot[O2-]/d[O2] */
    /* d()/d[O2+] */
    dqdci =  + k_f*sc[1]*sc[9];
    J[45] += 2 * dqdci;           /* dwdot[O2]/d[O2+] */
    J[48] -= dqdci;               /* dwdot[O2+]/d[O2+] */
    J[53] -= dqdci;               /* dwdot[O2-]/d[O2+] */
    /* d()/d[O2-] */
    dqdci =  + k_f*sc[1]*sc[4];
    J[100] += 2 * dqdci;          /* dwdot[O2]/d[O2-] */
    J[103] -= dqdci;              /* dwdot[O2+]/d[O2-] */
    J[108] -= dqdci;              /* dwdot[O2-]/d[O2-] */
    /* d()/dT */
    J[111] += 2 * dqdT;           /* dwdot[O2]/dT */
    J[114] -= dqdT;               /* dwdot[O2+]/dT */
    J[119] -= dqdT;               /* dwdot[O2-]/dT */

    /*reaction 20: O2- + O2+ + N2 => O2 + O2 + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[2]*sc[4]*sc[9];
    k_f = 1.0000000000000002e-12 * 7.2528968000000003e+22
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[1] += 2 * q; /* O2 */
    wdot[4] -= q; /* O2+ */
    wdot[9] -= q; /* O2- */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[4]*sc[9];
    J[23] += 2 * dqdci;           /* dwdot[O2]/d[N2] */
    J[26] -= dqdci;               /* dwdot[O2+]/d[N2] */
    J[31] -= dqdci;               /* dwdot[O2-]/d[N2] */
    /* d()/d[O2+] */
    dqdci =  + k_f*sc[2]*sc[9];
    J[45] += 2 * dqdci;           /* dwdot[O2]/d[O2+] */
    J[48] -= dqdci;               /* dwdot[O2+]/d[O2+] */
    J[53] -= dqdci;               /* dwdot[O2-]/d[O2+] */
    /* d()/d[O2-] */
    dqdci =  + k_f*sc[2]*sc[4];
    J[100] += 2 * dqdci;          /* dwdot[O2]/d[O2-] */
    J[103] -= dqdci;              /* dwdot[O2+]/d[O2-] */
    J[108] -= dqdci;              /* dwdot[O2-]/d[O2-] */
    /* d()/dT */
    J[111] += 2 * dqdT;           /* dwdot[O2]/dT */
    J[114] -= dqdT;               /* dwdot[O2+]/dT */
    J[119] -= dqdT;               /* dwdot[O2-]/dT */

    /*reaction 21: O2- + O2 => E + O2 + O2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[1]*sc[9];
    k_f = 1.0000000000000002e-06 * 1
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] += q; /* E */
    wdot[1] += q; /* O2 */
    wdot[9] -= q; /* O2- */
    /* d()/d[O2] */
    dqdci =  + k_f*sc[9];
    J[11] += dqdci;               /* dwdot[E]/d[O2] */
    J[12] += dqdci;               /* dwdot[O2]/d[O2] */
    J[20] -= dqdci;               /* dwdot[O2-]/d[O2] */
    /* d()/d[O2-] */
    dqdci =  + k_f*sc[1];
    J[99] += dqdci;               /* dwdot[E]/d[O2-] */
    J[100] += dqdci;              /* dwdot[O2]/d[O2-] */
    J[108] -= dqdci;              /* dwdot[O2-]/d[O2-] */
    /* d()/dT */
    J[110] += dqdT;               /* dwdot[E]/dT */
    J[111] += dqdT;               /* dwdot[O2]/dT */
    J[119] -= dqdT;               /* dwdot[O2-]/dT */

    /*reaction 22: O2- + N2 => E + O2 + N2 */
    /*a non-third-body and non-pressure-fall-off reaction */
    /* forward */
    phi_f = sc[2]*sc[9];
    k_f = 1.0000000000000002e-06 * 1
                * exp(0 * tc[0] - 0.50321666580471969 * (0) * invT);
    dlnkfdT = 0 * invT + 0.50321666580471969 *  (0)  * invT2;
    /* rate of progress */
    q = k_f*phi_f;
    dqdT = dlnkfdT*k_f*phi_f;
    /* update wdot */
    wdot[0] += q; /* E */
    wdot[1] += q; /* O2 */
    wdot[9] -= q; /* O2- */
    /* d()/d[N2] */
    dqdci =  + k_f*sc[9];
    J[22] += dqdci;               /* dwdot[E]/d[N2] */
    J[23] += dqdci;               /* dwdot[O2]/d[N2] */
    J[31] -= dqdci;               /* dwdot[O2-]/d[N2] */
    /* d()/d[O2-] */
    dqdci =  + k_f*sc[2];
    J[99] += dqdci;               /* dwdot[E]/d[O2-] */
    J[100] += dqdci;              /* dwdot[O2]/d[O2-] */
    J[108] -= dqdci;              /* dwdot[O2-]/d[O2-] */
    /* d()/dT */
    J[110] += dqdT;               /* dwdot[E]/dT */
    J[111] += dqdT;               /* dwdot[O2]/dT */
    J[119] -= dqdT;               /* dwdot[O2-]/dT */

    double c_R[10], dcRdT[10], e_RT[10];
    double * eh_RT;
    if (HP) {
        cp_R(c_R, tc);
        dcvpRdT(dcRdT, tc);
        eh_RT = &h_RT[0];
    }
    else {
        cv_R(c_R, tc);
        dcvpRdT(dcRdT, tc);
        speciesInternalEnergy(e_RT, tc);
        eh_RT = &e_RT[0];
    }

    double cmix = 0.0, ehmix = 0.0, dcmixdT=0.0, dehmixdT=0.0;
    for (int k = 0; k < 10; ++k) {
        cmix += c_R[k]*sc[k];
        dcmixdT += dcRdT[k]*sc[k];
        ehmix += eh_RT[k]*wdot[k];
        dehmixdT += invT*(c_R[k]-eh_RT[k])*wdot[k] + eh_RT[k]*J[110+k];
    }

    double cmixinv = 1.0/cmix;
    double tmp1 = ehmix*cmixinv;
    double tmp3 = cmixinv*T;
    double tmp2 = tmp1*tmp3;
    double dehmixdc;
    /* dTdot/d[X] */
    for (int k = 0; k < 10; ++k) {
        dehmixdc = 0.0;
        for (int m = 0; m < 10; ++m) {
            dehmixdc += eh_RT[m]*J[k*11+m];
        }
        J[k*11+10] = tmp2*c_R[k] - tmp3*dehmixdc;
    }
    /* dTdot/dT */
    J[120] = -tmp1 + tmp2*dcmixdT - tmp3*dehmixdT;
}


/*compute d(Cp/R)/dT and d(Cv/R)/dT at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void dcvpRdT(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: E */
        species[0] =
            +0.00000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3];
        /*species 1: O2 */
        species[1] =
            +1.12748635e-03
            -1.15123009e-06 * tc[1]
            +3.94163169e-09 * tc[2]
            -3.50742157e-12 * tc[3];
        /*species 2: N2 */
        species[2] =
            +1.40824000e-03
            -7.92644400e-06 * tc[1]
            +1.69245450e-08 * tc[2]
            -9.77942000e-12 * tc[3];
        /*species 3: O */
        species[3] =
            -1.63816649e-03
            +4.84206340e-06 * tc[1]
            -4.80852957e-09 * tc[2]
            +1.55627854e-12 * tc[3];
        /*species 4: O2+ */
        species[4] =
            -6.35951952e-03
            +2.84851248e-05 * tc[1]
            -3.62993769e-08 * tc[2]
            +1.48382751e-11 * tc[3];
        /*species 5: N2+ */
        species[5] =
            -2.06459157e-03
            +9.51504602e-06 * tc[1]
            -9.46992684e-09 * tc[2]
            +2.68203989e-12 * tc[3];
        /*species 6: O4+ */
        species[6] =
            +2.15570328e-02
            -4.20541878e-05 * tc[1]
            +2.46342243e-08 * tc[2]
            -2.59904392e-12 * tc[3];
        /*species 7: N4+ */
        species[7] =
            +1.43909181e-02
            -3.13594506e-05 * tc[1]
            +2.85686354e-08 * tc[2]
            -1.01426659e-11 * tc[3];
        /*species 8: O2pN2 */
        species[8] =
            +2.15570328e-02
            -4.20541878e-05 * tc[1]
            +2.46342243e-08 * tc[2]
            -2.59904392e-12 * tc[3];
        /*species 9: O2- */
        species[9] =
            -9.28741138e-04
            +1.29095416e-05 * tc[1]
            -2.32411014e-08 * tc[2]
            +1.17333065e-11 * tc[3];
    } else {
        /*species 0: E */
        species[0] =
            +0.00000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3];
        /*species 1: O2 */
        species[1] =
            +6.13519689e-04
            -2.51768398e-07 * tc[1]
            +5.32584444e-11 * tc[2]
            -4.54574124e-15 * tc[3];
        /*species 2: N2 */
        species[2] =
            +1.48797700e-03
            -1.13695220e-06 * tc[1]
            +3.02911200e-10 * tc[2]
            -2.70134040e-14 * tc[3];
        /*species 3: O */
        species[3] =
            -2.75506191e-05
            -6.20560670e-09 * tc[1]
            +1.36532023e-11 * tc[2]
            -1.74722060e-15 * tc[3];
        /*species 4: O2+ */
        species[4] =
            +1.11522244e-03
            -7.66985112e-07 * tc[1]
            +1.71835406e-10 * tc[2]
            -1.11059352e-14 * tc[3];
        /*species 5: N2+ */
        species[5] =
            +2.53071949e-04
            +3.69556428e-07 * tc[1]
            -1.36577167e-10 * tc[2]
            +1.30727212e-14 * tc[3];
        /*species 6: O4+ */
        species[6] =
            +2.48052052e-03
            -1.94226686e-06 * tc[1]
            +4.97214753e-10 * tc[2]
            -4.12190960e-14 * tc[3];
        /*species 7: N4+ */
        species[7] =
            +2.91094469e-03
            -2.23147998e-06 * tc[1]
            +5.63651832e-10 * tc[2]
            -4.63076052e-14 * tc[3];
        /*species 8: O2pN2 */
        species[8] =
            +2.48052052e-03
            -1.94226686e-06 * tc[1]
            +4.97214753e-10 * tc[2]
            -4.12190960e-14 * tc[3];
        /*species 9: O2- */
        species[9] =
            +5.98141823e-04
            -4.24267810e-07 * tc[1]
            +1.08980274e-10 * tc[2]
            -8.99956912e-15 * tc[3];
    }
    return;
}


/*compute the progress rate for each reaction */
AMREX_GPU_HOST_DEVICE void progressRate(double *  qdot, double *  sc, double T)
{
    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double invT = 1.0 / tc[1];

#ifndef AMREX_USE_CUDA
    if (T != T_save)
    {
        T_save = T;
        comp_k_f(tc,invT,k_f_save);
        comp_Kc(tc,invT,Kc_save);
    }
#endif

    double q_f[22], q_r[22];
    comp_qfqr(q_f, q_r, sc, tc, invT);

    for (int i = 0; i < 22; ++i) {
        qdot[i] = q_f[i] - q_r[i];
    }

    return;
}


/*compute the progress rate for each reaction */
AMREX_GPU_HOST_DEVICE void progressRateFR(double *  q_f, double *  q_r, double *  sc, double T)
{
    double tc[] = { log(T), T, T*T, T*T*T, T*T*T*T }; /*temperature cache */
    double invT = 1.0 / tc[1];
#ifndef AMREX_USE_CUDA

    if (T != T_save)
    {
        T_save = T;
        comp_k_f(tc,invT,k_f_save);
        comp_Kc(tc,invT,Kc_save);
    }
#endif

    comp_qfqr(q_f, q_r, sc, tc, invT);

    return;
}


/*compute the equilibrium constants for each reaction */
void equilibriumConstants(double *  kc, double *  g_RT, double T)
{
    /*reference concentration: P_atm / (RT) in inverse mol/m^3 */
    double refC = 101325 / 8.31446 / T;

    /*reaction 1: E + N2 => E + E + N2+ */
    kc[0] = refC * exp((g_RT[0] + g_RT[2]) - (g_RT[0] + g_RT[0] + g_RT[5]));

    /*reaction 2: E + O2 => E + E + O2+ */
    kc[1] = refC * exp((g_RT[0] + g_RT[1]) - (g_RT[0] + g_RT[0] + g_RT[4]));

    /*reaction 3: N2+ + N2 + N2 => N4+ + N2 */
    kc[2] = 1.0 / (refC) * exp((g_RT[5] + g_RT[2] + g_RT[2]) - (g_RT[7] + g_RT[2]));

    /*reaction 4: N2+ + N2 + O2 => N4+ + O2 */
    kc[3] = 1.0 / (refC) * exp((g_RT[5] + g_RT[2] + g_RT[1]) - (g_RT[7] + g_RT[1]));

    /*reaction 5: N4+ + O2 => O2+ + N2 + N2 */
    kc[4] = refC * exp((g_RT[7] + g_RT[1]) - (g_RT[4] + g_RT[2] + g_RT[2]));

    /*reaction 6: N2+ + O2 => O2+ + N2 */
    kc[5] = exp((g_RT[5] + g_RT[1]) - (g_RT[4] + g_RT[2]));

    /*reaction 7: O2+ + N2 + N2 => O2pN2 + N2 */
    kc[6] = 1.0 / (refC) * exp((g_RT[4] + g_RT[2] + g_RT[2]) - (g_RT[8] + g_RT[2]));

    /*reaction 8: O2pN2 + N2 => O2+ + N2 + N2 */
    kc[7] = refC * exp((g_RT[8] + g_RT[2]) - (g_RT[4] + g_RT[2] + g_RT[2]));

    /*reaction 9: O2pN2 + O2 => O4+ + N2 */
    kc[8] = exp((g_RT[8] + g_RT[1]) - (g_RT[6] + g_RT[2]));

    /*reaction 10: O2+ + O2 + N2 => O4+ + N2 */
    kc[9] = 1.0 / (refC) * exp((g_RT[4] + g_RT[1] + g_RT[2]) - (g_RT[6] + g_RT[2]));

    /*reaction 11: O2+ + O2 + O2 => O4+ + O2 */
    kc[10] = 1.0 / (refC) * exp((g_RT[4] + g_RT[1] + g_RT[1]) - (g_RT[6] + g_RT[1]));

    /*reaction 12: E + O4+ => O2 + O2 */
    kc[11] = exp((g_RT[0] + g_RT[6]) - (g_RT[1] + g_RT[1]));

    /*reaction 13: E + O2+ => O + O */
    kc[12] = exp((g_RT[0] + g_RT[4]) - (g_RT[3] + g_RT[3]));

    /*reaction 14: E + O2 + O2 => O2- + O2 */
    kc[13] = 1.0 / (refC) * exp((g_RT[0] + g_RT[1] + g_RT[1]) - (g_RT[9] + g_RT[1]));

    /*reaction 15: E + O2 + N2 => O2- + N2 */
    kc[14] = 1.0 / (refC) * exp((g_RT[0] + g_RT[1] + g_RT[2]) - (g_RT[9] + g_RT[2]));

    /*reaction 16: O2- + O4+ => O2 + O2 + O2 */
    kc[15] = refC * exp((g_RT[9] + g_RT[6]) - (g_RT[1] + g_RT[1] + g_RT[1]));

    /*reaction 17: O2- + O4+ + O2 => O2 + O2 + O2 + O2 */
    kc[16] = refC * exp((g_RT[9] + g_RT[6] + g_RT[1]) - (g_RT[1] + g_RT[1] + g_RT[1] + g_RT[1]));

    /*reaction 18: O2- + O4+ + N2 => O2 + O2 + O2 + N2 */
    kc[17] = refC * exp((g_RT[9] + g_RT[6] + g_RT[2]) - (g_RT[1] + g_RT[1] + g_RT[1] + g_RT[2]));

    /*reaction 19: O2- + O2+ + O2 => O2 + O2 + O2 */
    kc[18] = exp((g_RT[9] + g_RT[4] + g_RT[1]) - (g_RT[1] + g_RT[1] + g_RT[1]));

    /*reaction 20: O2- + O2+ + N2 => O2 + O2 + N2 */
    kc[19] = exp((g_RT[9] + g_RT[4] + g_RT[2]) - (g_RT[1] + g_RT[1] + g_RT[2]));

    /*reaction 21: O2- + O2 => E + O2 + O2 */
    kc[20] = refC * exp((g_RT[9] + g_RT[1]) - (g_RT[0] + g_RT[1] + g_RT[1]));

    /*reaction 22: O2- + N2 => E + O2 + N2 */
    kc[21] = refC * exp((g_RT[9] + g_RT[2]) - (g_RT[0] + g_RT[1] + g_RT[2]));

    return;
}


/*compute the g/(RT) at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void gibbs(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    double invT = 1 / T;

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: E */
        species[0] =
            -7.453750000000000e+02 * invT
            +1.422081220000000e+01
            -2.500000000000000e+00 * tc[0]
            -0.000000000000000e+00 * tc[1]
            -0.000000000000000e+00 * tc[2]
            -0.000000000000000e+00 * tc[3]
            -0.000000000000000e+00 * tc[4];
        /*species 1: O2 */
        species[1] =
            -1.005249020000000e+03 * invT
            -2.821801190000000e+00
            -3.212936400000000e+00 * tc[0]
            -5.637431750000000e-04 * tc[1]
            +9.593584116666666e-08 * tc[2]
            -1.094897691666667e-10 * tc[3]
            +4.384276960000000e-14 * tc[4];
        /*species 2: N2 */
        species[2] =
            -1.020900000000000e+03 * invT
            -6.516950000000001e-01
            -3.298677000000000e+00 * tc[0]
            -7.041200000000000e-04 * tc[1]
            +6.605369999999999e-07 * tc[2]
            -4.701262500000001e-10 * tc[3]
            +1.222427500000000e-13 * tc[4];
        /*species 3: O */
        species[3] =
            +2.914764450000000e+04 * invT
            -1.756619999999964e-02
            -2.946428780000000e+00 * tc[0]
            +8.190832450000000e-04 * tc[1]
            -4.035052833333333e-07 * tc[2]
            +1.335702658333333e-10 * tc[3]
            -1.945348180000000e-14 * tc[4];
        /*species 4: O2+ */
        species[4] =
            +1.397422290000000e+05 * invT
            +4.811498610999999e+00
            -4.610171670000000e+00 * tc[0]
            +3.179759760000000e-03 * tc[1]
            -2.373760400000000e-06 * tc[2]
            +1.008316025000000e-09 * tc[3]
            -1.854784390000000e-13 * tc[4];
        /*species 5: N2+ */
        species[5] =
            +1.804811150000000e+05 * invT
            +1.082185330000000e+00
            -3.775407110000000e+00 * tc[0]
            +1.032295785000000e-03 * tc[1]
            -7.929205016666667e-07 * tc[2]
            +2.630535233333333e-10 * tc[3]
            -3.352549865000000e-14 * tc[4];
        /*species 6: O4+ */
        species[6] =
            +1.725300970000000e+05 * invT
            -1.699199678000000e+01
            -1.151078020000000e+00 * tc[0]
            -1.077851640000000e-02 * tc[1]
            +3.504515650000000e-06 * tc[2]
            -6.842840075000000e-10 * tc[3]
            +3.248804900000000e-14 * tc[4];
        /*species 7: N4+ */
        species[7] =
            +2.155531830000000e+05 * invT
            -4.496941250000000e+00
            -3.325965150000000e+00 * tc[0]
            -7.195459050000000e-03 * tc[1]
            +2.613287550000000e-06 * tc[2]
            -7.935732066666667e-10 * tc[3]
            +1.267833240000000e-13 * tc[4];
        /*species 8: O2pN2 */
        species[8] =
            +1.725300970000000e+05 * invT
            -1.699199678000000e+01
            -1.151078020000000e+00 * tc[0]
            -1.077851640000000e-02 * tc[1]
            +3.504515650000000e-06 * tc[2]
            -6.842840075000000e-10 * tc[3]
            +3.248804900000000e-14 * tc[4];
        /*species 9: O2- */
        species[9] =
            -6.870769830000000e+03 * invT
            -6.869815900000003e-01
            -3.664425220000000e+00 * tc[0]
            +4.643705690000000e-04 * tc[1]
            -1.075795136666667e-06 * tc[2]
            +6.455861500000001e-10 * tc[3]
            -1.466663310000000e-13 * tc[4];
    } else {
        /*species 0: E */
        species[0] =
            -7.453750000000000e+02 * invT
            +1.422081220000000e+01
            -2.500000000000000e+00 * tc[0]
            -0.000000000000000e+00 * tc[1]
            -0.000000000000000e+00 * tc[2]
            -0.000000000000000e+00 * tc[3]
            -0.000000000000000e+00 * tc[4];
        /*species 1: O2 */
        species[1] =
            -1.233930180000000e+03 * invT
            +5.084126000000002e-01
            -3.697578190000000e+00 * tc[0]
            -3.067598445000000e-04 * tc[1]
            +2.098069983333333e-08 * tc[2]
            -1.479401233333333e-12 * tc[3]
            +5.682176550000000e-17 * tc[4];
        /*species 2: N2 */
        species[2] =
            -9.227977000000000e+02 * invT
            -3.053888000000000e+00
            -2.926640000000000e+00 * tc[0]
            -7.439885000000000e-04 * tc[1]
            +9.474601666666666e-08 * tc[2]
            -8.414199999999999e-12 * tc[3]
            +3.376675500000000e-16 * tc[4];
        /*species 3: O */
        species[3] =
            +2.923080270000000e+04 * invT
            -2.378248450000000e+00
            -2.542059660000000e+00 * tc[0]
            +1.377530955000000e-05 * tc[1]
            +5.171338916666667e-10 * tc[2]
            -3.792556183333333e-13 * tc[3]
            +2.184025750000000e-17 * tc[4];
        /*species 4: O2+ */
        species[4] =
            +1.398768230000000e+05 * invT
            -2.130505470000000e+00
            -3.316759220000000e+00 * tc[0]
            -5.576112200000001e-04 * tc[1]
            +6.391542600000000e-08 * tc[2]
            -4.773205725000000e-12 * tc[3]
            +1.388241905000000e-16 * tc[4];
        /*species 5: N2+ */
        species[5] =
            +1.803909940000000e+05 * invT
            +4.907721999999999e-01
            -3.586613630000000e+00 * tc[0]
            -1.265359745000000e-04 * tc[1]
            -3.079636900000000e-08 * tc[2]
            +3.793810191666666e-12 * tc[3]
            -1.634090145000000e-16 * tc[4];
        /*species 6: O4+ */
        species[6] =
            +1.708397930000000e+05 * invT
            +2.225873356000000e+01
            -7.577843460000000e+00 * tc[0]
            -1.240260260000000e-03 * tc[1]
            +1.618555716666667e-07 * tc[2]
            -1.381152091666667e-11 * tc[3]
            +5.152387000000000e-16 * tc[4];
        /*species 7: N4+ */
        species[7] =
            +2.145405450000000e+05 * invT
            +1.829579346000000e+01
            -7.052858160000000e+00 * tc[0]
            -1.455472345000000e-03 * tc[1]
            +1.859566650000000e-07 * tc[2]
            -1.565699533333333e-11 * tc[3]
            +5.788450650000000e-16 * tc[4];
        /*species 8: O2pN2 */
        species[8] =
            +1.708397930000000e+05 * invT
            +2.225873356000000e+01
            -7.577843460000000e+00 * tc[0]
            -1.240260260000000e-03 * tc[1]
            +1.618555716666667e-07 * tc[2]
            -1.381152091666667e-11 * tc[3]
            +5.152387000000000e-16 * tc[4];
        /*species 9: O2- */
        species[9] =
            -7.062872290000000e+03 * invT
            +1.677952770000000e+00
            -3.956662940000000e+00 * tc[0]
            -2.990709115000000e-04 * tc[1]
            +3.535565083333334e-08 * tc[2]
            -3.027229841666666e-12 * tc[3]
            +1.124946140000000e-16 * tc[4];
    }
    return;
}


/*compute the a/(RT) at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void helmholtz(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    double invT = 1 / T;

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: E */
        species[0] =
            -7.45375000e+02 * invT
            +1.32208122e+01
            -2.50000000e+00 * tc[0]
            -0.00000000e+00 * tc[1]
            -0.00000000e+00 * tc[2]
            -0.00000000e+00 * tc[3]
            -0.00000000e+00 * tc[4];
        /*species 1: O2 */
        species[1] =
            -1.00524902e+03 * invT
            -3.82180119e+00
            -3.21293640e+00 * tc[0]
            -5.63743175e-04 * tc[1]
            +9.59358412e-08 * tc[2]
            -1.09489769e-10 * tc[3]
            +4.38427696e-14 * tc[4];
        /*species 2: N2 */
        species[2] =
            -1.02090000e+03 * invT
            -1.65169500e+00
            -3.29867700e+00 * tc[0]
            -7.04120000e-04 * tc[1]
            +6.60537000e-07 * tc[2]
            -4.70126250e-10 * tc[3]
            +1.22242750e-13 * tc[4];
        /*species 3: O */
        species[3] =
            +2.91476445e+04 * invT
            -1.01756620e+00
            -2.94642878e+00 * tc[0]
            +8.19083245e-04 * tc[1]
            -4.03505283e-07 * tc[2]
            +1.33570266e-10 * tc[3]
            -1.94534818e-14 * tc[4];
        /*species 4: O2+ */
        species[4] =
            +1.39742229e+05 * invT
            +3.81149861e+00
            -4.61017167e+00 * tc[0]
            +3.17975976e-03 * tc[1]
            -2.37376040e-06 * tc[2]
            +1.00831603e-09 * tc[3]
            -1.85478439e-13 * tc[4];
        /*species 5: N2+ */
        species[5] =
            +1.80481115e+05 * invT
            +8.21853300e-02
            -3.77540711e+00 * tc[0]
            +1.03229579e-03 * tc[1]
            -7.92920502e-07 * tc[2]
            +2.63053523e-10 * tc[3]
            -3.35254986e-14 * tc[4];
        /*species 6: O4+ */
        species[6] =
            +1.72530097e+05 * invT
            -1.79919968e+01
            -1.15107802e+00 * tc[0]
            -1.07785164e-02 * tc[1]
            +3.50451565e-06 * tc[2]
            -6.84284007e-10 * tc[3]
            +3.24880490e-14 * tc[4];
        /*species 7: N4+ */
        species[7] =
            +2.15553183e+05 * invT
            -5.49694125e+00
            -3.32596515e+00 * tc[0]
            -7.19545905e-03 * tc[1]
            +2.61328755e-06 * tc[2]
            -7.93573207e-10 * tc[3]
            +1.26783324e-13 * tc[4];
        /*species 8: O2pN2 */
        species[8] =
            +1.72530097e+05 * invT
            -1.79919968e+01
            -1.15107802e+00 * tc[0]
            -1.07785164e-02 * tc[1]
            +3.50451565e-06 * tc[2]
            -6.84284007e-10 * tc[3]
            +3.24880490e-14 * tc[4];
        /*species 9: O2- */
        species[9] =
            -6.87076983e+03 * invT
            -1.68698159e+00
            -3.66442522e+00 * tc[0]
            +4.64370569e-04 * tc[1]
            -1.07579514e-06 * tc[2]
            +6.45586150e-10 * tc[3]
            -1.46666331e-13 * tc[4];
    } else {
        /*species 0: E */
        species[0] =
            -7.45375000e+02 * invT
            +1.32208122e+01
            -2.50000000e+00 * tc[0]
            -0.00000000e+00 * tc[1]
            -0.00000000e+00 * tc[2]
            -0.00000000e+00 * tc[3]
            -0.00000000e+00 * tc[4];
        /*species 1: O2 */
        species[1] =
            -1.23393018e+03 * invT
            -4.91587400e-01
            -3.69757819e+00 * tc[0]
            -3.06759845e-04 * tc[1]
            +2.09806998e-08 * tc[2]
            -1.47940123e-12 * tc[3]
            +5.68217655e-17 * tc[4];
        /*species 2: N2 */
        species[2] =
            -9.22797700e+02 * invT
            -4.05388800e+00
            -2.92664000e+00 * tc[0]
            -7.43988500e-04 * tc[1]
            +9.47460167e-08 * tc[2]
            -8.41420000e-12 * tc[3]
            +3.37667550e-16 * tc[4];
        /*species 3: O */
        species[3] =
            +2.92308027e+04 * invT
            -3.37824845e+00
            -2.54205966e+00 * tc[0]
            +1.37753096e-05 * tc[1]
            +5.17133892e-10 * tc[2]
            -3.79255618e-13 * tc[3]
            +2.18402575e-17 * tc[4];
        /*species 4: O2+ */
        species[4] =
            +1.39876823e+05 * invT
            -3.13050547e+00
            -3.31675922e+00 * tc[0]
            -5.57611220e-04 * tc[1]
            +6.39154260e-08 * tc[2]
            -4.77320572e-12 * tc[3]
            +1.38824191e-16 * tc[4];
        /*species 5: N2+ */
        species[5] =
            +1.80390994e+05 * invT
            -5.09227800e-01
            -3.58661363e+00 * tc[0]
            -1.26535975e-04 * tc[1]
            -3.07963690e-08 * tc[2]
            +3.79381019e-12 * tc[3]
            -1.63409014e-16 * tc[4];
        /*species 6: O4+ */
        species[6] =
            +1.70839793e+05 * invT
            +2.12587336e+01
            -7.57784346e+00 * tc[0]
            -1.24026026e-03 * tc[1]
            +1.61855572e-07 * tc[2]
            -1.38115209e-11 * tc[3]
            +5.15238700e-16 * tc[4];
        /*species 7: N4+ */
        species[7] =
            +2.14540545e+05 * invT
            +1.72957935e+01
            -7.05285816e+00 * tc[0]
            -1.45547234e-03 * tc[1]
            +1.85956665e-07 * tc[2]
            -1.56569953e-11 * tc[3]
            +5.78845065e-16 * tc[4];
        /*species 8: O2pN2 */
        species[8] =
            +1.70839793e+05 * invT
            +2.12587336e+01
            -7.57784346e+00 * tc[0]
            -1.24026026e-03 * tc[1]
            +1.61855572e-07 * tc[2]
            -1.38115209e-11 * tc[3]
            +5.15238700e-16 * tc[4];
        /*species 9: O2- */
        species[9] =
            -7.06287229e+03 * invT
            +6.77952770e-01
            -3.95666294e+00 * tc[0]
            -2.99070912e-04 * tc[1]
            +3.53556508e-08 * tc[2]
            -3.02722984e-12 * tc[3]
            +1.12494614e-16 * tc[4];
    }
    return;
}


/*compute Cv/R at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void cv_R(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: E */
        species[0] =
            +1.50000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4];
        /*species 1: O2 */
        species[1] =
            +2.21293640e+00
            +1.12748635e-03 * tc[1]
            -5.75615047e-07 * tc[2]
            +1.31387723e-09 * tc[3]
            -8.76855392e-13 * tc[4];
        /*species 2: N2 */
        species[2] =
            +2.29867700e+00
            +1.40824000e-03 * tc[1]
            -3.96322200e-06 * tc[2]
            +5.64151500e-09 * tc[3]
            -2.44485500e-12 * tc[4];
        /*species 3: O */
        species[3] =
            +1.94642878e+00
            -1.63816649e-03 * tc[1]
            +2.42103170e-06 * tc[2]
            -1.60284319e-09 * tc[3]
            +3.89069636e-13 * tc[4];
        /*species 4: O2+ */
        species[4] =
            +3.61017167e+00
            -6.35951952e-03 * tc[1]
            +1.42425624e-05 * tc[2]
            -1.20997923e-08 * tc[3]
            +3.70956878e-12 * tc[4];
        /*species 5: N2+ */
        species[5] =
            +2.77540711e+00
            -2.06459157e-03 * tc[1]
            +4.75752301e-06 * tc[2]
            -3.15664228e-09 * tc[3]
            +6.70509973e-13 * tc[4];
        /*species 6: O4+ */
        species[6] =
            +1.51078020e-01
            +2.15570328e-02 * tc[1]
            -2.10270939e-05 * tc[2]
            +8.21140809e-09 * tc[3]
            -6.49760980e-13 * tc[4];
        /*species 7: N4+ */
        species[7] =
            +2.32596515e+00
            +1.43909181e-02 * tc[1]
            -1.56797253e-05 * tc[2]
            +9.52287848e-09 * tc[3]
            -2.53566648e-12 * tc[4];
        /*species 8: O2pN2 */
        species[8] =
            +1.51078020e-01
            +2.15570328e-02 * tc[1]
            -2.10270939e-05 * tc[2]
            +8.21140809e-09 * tc[3]
            -6.49760980e-13 * tc[4];
        /*species 9: O2- */
        species[9] =
            +2.66442522e+00
            -9.28741138e-04 * tc[1]
            +6.45477082e-06 * tc[2]
            -7.74703380e-09 * tc[3]
            +2.93332662e-12 * tc[4];
    } else {
        /*species 0: E */
        species[0] =
            +1.50000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4];
        /*species 1: O2 */
        species[1] =
            +2.69757819e+00
            +6.13519689e-04 * tc[1]
            -1.25884199e-07 * tc[2]
            +1.77528148e-11 * tc[3]
            -1.13643531e-15 * tc[4];
        /*species 2: N2 */
        species[2] =
            +1.92664000e+00
            +1.48797700e-03 * tc[1]
            -5.68476100e-07 * tc[2]
            +1.00970400e-10 * tc[3]
            -6.75335100e-15 * tc[4];
        /*species 3: O */
        species[3] =
            +1.54205966e+00
            -2.75506191e-05 * tc[1]
            -3.10280335e-09 * tc[2]
            +4.55106742e-12 * tc[3]
            -4.36805150e-16 * tc[4];
        /*species 4: O2+ */
        species[4] =
            +2.31675922e+00
            +1.11522244e-03 * tc[1]
            -3.83492556e-07 * tc[2]
            +5.72784687e-11 * tc[3]
            -2.77648381e-15 * tc[4];
        /*species 5: N2+ */
        species[5] =
            +2.58661363e+00
            +2.53071949e-04 * tc[1]
            +1.84778214e-07 * tc[2]
            -4.55257223e-11 * tc[3]
            +3.26818029e-15 * tc[4];
        /*species 6: O4+ */
        species[6] =
            +6.57784346e+00
            +2.48052052e-03 * tc[1]
            -9.71133430e-07 * tc[2]
            +1.65738251e-10 * tc[3]
            -1.03047740e-14 * tc[4];
        /*species 7: N4+ */
        species[7] =
            +6.05285816e+00
            +2.91094469e-03 * tc[1]
            -1.11573999e-06 * tc[2]
            +1.87883944e-10 * tc[3]
            -1.15769013e-14 * tc[4];
        /*species 8: O2pN2 */
        species[8] =
            +6.57784346e+00
            +2.48052052e-03 * tc[1]
            -9.71133430e-07 * tc[2]
            +1.65738251e-10 * tc[3]
            -1.03047740e-14 * tc[4];
        /*species 9: O2- */
        species[9] =
            +2.95666294e+00
            +5.98141823e-04 * tc[1]
            -2.12133905e-07 * tc[2]
            +3.63267581e-11 * tc[3]
            -2.24989228e-15 * tc[4];
    }
    return;
}


/*compute Cp/R at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void cp_R(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: E */
        species[0] =
            +2.50000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4];
        /*species 1: O2 */
        species[1] =
            +3.21293640e+00
            +1.12748635e-03 * tc[1]
            -5.75615047e-07 * tc[2]
            +1.31387723e-09 * tc[3]
            -8.76855392e-13 * tc[4];
        /*species 2: N2 */
        species[2] =
            +3.29867700e+00
            +1.40824000e-03 * tc[1]
            -3.96322200e-06 * tc[2]
            +5.64151500e-09 * tc[3]
            -2.44485500e-12 * tc[4];
        /*species 3: O */
        species[3] =
            +2.94642878e+00
            -1.63816649e-03 * tc[1]
            +2.42103170e-06 * tc[2]
            -1.60284319e-09 * tc[3]
            +3.89069636e-13 * tc[4];
        /*species 4: O2+ */
        species[4] =
            +4.61017167e+00
            -6.35951952e-03 * tc[1]
            +1.42425624e-05 * tc[2]
            -1.20997923e-08 * tc[3]
            +3.70956878e-12 * tc[4];
        /*species 5: N2+ */
        species[5] =
            +3.77540711e+00
            -2.06459157e-03 * tc[1]
            +4.75752301e-06 * tc[2]
            -3.15664228e-09 * tc[3]
            +6.70509973e-13 * tc[4];
        /*species 6: O4+ */
        species[6] =
            +1.15107802e+00
            +2.15570328e-02 * tc[1]
            -2.10270939e-05 * tc[2]
            +8.21140809e-09 * tc[3]
            -6.49760980e-13 * tc[4];
        /*species 7: N4+ */
        species[7] =
            +3.32596515e+00
            +1.43909181e-02 * tc[1]
            -1.56797253e-05 * tc[2]
            +9.52287848e-09 * tc[3]
            -2.53566648e-12 * tc[4];
        /*species 8: O2pN2 */
        species[8] =
            +1.15107802e+00
            +2.15570328e-02 * tc[1]
            -2.10270939e-05 * tc[2]
            +8.21140809e-09 * tc[3]
            -6.49760980e-13 * tc[4];
        /*species 9: O2- */
        species[9] =
            +3.66442522e+00
            -9.28741138e-04 * tc[1]
            +6.45477082e-06 * tc[2]
            -7.74703380e-09 * tc[3]
            +2.93332662e-12 * tc[4];
    } else {
        /*species 0: E */
        species[0] =
            +2.50000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4];
        /*species 1: O2 */
        species[1] =
            +3.69757819e+00
            +6.13519689e-04 * tc[1]
            -1.25884199e-07 * tc[2]
            +1.77528148e-11 * tc[3]
            -1.13643531e-15 * tc[4];
        /*species 2: N2 */
        species[2] =
            +2.92664000e+00
            +1.48797700e-03 * tc[1]
            -5.68476100e-07 * tc[2]
            +1.00970400e-10 * tc[3]
            -6.75335100e-15 * tc[4];
        /*species 3: O */
        species[3] =
            +2.54205966e+00
            -2.75506191e-05 * tc[1]
            -3.10280335e-09 * tc[2]
            +4.55106742e-12 * tc[3]
            -4.36805150e-16 * tc[4];
        /*species 4: O2+ */
        species[4] =
            +3.31675922e+00
            +1.11522244e-03 * tc[1]
            -3.83492556e-07 * tc[2]
            +5.72784687e-11 * tc[3]
            -2.77648381e-15 * tc[4];
        /*species 5: N2+ */
        species[5] =
            +3.58661363e+00
            +2.53071949e-04 * tc[1]
            +1.84778214e-07 * tc[2]
            -4.55257223e-11 * tc[3]
            +3.26818029e-15 * tc[4];
        /*species 6: O4+ */
        species[6] =
            +7.57784346e+00
            +2.48052052e-03 * tc[1]
            -9.71133430e-07 * tc[2]
            +1.65738251e-10 * tc[3]
            -1.03047740e-14 * tc[4];
        /*species 7: N4+ */
        species[7] =
            +7.05285816e+00
            +2.91094469e-03 * tc[1]
            -1.11573999e-06 * tc[2]
            +1.87883944e-10 * tc[3]
            -1.15769013e-14 * tc[4];
        /*species 8: O2pN2 */
        species[8] =
            +7.57784346e+00
            +2.48052052e-03 * tc[1]
            -9.71133430e-07 * tc[2]
            +1.65738251e-10 * tc[3]
            -1.03047740e-14 * tc[4];
        /*species 9: O2- */
        species[9] =
            +3.95666294e+00
            +5.98141823e-04 * tc[1]
            -2.12133905e-07 * tc[2]
            +3.63267581e-11 * tc[3]
            -2.24989228e-15 * tc[4];
    }
    return;
}


/*compute the e/(RT) at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void speciesInternalEnergy(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    double invT = 1 / T;

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: E */
        species[0] =
            +1.50000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4]
            -7.45375000e+02 * invT;
        /*species 1: O2 */
        species[1] =
            +2.21293640e+00
            +5.63743175e-04 * tc[1]
            -1.91871682e-07 * tc[2]
            +3.28469308e-10 * tc[3]
            -1.75371078e-13 * tc[4]
            -1.00524902e+03 * invT;
        /*species 2: N2 */
        species[2] =
            +2.29867700e+00
            +7.04120000e-04 * tc[1]
            -1.32107400e-06 * tc[2]
            +1.41037875e-09 * tc[3]
            -4.88971000e-13 * tc[4]
            -1.02090000e+03 * invT;
        /*species 3: O */
        species[3] =
            +1.94642878e+00
            -8.19083245e-04 * tc[1]
            +8.07010567e-07 * tc[2]
            -4.00710797e-10 * tc[3]
            +7.78139272e-14 * tc[4]
            +2.91476445e+04 * invT;
        /*species 4: O2+ */
        species[4] =
            +3.61017167e+00
            -3.17975976e-03 * tc[1]
            +4.74752080e-06 * tc[2]
            -3.02494808e-09 * tc[3]
            +7.41913756e-13 * tc[4]
            +1.39742229e+05 * invT;
        /*species 5: N2+ */
        species[5] =
            +2.77540711e+00
            -1.03229579e-03 * tc[1]
            +1.58584100e-06 * tc[2]
            -7.89160570e-10 * tc[3]
            +1.34101995e-13 * tc[4]
            +1.80481115e+05 * invT;
        /*species 6: O4+ */
        species[6] =
            +1.51078020e-01
            +1.07785164e-02 * tc[1]
            -7.00903130e-06 * tc[2]
            +2.05285202e-09 * tc[3]
            -1.29952196e-13 * tc[4]
            +1.72530097e+05 * invT;
        /*species 7: N4+ */
        species[7] =
            +2.32596515e+00
            +7.19545905e-03 * tc[1]
            -5.22657510e-06 * tc[2]
            +2.38071962e-09 * tc[3]
            -5.07133296e-13 * tc[4]
            +2.15553183e+05 * invT;
        /*species 8: O2pN2 */
        species[8] =
            +1.51078020e-01
            +1.07785164e-02 * tc[1]
            -7.00903130e-06 * tc[2]
            +2.05285202e-09 * tc[3]
            -1.29952196e-13 * tc[4]
            +1.72530097e+05 * invT;
        /*species 9: O2- */
        species[9] =
            +2.66442522e+00
            -4.64370569e-04 * tc[1]
            +2.15159027e-06 * tc[2]
            -1.93675845e-09 * tc[3]
            +5.86665324e-13 * tc[4]
            -6.87076983e+03 * invT;
    } else {
        /*species 0: E */
        species[0] =
            +1.50000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4]
            -7.45375000e+02 * invT;
        /*species 1: O2 */
        species[1] =
            +2.69757819e+00
            +3.06759845e-04 * tc[1]
            -4.19613997e-08 * tc[2]
            +4.43820370e-12 * tc[3]
            -2.27287062e-16 * tc[4]
            -1.23393018e+03 * invT;
        /*species 2: N2 */
        species[2] =
            +1.92664000e+00
            +7.43988500e-04 * tc[1]
            -1.89492033e-07 * tc[2]
            +2.52426000e-11 * tc[3]
            -1.35067020e-15 * tc[4]
            -9.22797700e+02 * invT;
        /*species 3: O */
        species[3] =
            +1.54205966e+00
            -1.37753096e-05 * tc[1]
            -1.03426778e-09 * tc[2]
            +1.13776685e-12 * tc[3]
            -8.73610300e-17 * tc[4]
            +2.92308027e+04 * invT;
        /*species 4: O2+ */
        species[4] =
            +2.31675922e+00
            +5.57611220e-04 * tc[1]
            -1.27830852e-07 * tc[2]
            +1.43196172e-11 * tc[3]
            -5.55296762e-16 * tc[4]
            +1.39876823e+05 * invT;
        /*species 5: N2+ */
        species[5] =
            +2.58661363e+00
            +1.26535975e-04 * tc[1]
            +6.15927380e-08 * tc[2]
            -1.13814306e-11 * tc[3]
            +6.53636058e-16 * tc[4]
            +1.80390994e+05 * invT;
        /*species 6: O4+ */
        species[6] =
            +6.57784346e+00
            +1.24026026e-03 * tc[1]
            -3.23711143e-07 * tc[2]
            +4.14345627e-11 * tc[3]
            -2.06095480e-15 * tc[4]
            +1.70839793e+05 * invT;
        /*species 7: N4+ */
        species[7] =
            +6.05285816e+00
            +1.45547234e-03 * tc[1]
            -3.71913330e-07 * tc[2]
            +4.69709860e-11 * tc[3]
            -2.31538026e-15 * tc[4]
            +2.14540545e+05 * invT;
        /*species 8: O2pN2 */
        species[8] =
            +6.57784346e+00
            +1.24026026e-03 * tc[1]
            -3.23711143e-07 * tc[2]
            +4.14345627e-11 * tc[3]
            -2.06095480e-15 * tc[4]
            +1.70839793e+05 * invT;
        /*species 9: O2- */
        species[9] =
            +2.95666294e+00
            +2.99070912e-04 * tc[1]
            -7.07113017e-08 * tc[2]
            +9.08168952e-12 * tc[3]
            -4.49978456e-16 * tc[4]
            -7.06287229e+03 * invT;
    }
    return;
}


/*compute the h/(RT) at the given temperature (Eq 20) */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void speciesEnthalpy(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];
    double invT = 1 / T;

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: E */
        species[0] =
            +2.50000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4]
            -7.45375000e+02 * invT;
        /*species 1: O2 */
        species[1] =
            +3.21293640e+00
            +5.63743175e-04 * tc[1]
            -1.91871682e-07 * tc[2]
            +3.28469308e-10 * tc[3]
            -1.75371078e-13 * tc[4]
            -1.00524902e+03 * invT;
        /*species 2: N2 */
        species[2] =
            +3.29867700e+00
            +7.04120000e-04 * tc[1]
            -1.32107400e-06 * tc[2]
            +1.41037875e-09 * tc[3]
            -4.88971000e-13 * tc[4]
            -1.02090000e+03 * invT;
        /*species 3: O */
        species[3] =
            +2.94642878e+00
            -8.19083245e-04 * tc[1]
            +8.07010567e-07 * tc[2]
            -4.00710797e-10 * tc[3]
            +7.78139272e-14 * tc[4]
            +2.91476445e+04 * invT;
        /*species 4: O2+ */
        species[4] =
            +4.61017167e+00
            -3.17975976e-03 * tc[1]
            +4.74752080e-06 * tc[2]
            -3.02494808e-09 * tc[3]
            +7.41913756e-13 * tc[4]
            +1.39742229e+05 * invT;
        /*species 5: N2+ */
        species[5] =
            +3.77540711e+00
            -1.03229579e-03 * tc[1]
            +1.58584100e-06 * tc[2]
            -7.89160570e-10 * tc[3]
            +1.34101995e-13 * tc[4]
            +1.80481115e+05 * invT;
        /*species 6: O4+ */
        species[6] =
            +1.15107802e+00
            +1.07785164e-02 * tc[1]
            -7.00903130e-06 * tc[2]
            +2.05285202e-09 * tc[3]
            -1.29952196e-13 * tc[4]
            +1.72530097e+05 * invT;
        /*species 7: N4+ */
        species[7] =
            +3.32596515e+00
            +7.19545905e-03 * tc[1]
            -5.22657510e-06 * tc[2]
            +2.38071962e-09 * tc[3]
            -5.07133296e-13 * tc[4]
            +2.15553183e+05 * invT;
        /*species 8: O2pN2 */
        species[8] =
            +1.15107802e+00
            +1.07785164e-02 * tc[1]
            -7.00903130e-06 * tc[2]
            +2.05285202e-09 * tc[3]
            -1.29952196e-13 * tc[4]
            +1.72530097e+05 * invT;
        /*species 9: O2- */
        species[9] =
            +3.66442522e+00
            -4.64370569e-04 * tc[1]
            +2.15159027e-06 * tc[2]
            -1.93675845e-09 * tc[3]
            +5.86665324e-13 * tc[4]
            -6.87076983e+03 * invT;
    } else {
        /*species 0: E */
        species[0] =
            +2.50000000e+00
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4]
            -7.45375000e+02 * invT;
        /*species 1: O2 */
        species[1] =
            +3.69757819e+00
            +3.06759845e-04 * tc[1]
            -4.19613997e-08 * tc[2]
            +4.43820370e-12 * tc[3]
            -2.27287062e-16 * tc[4]
            -1.23393018e+03 * invT;
        /*species 2: N2 */
        species[2] =
            +2.92664000e+00
            +7.43988500e-04 * tc[1]
            -1.89492033e-07 * tc[2]
            +2.52426000e-11 * tc[3]
            -1.35067020e-15 * tc[4]
            -9.22797700e+02 * invT;
        /*species 3: O */
        species[3] =
            +2.54205966e+00
            -1.37753096e-05 * tc[1]
            -1.03426778e-09 * tc[2]
            +1.13776685e-12 * tc[3]
            -8.73610300e-17 * tc[4]
            +2.92308027e+04 * invT;
        /*species 4: O2+ */
        species[4] =
            +3.31675922e+00
            +5.57611220e-04 * tc[1]
            -1.27830852e-07 * tc[2]
            +1.43196172e-11 * tc[3]
            -5.55296762e-16 * tc[4]
            +1.39876823e+05 * invT;
        /*species 5: N2+ */
        species[5] =
            +3.58661363e+00
            +1.26535975e-04 * tc[1]
            +6.15927380e-08 * tc[2]
            -1.13814306e-11 * tc[3]
            +6.53636058e-16 * tc[4]
            +1.80390994e+05 * invT;
        /*species 6: O4+ */
        species[6] =
            +7.57784346e+00
            +1.24026026e-03 * tc[1]
            -3.23711143e-07 * tc[2]
            +4.14345627e-11 * tc[3]
            -2.06095480e-15 * tc[4]
            +1.70839793e+05 * invT;
        /*species 7: N4+ */
        species[7] =
            +7.05285816e+00
            +1.45547234e-03 * tc[1]
            -3.71913330e-07 * tc[2]
            +4.69709860e-11 * tc[3]
            -2.31538026e-15 * tc[4]
            +2.14540545e+05 * invT;
        /*species 8: O2pN2 */
        species[8] =
            +7.57784346e+00
            +1.24026026e-03 * tc[1]
            -3.23711143e-07 * tc[2]
            +4.14345627e-11 * tc[3]
            -2.06095480e-15 * tc[4]
            +1.70839793e+05 * invT;
        /*species 9: O2- */
        species[9] =
            +3.95666294e+00
            +2.99070912e-04 * tc[1]
            -7.07113017e-08 * tc[2]
            +9.08168952e-12 * tc[3]
            -4.49978456e-16 * tc[4]
            -7.06287229e+03 * invT;
    }
    return;
}


/*compute the S/R at the given temperature (Eq 21) */
/*tc contains precomputed powers of T, tc[0] = log(T) */
AMREX_GPU_HOST_DEVICE void speciesEntropy(double * species, double *  tc)
{

    /*temperature */
    double T = tc[1];

    /*species with midpoint at T=1000 kelvin */
    if (T < 1000) {
        /*species 0: E */
        species[0] =
            +2.50000000e+00 * tc[0]
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4]
            -1.17208122e+01 ;
        /*species 1: O2 */
        species[1] =
            +3.21293640e+00 * tc[0]
            +1.12748635e-03 * tc[1]
            -2.87807523e-07 * tc[2]
            +4.37959077e-10 * tc[3]
            -2.19213848e-13 * tc[4]
            +6.03473759e+00 ;
        /*species 2: N2 */
        species[2] =
            +3.29867700e+00 * tc[0]
            +1.40824000e-03 * tc[1]
            -1.98161100e-06 * tc[2]
            +1.88050500e-09 * tc[3]
            -6.11213750e-13 * tc[4]
            +3.95037200e+00 ;
        /*species 3: O */
        species[3] =
            +2.94642878e+00 * tc[0]
            -1.63816649e-03 * tc[1]
            +1.21051585e-06 * tc[2]
            -5.34281063e-10 * tc[3]
            +9.72674090e-14 * tc[4]
            +2.96399498e+00 ;
        /*species 4: O2+ */
        species[4] =
            +4.61017167e+00 * tc[0]
            -6.35951952e-03 * tc[1]
            +7.12128120e-06 * tc[2]
            -4.03326410e-09 * tc[3]
            +9.27392195e-13 * tc[4]
            -2.01326941e-01 ;
        /*species 5: N2+ */
        species[5] =
            +3.77540711e+00 * tc[0]
            -2.06459157e-03 * tc[1]
            +2.37876151e-06 * tc[2]
            -1.05221409e-09 * tc[3]
            +1.67627493e-13 * tc[4]
            +2.69322178e+00 ;
        /*species 6: O4+ */
        species[6] =
            +1.15107802e+00 * tc[0]
            +2.15570328e-02 * tc[1]
            -1.05135469e-05 * tc[2]
            +2.73713603e-09 * tc[3]
            -1.62440245e-13 * tc[4]
            +1.81430748e+01 ;
        /*species 7: N4+ */
        species[7] =
            +3.32596515e+00 * tc[0]
            +1.43909181e-02 * tc[1]
            -7.83986265e-06 * tc[2]
            +3.17429283e-09 * tc[3]
            -6.33916620e-13 * tc[4]
            +7.82290640e+00 ;
        /*species 8: O2pN2 */
        species[8] =
            +1.15107802e+00 * tc[0]
            +2.15570328e-02 * tc[1]
            -1.05135469e-05 * tc[2]
            +2.73713603e-09 * tc[3]
            -1.62440245e-13 * tc[4]
            +1.81430748e+01 ;
        /*species 9: O2- */
        species[9] =
            +3.66442522e+00 * tc[0]
            -9.28741138e-04 * tc[1]
            +3.22738541e-06 * tc[2]
            -2.58234460e-09 * tc[3]
            +7.33331655e-13 * tc[4]
            +4.35140681e+00 ;
    } else {
        /*species 0: E */
        species[0] =
            +2.50000000e+00 * tc[0]
            +0.00000000e+00 * tc[1]
            +0.00000000e+00 * tc[2]
            +0.00000000e+00 * tc[3]
            +0.00000000e+00 * tc[4]
            -1.17208122e+01 ;
        /*species 1: O2 */
        species[1] =
            +3.69757819e+00 * tc[0]
            +6.13519689e-04 * tc[1]
            -6.29420995e-08 * tc[2]
            +5.91760493e-12 * tc[3]
            -2.84108828e-16 * tc[4]
            +3.18916559e+00 ;
        /*species 2: N2 */
        species[2] =
            +2.92664000e+00 * tc[0]
            +1.48797700e-03 * tc[1]
            -2.84238050e-07 * tc[2]
            +3.36568000e-11 * tc[3]
            -1.68833775e-15 * tc[4]
            +5.98052800e+00 ;
        /*species 3: O */
        species[3] =
            +2.54205966e+00 * tc[0]
            -2.75506191e-05 * tc[1]
            -1.55140167e-09 * tc[2]
            +1.51702247e-12 * tc[3]
            -1.09201287e-16 * tc[4]
            +4.92030811e+00 ;
        /*species 4: O2+ */
        species[4] =
            +3.31675922e+00 * tc[0]
            +1.11522244e-03 * tc[1]
            -1.91746278e-07 * tc[2]
            +1.90928229e-11 * tc[3]
            -6.94120952e-16 * tc[4]
            +5.44726469e+00 ;
        /*species 5: N2+ */
        species[5] =
            +3.58661363e+00 * tc[0]
            +2.53071949e-04 * tc[1]
            +9.23891070e-08 * tc[2]
            -1.51752408e-11 * tc[3]
            +8.17045073e-16 * tc[4]
            +3.09584143e+00 ;
        /*species 6: O4+ */
        species[6] =
            +7.57784346e+00 * tc[0]
            +2.48052052e-03 * tc[1]
            -4.85566715e-07 * tc[2]
            +5.52460837e-11 * tc[3]
            -2.57619350e-15 * tc[4]
            -1.46808901e+01 ;
        /*species 7: N4+ */
        species[7] =
            +7.05285816e+00 * tc[0]
            +2.91094469e-03 * tc[1]
            -5.57869995e-07 * tc[2]
            +6.26279813e-11 * tc[3]
            -2.89422533e-15 * tc[4]
            -1.12429353e+01 ;
        /*species 8: O2pN2 */
        species[8] =
            +7.57784346e+00 * tc[0]
            +2.48052052e-03 * tc[1]
            -4.85566715e-07 * tc[2]
            +5.52460837e-11 * tc[3]
            -2.57619350e-15 * tc[4]
            -1.46808901e+01 ;
        /*species 9: O2- */
        species[9] =
            +3.95666294e+00 * tc[0]
            +5.98141823e-04 * tc[1]
            -1.06066953e-07 * tc[2]
            +1.21089194e-11 * tc[3]
            -5.62473070e-16 * tc[4]
            +2.27871017e+00 ;
    }
    return;
}


/*save atomic weights into array */
void atomicWeight(double *  awt)
{
    awt[0] = 15.999400; /*O */
    awt[1] = 14.006700; /*N */
    awt[2] = 0.000549; /*E */

    return;
}


/* get temperature given internal energy in mass units and mass fracs */
AMREX_GPU_HOST_DEVICE void GET_T_GIVEN_EY(double *  e, double *  y, double *  t, int * ierr)
{
#ifdef CONVERGENCE
    const int maxiter = 5000;
    const double tol  = 1.e-12;
#else
    const int maxiter = 200;
    const double tol  = 1.e-6;
#endif
    double ein  = *e;
    double tmin = 90;/*max lower bound for thermo def */
    double tmax = 4000;/*min upper bound for thermo def */
    double e1,emin,emax,cv,t1,dt;
    int i;/* loop counter */
    CKUBMS(&tmin, y, &emin);
    CKUBMS(&tmax, y, &emax);
    if (ein < emin) {
        /*Linear Extrapolation below tmin */
        CKCVBS(&tmin, y, &cv);
        *t = tmin - (emin-ein)/cv;
        *ierr = 1;
        return;
    }
    if (ein > emax) {
        /*Linear Extrapolation above tmax */
        CKCVBS(&tmax, y, &cv);
        *t = tmax - (emax-ein)/cv;
        *ierr = 1;
        return;
    }
    t1 = *t;
    if (t1 < tmin || t1 > tmax) {
        t1 = tmin + (tmax-tmin)/(emax-emin)*(ein-emin);
    }
    for (i = 0; i < maxiter; ++i) {
        CKUBMS(&t1,y,&e1);
        CKCVBS(&t1,y,&cv);
        dt = (ein - e1) / cv;
        if (dt > 100.) { dt = 100.; }
        else if (dt < -100.) { dt = -100.; }
        else if (fabs(dt) < tol) break;
        else if (t1+dt == t1) break;
        t1 += dt;
    }
    *t = t1;
    *ierr = 0;
    return;
}

/* get temperature given enthalpy in mass units and mass fracs */
AMREX_GPU_HOST_DEVICE void GET_T_GIVEN_HY(double *  h, double *  y, double *  t, int * ierr)
{
#ifdef CONVERGENCE
    const int maxiter = 5000;
    const double tol  = 1.e-12;
#else
    const int maxiter = 200;
    const double tol  = 1.e-6;
#endif
    double hin  = *h;
    double tmin = 90;/*max lower bound for thermo def */
    double tmax = 4000;/*min upper bound for thermo def */
    double h1,hmin,hmax,cp,t1,dt;
    int i;/* loop counter */
    CKHBMS(&tmin, y, &hmin);
    CKHBMS(&tmax, y, &hmax);
    if (hin < hmin) {
        /*Linear Extrapolation below tmin */
        CKCPBS(&tmin, y, &cp);
        *t = tmin - (hmin-hin)/cp;
        *ierr = 1;
        return;
    }
    if (hin > hmax) {
        /*Linear Extrapolation above tmax */
        CKCPBS(&tmax, y, &cp);
        *t = tmax - (hmax-hin)/cp;
        *ierr = 1;
        return;
    }
    t1 = *t;
    if (t1 < tmin || t1 > tmax) {
        t1 = tmin + (tmax-tmin)/(hmax-hmin)*(hin-hmin);
    }
    for (i = 0; i < maxiter; ++i) {
        CKHBMS(&t1,y,&h1);
        CKCPBS(&t1,y,&cp);
        dt = (hin - h1) / cp;
        if (dt > 100.) { dt = 100.; }
        else if (dt < -100.) { dt = -100.; }
        else if (fabs(dt) < tol) break;
        else if (t1+dt == t1) break;
        t1 += dt;
    }
    *t = t1;
    *ierr = 0;
    return;
}


/*compute the critical parameters for each species */
void GET_CRITPARAMS(double *  Tci, double *  ai, double *  bi, double *  acentric_i)
{

    double   EPS[10];
    double   SIG[10];
    double    wt[10];
    double avogadro = 6.02214199e23;
    double boltzmann = 1.3806503e-16; //we work in CGS
    double Rcst = 83.144598; //in bar [CGS] !

    egtransetEPS(EPS);
    egtransetSIG(SIG);
    get_mw(wt);

    /*species 0: E */
    Tci[0] = 1.316 * EPS[0] ; 
    ai[0] = (5.55 * pow(avogadro,2.0) * EPS[0]*boltzmann * pow(1e-8*SIG[0],3.0) ) / (pow(wt[0],2.0)); 
    bi[0] = 0.855 * avogadro * pow(1e-8*SIG[0],3.0) / (wt[0]); 
    acentric_i[0] = 0.0 ;

    /*species 1: O2 */
    /*Imported from NIST */
    Tci[1] = 154.581000 ; 
    ai[1] = 1e6 * 0.42748 * pow(Rcst,2.0) * pow(Tci[1],2.0) / (pow(31.998800,2.0) * 50.430466); 
    bi[1] = 0.08664 * Rcst * Tci[1] / (31.998800 * 50.430466); 
    acentric_i[1] = 0.022200 ;

    /*species 2: N2 */
    /*Imported from NIST */
    Tci[2] = 126.192000 ; 
    ai[2] = 1e6 * 0.42748 * pow(Rcst,2.0) * pow(Tci[2],2.0) / (pow(28.013400,2.0) * 33.958000); 
    bi[2] = 0.08664 * Rcst * Tci[2] / (28.013400 * 33.958000); 
    acentric_i[2] = 0.037200 ;

    /*species 3: O */
    Tci[3] = 1.316 * EPS[3] ; 
    ai[3] = (5.55 * pow(avogadro,2.0) * EPS[3]*boltzmann * pow(1e-8*SIG[3],3.0) ) / (pow(wt[3],2.0)); 
    bi[3] = 0.855 * avogadro * pow(1e-8*SIG[3],3.0) / (wt[3]); 
    acentric_i[3] = 0.0 ;

    /*species 4: O2+ */
    Tci[4] = 1.316 * EPS[4] ; 
    ai[4] = (5.55 * pow(avogadro,2.0) * EPS[4]*boltzmann * pow(1e-8*SIG[4],3.0) ) / (pow(wt[4],2.0)); 
    bi[4] = 0.855 * avogadro * pow(1e-8*SIG[4],3.0) / (wt[4]); 
    acentric_i[4] = 0.0 ;

    /*species 5: N2+ */
    Tci[5] = 1.316 * EPS[5] ; 
    ai[5] = (5.55 * pow(avogadro,2.0) * EPS[5]*boltzmann * pow(1e-8*SIG[5],3.0) ) / (pow(wt[5],2.0)); 
    bi[5] = 0.855 * avogadro * pow(1e-8*SIG[5],3.0) / (wt[5]); 
    acentric_i[5] = 0.0 ;

    /*species 6: O4+ */
    Tci[6] = 1.316 * EPS[6] ; 
    ai[6] = (5.55 * pow(avogadro,2.0) * EPS[6]*boltzmann * pow(1e-8*SIG[6],3.0) ) / (pow(wt[6],2.0)); 
    bi[6] = 0.855 * avogadro * pow(1e-8*SIG[6],3.0) / (wt[6]); 
    acentric_i[6] = 0.0 ;

    /*species 7: N4+ */
    Tci[7] = 1.316 * EPS[7] ; 
    ai[7] = (5.55 * pow(avogadro,2.0) * EPS[7]*boltzmann * pow(1e-8*SIG[7],3.0) ) / (pow(wt[7],2.0)); 
    bi[7] = 0.855 * avogadro * pow(1e-8*SIG[7],3.0) / (wt[7]); 
    acentric_i[7] = 0.0 ;

    /*species 8: O2pN2 */
    Tci[8] = 1.316 * EPS[8] ; 
    ai[8] = (5.55 * pow(avogadro,2.0) * EPS[8]*boltzmann * pow(1e-8*SIG[8],3.0) ) / (pow(wt[8],2.0)); 
    bi[8] = 0.855 * avogadro * pow(1e-8*SIG[8],3.0) / (wt[8]); 
    acentric_i[8] = 0.0 ;

    /*species 9: O2- */
    Tci[9] = 1.316 * EPS[9] ; 
    ai[9] = (5.55 * pow(avogadro,2.0) * EPS[9]*boltzmann * pow(1e-8*SIG[9],3.0) ) / (pow(wt[9],2.0)); 
    bi[9] = 0.855 * avogadro * pow(1e-8*SIG[9],3.0) / (wt[9]); 
    acentric_i[9] = 0.0 ;

    return;
}


void egtransetLENIMC(int* LENIMC ) {
    *LENIMC = 41;}


void egtransetLENRMC(int* LENRMC ) {
    *LENRMC = 2210;}


void egtransetNO(int* NO ) {
    *NO = 4;}


void egtransetKK(int* KK ) {
    *KK = 10;}


void egtransetNLITE(int* NLITE ) {
    *NLITE = 1;}


/*Patm in ergs/cm3 */
void egtransetPATM(double* PATM) {
    *PATM =   0.1013250000000000E+07;}


/*the molecular weights in g/mol */
void egtransetWT(double* WT ) {
    WT[0] = 5.48580100E-04;
    WT[1] = 3.19988000E+01;
    WT[2] = 2.80134000E+01;
    WT[3] = 1.59994000E+01;
    WT[4] = 3.19982514E+01;
    WT[5] = 2.80128514E+01;
    WT[6] = 6.39970514E+01;
    WT[7] = 5.60262514E+01;
    WT[8] = 6.39970514E+01;
    WT[9] = 3.19993486E+01;
}


/*the lennard-jones potential well depth eps/kb in K */
void egtransetEPS(double* EPS ) {
    EPS[0] = 8.50000000E+02;
    EPS[1] = 1.07400000E+02;
    EPS[2] = 9.75300000E+01;
    EPS[3] = 1.07400000E+02;
    EPS[4] = 1.07400000E+02;
    EPS[5] = 9.75300000E+01;
    EPS[6] = 1.07400000E+02;
    EPS[7] = 9.75300000E+01;
    EPS[8] = 1.07400000E+02;
    EPS[9] = 1.07400000E+02;
}


/*the lennard-jones collision diameter in Angstroms */
void egtransetSIG(double* SIG ) {
    SIG[0] = 4.25000000E+02;
    SIG[1] = 3.45800000E+00;
    SIG[2] = 3.62100000E+00;
    SIG[3] = 3.45800000E+00;
    SIG[4] = 3.45800000E+00;
    SIG[5] = 3.62100000E+00;
    SIG[6] = 3.45800000E+00;
    SIG[7] = 3.62100000E+00;
    SIG[8] = 3.45800000E+00;
    SIG[9] = 3.45800000E+00;
}


/*the dipole moment in Debye */
void egtransetDIP(double* DIP ) {
    DIP[0] = 0.00000000E+00;
    DIP[1] = 0.00000000E+00;
    DIP[2] = 0.00000000E+00;
    DIP[3] = 0.00000000E+00;
    DIP[4] = 0.00000000E+00;
    DIP[5] = 0.00000000E+00;
    DIP[6] = 0.00000000E+00;
    DIP[7] = 0.00000000E+00;
    DIP[8] = 0.00000000E+00;
    DIP[9] = 0.00000000E+00;
}


/*the polarizability in cubic Angstroms */
void egtransetPOL(double* POL ) {
    POL[0] = 0.00000000E+00;
    POL[1] = 1.60000000E+00;
    POL[2] = 1.76000000E+00;
    POL[3] = 1.60000000E+00;
    POL[4] = 1.60000000E+00;
    POL[5] = 1.76000000E+00;
    POL[6] = 1.60000000E+00;
    POL[7] = 1.76000000E+00;
    POL[8] = 1.60000000E+00;
    POL[9] = 1.60000000E+00;
}


/*the rotational relaxation collision number at 298 K */
void egtransetZROT(double* ZROT ) {
    ZROT[0] = 1.00000000E+00;
    ZROT[1] = 3.80000000E+00;
    ZROT[2] = 4.00000000E+00;
    ZROT[3] = 3.80000000E+00;
    ZROT[4] = 3.80000000E+00;
    ZROT[5] = 4.00000000E+00;
    ZROT[6] = 3.80000000E+00;
    ZROT[7] = 4.00000000E+00;
    ZROT[8] = 3.80000000E+00;
    ZROT[9] = 3.80000000E+00;
}


/*0: monoatomic, 1: linear, 2: nonlinear */
void egtransetNLIN(int* NLIN) {
    NLIN[0] = 0;
    NLIN[1] = 1;
    NLIN[2] = 1;
    NLIN[3] = 1;
    NLIN[4] = 1;
    NLIN[5] = 1;
    NLIN[6] = 1;
    NLIN[7] = 1;
    NLIN[8] = 1;
    NLIN[9] = 1;
}


/*Poly fits for the viscosities, dim NO*KK */
void egtransetCOFETA(double* COFETA) {
    COFETA[0] = -2.69531456E+01;
    COFETA[1] = -7.88142336E-01;
    COFETA[2] = 3.18038402E-01;
    COFETA[3] = -1.85151697E-02;
    COFETA[4] = -1.60066324E+01;
    COFETA[5] = 2.16753735E+00;
    COFETA[6] = -1.97226850E-01;
    COFETA[7] = 8.50065468E-03;
    COFETA[8] = -1.55270326E+01;
    COFETA[9] = 1.92766908E+00;
    COFETA[10] = -1.66518287E-01;
    COFETA[11] = 7.19100649E-03;
    COFETA[12] = -1.63532060E+01;
    COFETA[13] = 2.16753735E+00;
    COFETA[14] = -1.97226850E-01;
    COFETA[15] = 8.50065468E-03;
    COFETA[16] = -1.60066409E+01;
    COFETA[17] = 2.16753735E+00;
    COFETA[18] = -1.97226850E-01;
    COFETA[19] = 8.50065468E-03;
    COFETA[20] = -1.55270424E+01;
    COFETA[21] = 1.92766908E+00;
    COFETA[22] = -1.66518287E-01;
    COFETA[23] = 7.19100649E-03;
    COFETA[24] = -1.56600631E+01;
    COFETA[25] = 2.16753735E+00;
    COFETA[26] = -1.97226850E-01;
    COFETA[27] = 8.50065468E-03;
    COFETA[28] = -1.51804639E+01;
    COFETA[29] = 1.92766908E+00;
    COFETA[30] = -1.66518287E-01;
    COFETA[31] = 7.19100649E-03;
    COFETA[32] = -1.56600631E+01;
    COFETA[33] = 2.16753735E+00;
    COFETA[34] = -1.97226850E-01;
    COFETA[35] = 8.50065468E-03;
    COFETA[36] = -1.60066238E+01;
    COFETA[37] = 2.16753735E+00;
    COFETA[38] = -1.97226850E-01;
    COFETA[39] = 8.50065468E-03;
}


/*Poly fits for the conductivities, dim NO*KK */
void egtransetCOFLAM(double* COFLAM) {
    COFLAM[0] = 1.12880481E-01;
    COFLAM[1] = -7.88142336E-01;
    COFLAM[2] = 3.18038402E-01;
    COFLAM[3] = -1.85151697E-02;
    COFLAM[4] = -2.11869892E+00;
    COFLAM[5] = 2.98568651E+00;
    COFLAM[6] = -2.86879123E-01;
    COFLAM[7] = 1.23850873E-02;
    COFLAM[8] = 7.60997504E+00;
    COFLAM[9] = -1.18418698E+00;
    COFLAM[10] = 3.03558703E-01;
    COFLAM[11] = -1.54159597E-02;
    COFLAM[12] = -1.57251793E+00;
    COFLAM[13] = 3.13346556E+00;
    COFLAM[14] = -3.50321268E-01;
    COFLAM[15] = 1.64459217E-02;
    COFLAM[16] = 3.51230902E+00;
    COFLAM[17] = 5.95034560E-01;
    COFLAM[18] = 5.19422945E-02;
    COFLAM[19] = -3.70612240E-03;
    COFLAM[20] = 2.72369808E+00;
    COFLAM[21] = 1.21113046E+00;
    COFLAM[22] = -8.57758782E-02;
    COFLAM[23] = 5.58437448E-03;
    COFLAM[24] = -1.73558510E+01;
    COFLAM[25] = 8.92291669E+00;
    COFLAM[26] = -1.02660500E+00;
    COFLAM[27] = 4.25347987E-02;
    COFLAM[28] = -7.15494206E+00;
    COFLAM[29] = 4.80044518E+00;
    COFLAM[30] = -4.76327334E-01;
    COFLAM[31] = 1.82163138E-02;
    COFLAM[32] = -1.73558510E+01;
    COFLAM[33] = 8.92291669E+00;
    COFLAM[34] = -1.02660500E+00;
    COFLAM[35] = 4.25347987E-02;
    COFLAM[36] = -4.17001666E+00;
    COFLAM[37] = 3.85409566E+00;
    COFLAM[38] = -3.99415108E-01;
    COFLAM[39] = 1.68318766E-02;
}


/*Poly fits for the diffusion coefficients, dim NO*KK*KK */
void egtransetCOFD(double* COFD) {
    COFD[0] = -1.82216966E+01;
    COFD[1] = 1.68992430E+00;
    COFD[2] = 1.09701574E-01;
    COFD[3] = -8.81888811E-03;
    COFD[4] = -2.43418402E+01;
    COFD[5] = 5.33526425E+00;
    COFD[6] = -4.46630688E-01;
    COFD[7] = 1.81086658E-02;
    COFD[8] = -2.41515313E+01;
    COFD[9] = 5.28689054E+00;
    COFD[10] = -4.42932076E-01;
    COFD[11] = 1.80445515E-02;
    COFD[12] = -2.43418316E+01;
    COFD[13] = 5.33526425E+00;
    COFD[14] = -4.46630688E-01;
    COFD[15] = 1.81086658E-02;
    COFD[16] = -2.43418402E+01;
    COFD[17] = 5.33526425E+00;
    COFD[18] = -4.46630688E-01;
    COFD[19] = 1.81086658E-02;
    COFD[20] = -2.41515313E+01;
    COFD[21] = 5.28689054E+00;
    COFD[22] = -4.42932076E-01;
    COFD[23] = 1.80445515E-02;
    COFD[24] = -2.43418445E+01;
    COFD[25] = 5.33526425E+00;
    COFD[26] = -4.46630688E-01;
    COFD[27] = 1.81086658E-02;
    COFD[28] = -2.41515362E+01;
    COFD[29] = 5.28689054E+00;
    COFD[30] = -4.42932076E-01;
    COFD[31] = 1.80445515E-02;
    COFD[32] = -2.43418445E+01;
    COFD[33] = 5.33526425E+00;
    COFD[34] = -4.46630688E-01;
    COFD[35] = 1.81086658E-02;
    COFD[36] = -2.43418402E+01;
    COFD[37] = 5.33526425E+00;
    COFD[38] = -4.46630688E-01;
    COFD[39] = 1.81086658E-02;
    COFD[40] = -2.43418402E+01;
    COFD[41] = 5.33526425E+00;
    COFD[42] = -4.46630688E-01;
    COFD[43] = 1.81086658E-02;
    COFD[44] = -1.47079646E+01;
    COFD[45] = 3.10657376E+00;
    COFD[46] = -1.85922460E-01;
    COFD[47] = 7.92680827E-03;
    COFD[48] = -1.44285949E+01;
    COFD[49] = 2.99858376E+00;
    COFD[50] = -1.72232643E-01;
    COFD[51] = 7.34804765E-03;
    COFD[52] = -1.45052320E+01;
    COFD[53] = 3.10657376E+00;
    COFD[54] = -1.85922460E-01;
    COFD[55] = 7.92680827E-03;
    COFD[56] = -1.47079603E+01;
    COFD[57] = 3.10657376E+00;
    COFD[58] = -1.85922460E-01;
    COFD[59] = 7.92680827E-03;
    COFD[60] = -1.44285897E+01;
    COFD[61] = 2.99858376E+00;
    COFD[62] = -1.72232643E-01;
    COFD[63] = 7.34804765E-03;
    COFD[64] = -1.48518042E+01;
    COFD[65] = 3.10657376E+00;
    COFD[66] = -1.85922460E-01;
    COFD[67] = 7.92680827E-03;
    COFD[68] = -1.45836268E+01;
    COFD[69] = 2.99858376E+00;
    COFD[70] = -1.72232643E-01;
    COFD[71] = 7.34804765E-03;
    COFD[72] = -1.48518042E+01;
    COFD[73] = 3.10657376E+00;
    COFD[74] = -1.85922460E-01;
    COFD[75] = 7.92680827E-03;
    COFD[76] = -1.47079688E+01;
    COFD[77] = 3.10657376E+00;
    COFD[78] = -1.85922460E-01;
    COFD[79] = 7.92680827E-03;
    COFD[80] = -2.41515313E+01;
    COFD[81] = 5.28689054E+00;
    COFD[82] = -4.42932076E-01;
    COFD[83] = 1.80445515E-02;
    COFD[84] = -1.44285949E+01;
    COFD[85] = 2.99858376E+00;
    COFD[86] = -1.72232643E-01;
    COFD[87] = 7.34804765E-03;
    COFD[88] = -1.42056656E+01;
    COFD[89] = 2.91297621E+00;
    COFD[90] = -1.61544771E-01;
    COFD[91] = 6.90271324E-03;
    COFD[92] = -1.42370550E+01;
    COFD[93] = 2.99858376E+00;
    COFD[94] = -1.72232643E-01;
    COFD[95] = 7.34804765E-03;
    COFD[96] = -1.44285909E+01;
    COFD[97] = 2.99858376E+00;
    COFD[98] = -1.72232643E-01;
    COFD[99] = 7.34804765E-03;
    COFD[100] = -1.42056607E+01;
    COFD[101] = 2.91297621E+00;
    COFD[102] = -1.61544771E-01;
    COFD[103] = 6.90271324E-03;
    COFD[104] = -1.45614871E+01;
    COFD[105] = 2.99858376E+00;
    COFD[106] = -1.72232643E-01;
    COFD[107] = 7.34804765E-03;
    COFD[108] = -1.43495050E+01;
    COFD[109] = 2.91297621E+00;
    COFD[110] = -1.61544771E-01;
    COFD[111] = 6.90271324E-03;
    COFD[112] = -1.45614871E+01;
    COFD[113] = 2.99858376E+00;
    COFD[114] = -1.72232643E-01;
    COFD[115] = 7.34804765E-03;
    COFD[116] = -1.44285989E+01;
    COFD[117] = 2.99858376E+00;
    COFD[118] = -1.72232643E-01;
    COFD[119] = 7.34804765E-03;
    COFD[120] = -2.43418316E+01;
    COFD[121] = 5.33526425E+00;
    COFD[122] = -4.46630688E-01;
    COFD[123] = 1.81086658E-02;
    COFD[124] = -1.45052320E+01;
    COFD[125] = 3.10657376E+00;
    COFD[126] = -1.85922460E-01;
    COFD[127] = 7.92680827E-03;
    COFD[128] = -1.42370550E+01;
    COFD[129] = 2.99858376E+00;
    COFD[130] = -1.72232643E-01;
    COFD[131] = 7.34804765E-03;
    COFD[132] = -1.43613910E+01;
    COFD[133] = 3.10657376E+00;
    COFD[134] = -1.85922460E-01;
    COFD[135] = 7.92680827E-03;
    COFD[136] = -1.45052291E+01;
    COFD[137] = 3.10657376E+00;
    COFD[138] = -1.85922460E-01;
    COFD[139] = 7.92680827E-03;
    COFD[140] = -1.42370515E+01;
    COFD[141] = 2.99858376E+00;
    COFD[142] = -1.72232643E-01;
    COFD[143] = 7.34804765E-03;
    COFD[144] = -1.45963919E+01;
    COFD[145] = 3.10657376E+00;
    COFD[146] = -1.85922460E-01;
    COFD[147] = 7.92680827E-03;
    COFD[148] = -1.43373528E+01;
    COFD[149] = 2.99858376E+00;
    COFD[150] = -1.72232643E-01;
    COFD[151] = 7.34804765E-03;
    COFD[152] = -1.45963919E+01;
    COFD[153] = 3.10657376E+00;
    COFD[154] = -1.85922460E-01;
    COFD[155] = 7.92680827E-03;
    COFD[156] = -1.45052349E+01;
    COFD[157] = 3.10657376E+00;
    COFD[158] = -1.85922460E-01;
    COFD[159] = 7.92680827E-03;
    COFD[160] = -2.43418402E+01;
    COFD[161] = 5.33526425E+00;
    COFD[162] = -4.46630688E-01;
    COFD[163] = 1.81086658E-02;
    COFD[164] = -1.47079603E+01;
    COFD[165] = 3.10657376E+00;
    COFD[166] = -1.85922460E-01;
    COFD[167] = 7.92680827E-03;
    COFD[168] = -1.44285909E+01;
    COFD[169] = 2.99858376E+00;
    COFD[170] = -1.72232643E-01;
    COFD[171] = 7.34804765E-03;
    COFD[172] = -1.45052291E+01;
    COFD[173] = 3.10657376E+00;
    COFD[174] = -1.85922460E-01;
    COFD[175] = 7.92680827E-03;
    COFD[176] = -1.47079560E+01;
    COFD[177] = 3.10657376E+00;
    COFD[178] = -1.85922460E-01;
    COFD[179] = 7.92680827E-03;
    COFD[180] = -1.44285857E+01;
    COFD[181] = 2.99858376E+00;
    COFD[182] = -1.72232643E-01;
    COFD[183] = 7.34804765E-03;
    COFD[184] = -1.48517985E+01;
    COFD[185] = 3.10657376E+00;
    COFD[186] = -1.85922460E-01;
    COFD[187] = 7.92680827E-03;
    COFD[188] = -1.45836214E+01;
    COFD[189] = 2.99858376E+00;
    COFD[190] = -1.72232643E-01;
    COFD[191] = 7.34804765E-03;
    COFD[192] = -1.48517985E+01;
    COFD[193] = 3.10657376E+00;
    COFD[194] = -1.85922460E-01;
    COFD[195] = 7.92680827E-03;
    COFD[196] = -1.47079646E+01;
    COFD[197] = 3.10657376E+00;
    COFD[198] = -1.85922460E-01;
    COFD[199] = 7.92680827E-03;
    COFD[200] = -2.41515313E+01;
    COFD[201] = 5.28689054E+00;
    COFD[202] = -4.42932076E-01;
    COFD[203] = 1.80445515E-02;
    COFD[204] = -1.44285897E+01;
    COFD[205] = 2.99858376E+00;
    COFD[206] = -1.72232643E-01;
    COFD[207] = 7.34804765E-03;
    COFD[208] = -1.42056607E+01;
    COFD[209] = 2.91297621E+00;
    COFD[210] = -1.61544771E-01;
    COFD[211] = 6.90271324E-03;
    COFD[212] = -1.42370515E+01;
    COFD[213] = 2.99858376E+00;
    COFD[214] = -1.72232643E-01;
    COFD[215] = 7.34804765E-03;
    COFD[216] = -1.44285857E+01;
    COFD[217] = 2.99858376E+00;
    COFD[218] = -1.72232643E-01;
    COFD[219] = 7.34804765E-03;
    COFD[220] = -1.42056558E+01;
    COFD[221] = 2.91297621E+00;
    COFD[222] = -1.61544771E-01;
    COFD[223] = 6.90271324E-03;
    COFD[224] = -1.45614803E+01;
    COFD[225] = 2.99858376E+00;
    COFD[226] = -1.72232643E-01;
    COFD[227] = 7.34804765E-03;
    COFD[228] = -1.43494984E+01;
    COFD[229] = 2.91297621E+00;
    COFD[230] = -1.61544771E-01;
    COFD[231] = 6.90271324E-03;
    COFD[232] = -1.45614803E+01;
    COFD[233] = 2.99858376E+00;
    COFD[234] = -1.72232643E-01;
    COFD[235] = 7.34804765E-03;
    COFD[236] = -1.44285937E+01;
    COFD[237] = 2.99858376E+00;
    COFD[238] = -1.72232643E-01;
    COFD[239] = 7.34804765E-03;
    COFD[240] = -2.43418445E+01;
    COFD[241] = 5.33526425E+00;
    COFD[242] = -4.46630688E-01;
    COFD[243] = 1.81086658E-02;
    COFD[244] = -1.48518042E+01;
    COFD[245] = 3.10657376E+00;
    COFD[246] = -1.85922460E-01;
    COFD[247] = 7.92680827E-03;
    COFD[248] = -1.45614871E+01;
    COFD[249] = 2.99858376E+00;
    COFD[250] = -1.72232643E-01;
    COFD[251] = 7.34804765E-03;
    COFD[252] = -1.45963919E+01;
    COFD[253] = 3.10657376E+00;
    COFD[254] = -1.85922460E-01;
    COFD[255] = 7.92680827E-03;
    COFD[256] = -1.48517985E+01;
    COFD[257] = 3.10657376E+00;
    COFD[258] = -1.85922460E-01;
    COFD[259] = 7.92680827E-03;
    COFD[260] = -1.45614803E+01;
    COFD[261] = 2.99858376E+00;
    COFD[262] = -1.72232643E-01;
    COFD[263] = 7.34804765E-03;
    COFD[264] = -1.50545339E+01;
    COFD[265] = 3.10657376E+00;
    COFD[266] = -1.85922460E-01;
    COFD[267] = 7.92680827E-03;
    COFD[268] = -1.47751639E+01;
    COFD[269] = 2.99858376E+00;
    COFD[270] = -1.72232643E-01;
    COFD[271] = 7.34804765E-03;
    COFD[272] = -1.50545339E+01;
    COFD[273] = 3.10657376E+00;
    COFD[274] = -1.85922460E-01;
    COFD[275] = 7.92680827E-03;
    COFD[276] = -1.48518099E+01;
    COFD[277] = 3.10657376E+00;
    COFD[278] = -1.85922460E-01;
    COFD[279] = 7.92680827E-03;
    COFD[280] = -2.41515362E+01;
    COFD[281] = 5.28689054E+00;
    COFD[282] = -4.42932076E-01;
    COFD[283] = 1.80445515E-02;
    COFD[284] = -1.45836268E+01;
    COFD[285] = 2.99858376E+00;
    COFD[286] = -1.72232643E-01;
    COFD[287] = 7.34804765E-03;
    COFD[288] = -1.43495050E+01;
    COFD[289] = 2.91297621E+00;
    COFD[290] = -1.61544771E-01;
    COFD[291] = 6.90271324E-03;
    COFD[292] = -1.43373528E+01;
    COFD[293] = 2.99858376E+00;
    COFD[294] = -1.72232643E-01;
    COFD[295] = 7.34804765E-03;
    COFD[296] = -1.45836214E+01;
    COFD[297] = 2.99858376E+00;
    COFD[298] = -1.72232643E-01;
    COFD[299] = 7.34804765E-03;
    COFD[300] = -1.43494984E+01;
    COFD[301] = 2.91297621E+00;
    COFD[302] = -1.61544771E-01;
    COFD[303] = 6.90271324E-03;
    COFD[304] = -1.47751639E+01;
    COFD[305] = 2.99858376E+00;
    COFD[306] = -1.72232643E-01;
    COFD[307] = 7.34804765E-03;
    COFD[308] = -1.45522342E+01;
    COFD[309] = 2.91297621E+00;
    COFD[310] = -1.61544771E-01;
    COFD[311] = 6.90271324E-03;
    COFD[312] = -1.47751639E+01;
    COFD[313] = 2.99858376E+00;
    COFD[314] = -1.72232643E-01;
    COFD[315] = 7.34804765E-03;
    COFD[316] = -1.45836323E+01;
    COFD[317] = 2.99858376E+00;
    COFD[318] = -1.72232643E-01;
    COFD[319] = 7.34804765E-03;
    COFD[320] = -2.43418445E+01;
    COFD[321] = 5.33526425E+00;
    COFD[322] = -4.46630688E-01;
    COFD[323] = 1.81086658E-02;
    COFD[324] = -1.48518042E+01;
    COFD[325] = 3.10657376E+00;
    COFD[326] = -1.85922460E-01;
    COFD[327] = 7.92680827E-03;
    COFD[328] = -1.45614871E+01;
    COFD[329] = 2.99858376E+00;
    COFD[330] = -1.72232643E-01;
    COFD[331] = 7.34804765E-03;
    COFD[332] = -1.45963919E+01;
    COFD[333] = 3.10657376E+00;
    COFD[334] = -1.85922460E-01;
    COFD[335] = 7.92680827E-03;
    COFD[336] = -1.48517985E+01;
    COFD[337] = 3.10657376E+00;
    COFD[338] = -1.85922460E-01;
    COFD[339] = 7.92680827E-03;
    COFD[340] = -1.45614803E+01;
    COFD[341] = 2.99858376E+00;
    COFD[342] = -1.72232643E-01;
    COFD[343] = 7.34804765E-03;
    COFD[344] = -1.50545339E+01;
    COFD[345] = 3.10657376E+00;
    COFD[346] = -1.85922460E-01;
    COFD[347] = 7.92680827E-03;
    COFD[348] = -1.47751639E+01;
    COFD[349] = 2.99858376E+00;
    COFD[350] = -1.72232643E-01;
    COFD[351] = 7.34804765E-03;
    COFD[352] = -1.50545339E+01;
    COFD[353] = 3.10657376E+00;
    COFD[354] = -1.85922460E-01;
    COFD[355] = 7.92680827E-03;
    COFD[356] = -1.48518099E+01;
    COFD[357] = 3.10657376E+00;
    COFD[358] = -1.85922460E-01;
    COFD[359] = 7.92680827E-03;
    COFD[360] = -2.43418402E+01;
    COFD[361] = 5.33526425E+00;
    COFD[362] = -4.46630688E-01;
    COFD[363] = 1.81086658E-02;
    COFD[364] = -1.47079688E+01;
    COFD[365] = 3.10657376E+00;
    COFD[366] = -1.85922460E-01;
    COFD[367] = 7.92680827E-03;
    COFD[368] = -1.44285989E+01;
    COFD[369] = 2.99858376E+00;
    COFD[370] = -1.72232643E-01;
    COFD[371] = 7.34804765E-03;
    COFD[372] = -1.45052349E+01;
    COFD[373] = 3.10657376E+00;
    COFD[374] = -1.85922460E-01;
    COFD[375] = 7.92680827E-03;
    COFD[376] = -1.47079646E+01;
    COFD[377] = 3.10657376E+00;
    COFD[378] = -1.85922460E-01;
    COFD[379] = 7.92680827E-03;
    COFD[380] = -1.44285937E+01;
    COFD[381] = 2.99858376E+00;
    COFD[382] = -1.72232643E-01;
    COFD[383] = 7.34804765E-03;
    COFD[384] = -1.48518099E+01;
    COFD[385] = 3.10657376E+00;
    COFD[386] = -1.85922460E-01;
    COFD[387] = 7.92680827E-03;
    COFD[388] = -1.45836323E+01;
    COFD[389] = 2.99858376E+00;
    COFD[390] = -1.72232643E-01;
    COFD[391] = 7.34804765E-03;
    COFD[392] = -1.48518099E+01;
    COFD[393] = 3.10657376E+00;
    COFD[394] = -1.85922460E-01;
    COFD[395] = 7.92680827E-03;
    COFD[396] = -1.47079731E+01;
    COFD[397] = 3.10657376E+00;
    COFD[398] = -1.85922460E-01;
    COFD[399] = 7.92680827E-03;
}


/*List of specs with small weight, dim NLITE */
void egtransetKTDIF(int* KTDIF) {
    KTDIF[0] = 1;
}


/*Poly fits for thermal diff ratios, dim NO*NLITE*KK */
void egtransetCOFTD(double* COFTD) {
    COFTD[0] = 0.00000000E+00;
    COFTD[1] = 0.00000000E+00;
    COFTD[2] = 0.00000000E+00;
    COFTD[3] = 0.00000000E+00;
    COFTD[4] = -8.18816846E-02;
    COFTD[5] = 6.25102119E-04;
    COFTD[6] = -1.93882484E-07;
    COFTD[7] = 1.90693525E-11;
    COFTD[8] = -6.27142420E-02;
    COFTD[9] = 6.17357858E-04;
    COFTD[10] = -1.93698207E-07;
    COFTD[11] = 1.91874330E-11;
    COFTD[12] = -8.18788771E-02;
    COFTD[13] = 6.25080686E-04;
    COFTD[14] = -1.93875836E-07;
    COFTD[15] = 1.90686987E-11;
    COFTD[16] = -8.18816845E-02;
    COFTD[17] = 6.25102118E-04;
    COFTD[18] = -1.93882484E-07;
    COFTD[19] = 1.90693525E-11;
    COFTD[20] = -6.27142420E-02;
    COFTD[21] = 6.17357858E-04;
    COFTD[22] = -1.93698207E-07;
    COFTD[23] = 1.91874330E-11;
    COFTD[24] = -8.18830883E-02;
    COFTD[25] = 6.25112835E-04;
    COFTD[26] = -1.93885808E-07;
    COFTD[27] = 1.90696794E-11;
    COFTD[28] = -6.27154702E-02;
    COFTD[29] = 6.17369948E-04;
    COFTD[30] = -1.93702001E-07;
    COFTD[31] = 1.91878088E-11;
    COFTD[32] = -8.18830883E-02;
    COFTD[33] = 6.25112835E-04;
    COFTD[34] = -1.93885808E-07;
    COFTD[35] = 1.90696794E-11;
    COFTD[36] = -8.18816846E-02;
    COFTD[37] = 6.25102119E-04;
    COFTD[38] = -1.93882484E-07;
    COFTD[39] = 1.90693525E-11;
}

/* Replace this routine with the one generated by the Gauss Jordan solver of DW */
AMREX_GPU_HOST_DEVICE void sgjsolve(double* A, double* x, double* b) {
    amrex::Abort("sgjsolve not implemented, choose a different solver ");
}

/* Replace this routine with the one generated by the Gauss Jordan solver of DW */
AMREX_GPU_HOST_DEVICE void sgjsolve_simplified(double* A, double* x, double* b) {
    amrex::Abort("sgjsolve_simplified not implemented, choose a different solver ");
}


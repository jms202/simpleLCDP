// Program to solve a lifecycle consumption-savings problem with AR(1)
// income

// Copyright (c) 2017, 2018 Jonathan Shaw

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.


// To compile:
// g++ -I /usr/local/boost_1_67_0/ -std=c++11 simpleLCDP.cpp -o simpleLCDP -lgsl -lgslcblas -lm

// To solve, simulate and plot
// ./simpleLCDP

// Next steps:
// - Add solveUsingValue function

// See https://www.gnu.org/software/gsl/manual/html_node/Interpolation.html
// https://www.gnu.org/software/gsl/doc/html/usage.html


//include <iostream>
using std::cout;
using std::endl;
#include <cmath>
using std::max;
#include <cstdio>
#include <stdlib.h>
#include <boost/math/distributions/normal.hpp>
using boost::math::normal;
#include <random>
using std::mt19937;
using std::normal_distribution;
using std::setw;
#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_interp2d.h>


// 1D linear interpolation
double myinterp(double *xgrid, double *ygrid, int len, double xval) {

    int ix;
    double mu;

    // Find higher of two grid points that bound xval
    for (ix = 0; ix < len; ++ix) {
        if (xgrid[ix] > xval)
            break;
    }
    if (ix == 0) ++ix;
    else if (ix == len) --ix;

    // Find fraction of distance from lower grid point (values of mu outside unit interval are allowed)
    mu = (xval - xgrid[ix-1]) / (xgrid[ix] - xgrid[ix-1]);
    
    // Calculate interpolated value
    return ygrid[ix-1]*(1. - mu) + ygrid[ix]*mu;

}


// 1D linear interpolation using GSL
double interp(double *xgrid, double *ygrid, int len, double xval) {

    double value;

    gsl_interp *linear_interp = gsl_interp_alloc(gsl_interp_linear, len);
    gsl_interp_init(linear_interp, xgrid, ygrid, len);
    gsl_interp_accel *accel =  gsl_interp_accel_alloc();
    value = gsl_interp_eval(linear_interp, xgrid, ygrid, xval, accel);
    gsl_interp_accel_free(accel);
    gsl_interp_free(linear_interp);

    return value;

}

// 2D bicubic interpolation using GSL
// Assets is j (or y), Income is i (or x)
double interp2d(double *xgrid, double *ygrid, double *zgrid, int xlen, int ylen, double xval, double yval) {

    double value;

    gsl_interp2d *bicubic_interp2d = gsl_interp2d_alloc(gsl_interp2d_bicubic, xlen, ylen);
    gsl_interp2d_init(bicubic_interp2d, xgrid, ygrid, zgrid, xlen, ylen);
    gsl_interp_accel *xaccel =  gsl_interp_accel_alloc();
    gsl_interp_accel *yaccel =  gsl_interp_accel_alloc();
    value = gsl_interp2d_eval_extrap(bicubic_interp2d, xgrid, ygrid, zgrid, xval, yval, xaccel, yaccel);
    gsl_interp_accel_free(xaccel);
    gsl_interp_accel_free(yaccel);
    gsl_interp2d_free(bicubic_interp2d);

    return value;

}




// Environment class
class Env {
friend class Model;
    double tol;
    double minCons;
    int T;
    int tretire;
    double r;
    double beta;
    double gamma;
    double mu;
    double rho;
    double sigma;
    double truncAt;
    int startA;
    int numPointsA;
    int numPointsY;
    bool borrowingAllowed;
    double *Q;
    double *Ygrid;
    double *Ymin;
    double *Ymax;
    double *Agrid;
    int numSims;
    void initYgrid(void);
    void initAgrid(void);
    void finYgrid(void);
    void finAgrid(void);
    double utility(double cons);
    double marginalUtility(double cons);
    double inverseMarginalUtility(double margUtils);
public:
    void init(void);
    void fin(void);
};


// Initialize environment
void Env::init(void) {
    tol = 1e-10;
    minCons = 1e-5;
    T = 20;             // First period after end of life
    tretire = T;        // First period of retirement
    r = 0.015;
    beta = 0.98;
    gamma = 1.5;
    mu = 0.0;
    rho = 0.75;
    sigma = 0.25;
    truncAt = 3.0;
    startA = 0;
    numPointsA = 8;
    numPointsY = 7;
    numSims = 100;
    initYgrid();
    borrowingAllowed = true;
    initAgrid();
}


// Initialize Ygrid
void Env::initYgrid(void) {

    int ixi, ixj, ixY;
    double sigmalny;
    double *lnY;
    double *lnYbounds;

    // Income process is lny = rho*lny_1 + (1-rho)*mu + epsilon
    // Mean is mu (ignores the truncation points)

    // Find std dev (ignores the truncation points)
    sigmalny = sigma / sqrt(1.0 - pow(rho, 2));
    // cout << "sigmalny is: " << sigmalny << endl;

    // Find the boundary points and expected values between these boundary points

    // Allocate arrays
    lnY = new double[numPointsY];
    lnYbounds = new double[numPointsY+1];

    // Set lowest and highest bounds at truncation points
    lnYbounds[0] = mu - (truncAt * sigmalny);
    lnYbounds[numPointsY] = mu + (truncAt * sigmalny);

    // Set other bounds by splitting distribution into equal probability areas ignoring the truncation
    normal norm(mu, sigmalny);
    for(ixY = 1; ixY < numPointsY; ++ixY)
        lnYbounds[ixY] = quantile(norm, (double)ixY / (double)numPointsY);

    /*
    cout << "lnYbounds values:" << endl;
    for(ixY = 0; ixY <= numPointsY; ++ixY)
        cout << setw(10) << lnYbounds[ixY];
    cout << endl;
    */

    // Find the N expected values from between the boundary points
    normal stdnorm;
    double pdf0, pdf1;
    for(ixY = 0; ixY < numPointsY; ++ixY) {
        pdf0 = pdf(stdnorm, (lnYbounds[ixY] - mu) / sigmalny);
        pdf1 = pdf(stdnorm, (lnYbounds[ixY+1] - mu) / sigmalny);
        lnY[ixY] = ((double)numPointsY * sigmalny * (pdf0 - pdf1)) + mu;
    }
    /*
    cout << "lnY values:" << endl;
    for(ixY = 0; ixY < numPointsY; ++ixY)
        cout << setw(10) << lnY[ixY];
    cout << endl;
    */

    // Allocate array Q (2D array as 1D)
    Q = new double[numPointsY*numPointsY];

    // Find probability of transitioning between different ranges
    double minDraw, maxDraw, sumj;
    for (ixi = 0; ixi < numPointsY; ++ixi) {
        sumj = 0.;
        for (ixj = 0; ixj < numPointsY; ++ixj) {
            minDraw = lnYbounds[ixj] - (rho * mu) - (rho * lnY[ixi]);
            maxDraw = lnYbounds[ixj+1] - (rho * mu) - (rho * lnY[ixi]);
            Q[ixi*numPointsY + ixj] = cdf(stdnorm, maxDraw / sigma) - cdf(stdnorm, minDraw / sigma);
            sumj += Q[ixi*numPointsY + ixj];
        }
        // Next loop needed to ensure probabilities sum to one (which they won't due to truncation)
        for (ixj = 0; ixj < numPointsY; ++ixj)
            Q[ixi*numPointsY + ixj] = Q[ixi*numPointsY + ixj] / sumj;
    }
    
    /*
    cout << "Q values:" << endl;
    for(ixj = 0; ixj < numPointsY; ++ixj)
        cout << setw(10) << Q[numPointsY + ixj];
    cout << endl;
    */

    // Allocate 2D array Ygrid[ixt][ixY]
    int ixt;
    Ygrid = new double[T*numPointsY];

    // Allocate 1D arrays Ymin[ixt] and Ymax[ixt]
    Ymin = new double[T];
    Ymax = new double[T];

    // Convert log income into income
    for (ixt = 0; ixt < T; ++ixt) {
        for (ixY = 0; ixY < numPointsY; ++ixY)
            Ygrid[ixt*numPointsY + ixY] = exp(lnY[ixY]);
        Ymin[ixt] = exp(lnYbounds[0]);
        Ymax[ixt] = exp(lnYbounds[numPointsY]);
    }

    // Deal with retirement
    if (tretire < T) {
        for (ixt = max(0, tretire); ixt < T; ++ixt) {
            for (ixY = 0; ixY < numPointsY; ++ixY)
                Ygrid[ixt*numPointsY + ixY] = 0.;
            Ymin[ixt] = 0.;
            Ymax[ixt] = 0.;
        }
    }

    // Highlight potential numerical instability
    double minYgrid = Ygrid[0];
    double maxYgrid = Ygrid[0*numPointsY + numPointsY-1];
    for (ixt = 1; ixt < T; ++ixt) {
        minYgrid = fmin(minYgrid, Ygrid[ixt*numPointsY + 0]);
        maxYgrid = fmax(maxYgrid, Ygrid[ixt*numPointsY + numPointsY-1]);
    }
    if ((minYgrid < 1e-4) || (maxYgrid > 1e5))
        cout << "Combination of sigma and rho give very high income variance - numerical instability possible";

    // Free arrays
    delete [] lnY;
    delete [] lnYbounds;
    
}

// Initialize Agrid
void Env::initAgrid(void) {

    // Allocate Agrid array (2D [ixt][ixA] as 1D)
    int ixt;
    Agrid = new double[(T+1)*numPointsA];

    // Maximum assets
    Agrid[0*numPointsA + numPointsA-1] = startA;
    for (ixt = 1; ixt <= T; ++ixt)
        Agrid[ixt*numPointsA + numPointsA-1] = (Agrid[(ixt-1)*numPointsA + numPointsA-1] + Ymax[ixt-1] - minCons) * (1. + r);
        
    // Minumum assets
    Agrid[T*numPointsA + 0]= 0.;
    for (ixt = T-1; ixt >= 0; --ixt) {
        Agrid[ixt*numPointsA + 0] = (Agrid[(ixt+1)*numPointsA + 0] / (1. + r)) - Ymin[ixt] + minCons;
        if ((!borrowingAllowed) && (Agrid[ixt*numPointsA + 0] < 0.)) Agrid[ixt*numPointsA + 0] = 0.;
    }
    
    /*
    cout << "Agrid[0*numPointsA + 0] is: " << Agrid[0*numPointsA + 0] << endl;
    cout << "Agrid[0*numPointsA + numPointsA-1] is: " << Agrid[0*numPointsA + numPointsA-1] << endl;
    */

    // Asset points in between
    double span, lo, hi;
    int ixA;
    double loggrid[numPointsA], grid[numPointsA];
    double AgridLo;
    for (ixt = 0; ixt <= T; ++ixt) {
        span = Agrid[ixt*numPointsA + numPointsA-1] - Agrid[ixt*numPointsA + 0];
        lo = 0.;
        hi = log(1. + log(1. + log(1. + span)));
        // Split interval between lo and hi into equal sized subintervals using numPointsA points
        for (ixA = 0; ixA < numPointsA; ++ixA)
            loggrid[ixA] = lo + ((double)ixA * (hi - lo) / (double)(numPointsA-1));
        // Convert to non-logged grid
        for (ixA = 0; ixA < numPointsA; ++ixA)
            grid[ixA] = exp(exp(exp(loggrid[ixA]) - 1) - 1) - 1;
        // Add on bottom of grid
        AgridLo = Agrid[ixt*numPointsA + 0];
        for (ixA = 0; ixA < numPointsA; ++ixA)
            Agrid[ixt*numPointsA + ixA] = grid[ixA] + AgridLo;
    }
    
    /*
    cout << "A values:" << endl;
    for(ixA = 0; ixA < numPointsA; ++ixA)
        cout << setw(10) << Agrid[T*numPointsA + ixA];
    cout << endl;
    */
    
}


// Finalize environment
void Env::fin(void) {
    finYgrid();
    finAgrid();
}

// Finalize Ygrid
void Env::finYgrid(void) {

    // free memory allocated for Q
    delete [] Q;

    // free memory allocated for Ygrid
    delete [] Ygrid;

    // Free memory allocated to Ymin, Ymax
    delete [] Ymin;
    delete [] Ymax;


}

// Finalize Agrid
void Env::finAgrid(void) {

    int ixt;
    
    // free memory allocated for Agrid
    delete [] Agrid;

}


// Utility function
double Env::utility(double cons) {
    if (gamma == 1.) return log(cons);
    else return (pow(cons, 1. - gamma) / (1. - gamma));
}

// Marginal utility function
double Env::marginalUtility(double cons) {
    return pow(cons, -gamma);
}

// Inverse marginal utility function
double Env::inverseMarginalUtility(double margUtils) {
    return pow(margUtils, -1. / gamma);
}


class Sln {
friend class Model;
    double *value;
    double *Evalue;
    double *policyC;
    double *policyA1;
    double *margUtil;
    double *EmargUtil;
public:
    void init(int T, int numPointsA, int numPointsY);
    void fin(void);
};


// Initialize solution
void Sln::init(int T, int numPointsA, int numPointsY) {
    
    // Allocate value (3D [ixt][ixA][ixY] as 1D - note T+1 in time dimension)
    value = new double[(T+1)*numPointsA*numPointsY];

    // Allocate Evalue (3D [ixt][ixA][ixY] as 1D - note T+1 in time dimension)
    Evalue = new double[(T+1)*numPointsA*numPointsY];

    // Allocate policyC
    policyC = new double[T*numPointsA*numPointsY];

    // Allocate policyA1
    policyA1 = new double[T*numPointsA*numPointsY];

    // Allocate margUtil
    margUtil = new double[T*numPointsA*numPointsY];

    // Allocate EmargUtil
    EmargUtil = new double[T*numPointsA*numPointsY];


}

// Finalize solution
void Sln::fin(void) {

    // Deallocate value
    delete [] value;

    // Deallocate Evalue
    delete [] Evalue;

    // Deallocate policyC
    delete [] policyC;

    // Deallocate policyA1
    delete [] policyA1;

    // Deallocate margUtil
    delete [] margUtil;

    // Deallocate EmargUtil
    delete [] EmargUtil;

}


class Sim {
friend class Model;
    double *v;
    double *y;
    double *A;
    double *c;
public:
    void init(int T, int numSims);
    void fin(void);
};

// Initialize simulations
void Sim::init(int numSims, int T) {
    
    // Allocate value (2D [ixS][ixt] as 1D)
    v = new double[numSims*T];

    // Allocate income (2D [ixS][ixt] as 1D)
    y = new double[numSims*T];

    // Allocate assets (2D [ixS][ixt] as 1D)
    A = new double[numSims*T];

    // Allocate consumption (2D [ixS][ixt] as 1D)
    c = new double[numSims*T];

}

// Finalize simulations
void Sim::fin(void) {

    // Deallocate value
    delete [] v;

    // Deallocate income
    delete [] y;

    // Deallocate assets
    delete [] A;

    // Deallocate consumption
    delete [] c;

}




class Model {
    Env env;
    Sln sln;
    Sim sim;
    double objectiveFunc(int ixt, int ixY, double A0, double A1);
    double eulerDiff(int ixt, int ixY, double A0, double A1);
    double eulerZero(int ixt, int ixY, double A0, double min, double max);
public:
    void init(void);
    void fin(void);
    void printSomeStuff(void);
    void solveUsingEuler(void);
    void slnPlot(int ixt, char outcomeChar, char const *filename, char const *graphname);
    void simulate(void);
    void simPlot(int startSim, int endSim, char outcomeChar, char const *filename, char const *graphname);
};


void Model::init(void) {
    env.init();
    sln.init(env.T, env.numPointsA, env.numPointsY);
    sim.init(env.numSims, env.T);
}

void Model::fin(void) {
    env.fin();
    sln.fin();
    sim.fin();
}


void Model::printSomeStuff(void) {
    cout << "Sigma is: " << env.sigma << endl;
    cout << "Q[1*env.numPointsY + 3] is: " << env.Q[1*env.numPointsY + 3] << endl;
}

// Objective function
double Model::objectiveFunc(int ixt, int ixY, double A0, double A1) {

    double Y;
    double cons;
    double VA1;
    double *Evalue;
    int ixA;
    double value;
    
    Y = env.Ygrid[ixt*env.numPointsY + ixY];
    cons = A0 + Y - (A1 / (1. + env.r));

    // Copy Evalue to new 1D array
    Evalue = new double[env.numPointsA];
    for (ixA = 0; ixA < env.numPointsA; ++ixA)
        Evalue[ixA] = sln.Evalue[(ixt+1)*env.numPointsA*env.numPointsY + ixA*env.numPointsY + ixY];
    
    // Linearly interpolate
    // We want to pass a pointer to the first element of env.Agrid[]. "." and "[]" bind more tightly than "&" so the following should be OK
    VA1 = interp(&env.Agrid[(ixt+1)*env.numPointsA + 0], Evalue, env.numPointsA, A1);

    // Deallocate 1D array
    delete [] Evalue;

    // Return value
    value = env.utility(cons) + (env.beta * VA1);
    return -value;
}


// Function for solving the Euler equation
// Interpolates inverse marginal utility because this is linear
double Model::eulerDiff(int ixt, int ixY, double A0, double A1) {

    double Y;
    double *invMargUtil;
    int ixA;
    double invMargUtilAtA1;
    double margUtilAtA1;
    double cons;

    Y = env.Ygrid[ixt*env.numPointsY + ixY];

    // Copy inverse MU to new 1D array
    invMargUtil = new double[env.numPointsA];
    for (ixA = 0; ixA < env.numPointsA; ++ixA)
        invMargUtil[ixA] = env.inverseMarginalUtility(sln.EmargUtil[(ixt+1)*env.numPointsA*env.numPointsY + ixA*env.numPointsY + ixY]);
    
    // Linearly interpolate
    invMargUtilAtA1 = interp(&env.Agrid[(ixt+1)*env.numPointsA + 0], invMargUtil, env.numPointsA, A1);
    
    // Deallocate 1D array
    delete [] invMargUtil;

    // Recover and return Euler diff
    margUtilAtA1 = env.marginalUtility(invMargUtilAtA1);
    cons = A0 + Y - (A1 / (1. + env.r));
/*
cout << "cons: " << cons << endl;
cout << "MU today: " << env.marginalUtility(cons) << endl;
cout << "MU tomorrow: " << (env.beta * (1. + env.r) * margUtilAtA1) << endl;
cout << "invMargUtilAtA1: " << invMargUtilAtA1 << endl;
cout << "margUtilAtA1: " << margUtilAtA1 << endl;
cout << "invMargUtil: ";
for (ixA = 0; ixA < env.numPointsA; ++ixA)
    cout << invMargUtil[ixA] << " ";
cout << endl;
cout << "EmargUtil: ";
for (ixA = 0; ixA < env.numPointsA; ++ixA)
    cout << sln.EmargUtil[(ixt+1)*env.numPointsA*env.numPointsY + ixA*env.numPointsY + ixY] << " ";
cout << endl;
*/
    return env.marginalUtility(cons) - (env.beta * (1. + env.r) * margUtilAtA1);

}



// Function to find zero of Euler equation
double Model::eulerZero(int ixt, int ixY, double A0, double min, double max) {

  using namespace boost::math::tools;           // For bracket_and_solve_root.

  const boost::uintmax_t maxit = 100;           // Limit to maximum iterations.
  boost::uintmax_t it = maxit;                  // Initally our chosen max iterations, but updated with actual.
  int digits = std::numeric_limits<double>::digits;  // Maximum possible binary digits accuracy for type T.
  // Some fraction of digits is used to control how accurate to try to make the result.
  int get_digits = digits - 3;                  // We have to have a non-zero interval at each step, so
                                                // maximum accuracy is digits - 1.  But we also have to
                                                // allow for inaccuracy in f(x), otherwise the last few
                                                // iterations just thrash around.
  eps_tolerance<double> tol(get_digits);             // Set the tolerance.

  // In lambda, 'this' needs to be included in capture to get access to member variables and functions. Access to 'this' is by reference
  std::pair<double, double> r = bisect(
    [this, ixt, ixY, A0] (double A1) { return eulerDiff(ixt, ixY, A0, A1); },
    min, max, tol, it
    );

  return r.first + (r.second - r.first)/2;      // Midway between brackets is our result, if necessary we could

}




// Solution function based on Euler equation
void Model::solveUsingEuler(void) {

    int ixt, ixA, ixY, jxY, ix3D, ix3Dj;
    double A0;
    double Y;
    double lbA1, ubA1;
    int lbSign, ubSign;

    // Set terminal (T) value functions
    ixt = env.T;
    for (ixA = 0; ixA < env.numPointsA; ++ixA) {
        for (ixY = 0; ixY < env.numPointsY; ++ixY) {
            ix3D = ixt*env.numPointsA*env.numPointsY + ixA*env.numPointsY + ixY;
            sln.value[ix3D] = 0.;
            sln.Evalue[ix3D] = 0.;
        }
    }
    
    // Solve problem at T-1 (last period of life)
    ixt = env.T - 1;
    for (ixA = 0; ixA < env.numPointsA; ++ixA) {
        // Calculate policy and value functions
        for (ixY = 0; ixY < env.numPointsY; ++ixY) {
            ix3D = ixt*env.numPointsA*env.numPointsY + ixA*env.numPointsY + ixY;
            sln.policyC[ix3D] = env.Agrid[ixt*env.numPointsA + ixA] + env.Ygrid[ixt*env.numPointsY + ixY];
            sln.policyA1[ix3D] = 0.;
            sln.value[ix3D] = env.utility(sln.policyC[ix3D]);
            sln.margUtil[ix3D] = env.marginalUtility(sln.policyC[ix3D]);
        }
        // Calculate Emax functions
        for (ixY = 0; ixY < env.numPointsY; ++ixY) {
            ix3D = ixt*env.numPointsA*env.numPointsY + ixA*env.numPointsY + ixY;
            sln.Evalue[ix3D] = 0.;
            sln.EmargUtil[ix3D] = 0.;
            for (jxY = 0; jxY < env.numPointsY; ++jxY) {
                ix3Dj = ixt*env.numPointsA*env.numPointsY + ixA*env.numPointsY + jxY;
                sln.Evalue[ix3D] += env.Q[ixY*env.numPointsY + jxY] * sln.value[ix3Dj];
                sln.EmargUtil[ix3D] += env.Q[ixY*env.numPointsY + jxY] * sln.margUtil[ix3Dj];
            }
        }
        
    }
    // cout << "Period " << ixt << " complete" << endl;
    
    // Solve problem at T-2 back to 0
    for (ixt = env.T - 2; ixt >= 0; --ixt) {
        for (ixA = 0; ixA < env.numPointsA; ++ixA) {
            for (ixY = 0; ixY < env.numPointsY; ++ixY) {
            
                ix3D = ixt*env.numPointsA*env.numPointsY + ixA*env.numPointsY + ixY;
            
                // Information for optimisation
                A0 = env.Agrid[ixt*env.numPointsA + ixA];
                Y = env.Ygrid[ixt*env.numPointsY + ixY];
                lbA1 = env.Agrid[(ixt + 1)*env.numPointsA + 0];
                ubA1 = (A0 + Y - env.minCons) * (1. + env.r);
                
                // Compute solution
                lbSign = eulerDiff(ixt, ixY, A0, lbA1) > 0 ? 1 : -1;
                // If liquidity constrained
                if ((lbSign == 1) || (ubA1 - lbA1 < env.tol))
                    sln.policyA1[ix3D] = lbA1;
                else {
                    ubSign = eulerDiff(ixt, ixY, A0, ubA1) > 0 ? 1 : -1;
                    if (lbSign*ubSign == 1) {
                        cout << "Sign of Euler difference at lower and upper bounds are the same. No solution to Euler equation" << endl;
                        cout << "A0: " << A0 << endl;
                        cout << "Y: " << Y << endl;
                        cout << "lbA1: " << lbA1 << endl;
                        cout << "ubA1: " << ubA1 << endl;
                        cout << "lbSign: " << lbSign << endl;
                        cout << "ubSign: " << ubSign << endl;
                        exit(-1);
                    }
                    // Find zero of eulerDifference function
                    sln.policyA1[ix3D] = eulerZero(ixt, ixY, A0, lbA1, ubA1);
                }
                
                // Store solution
                sln.policyC[ix3D] = A0 + Y - (sln.policyA1[ix3D] / (1. + env.r));
                sln.value[ix3D] = -objectiveFunc(ixt, ixY, A0, sln.policyA1[ix3D]);
                sln.margUtil[ix3D] = env.marginalUtility(sln.policyC[ix3D]);
                
            }

            // Calculate Emax functions
            for (ixY = 0; ixY < env.numPointsY; ++ixY) {
                ix3D = ixt*env.numPointsA*env.numPointsY + ixA*env.numPointsY + ixY;
                sln.Evalue[ix3D] = 0.;
                sln.EmargUtil[ix3D] = 0.;
                for (jxY = 0; jxY < env.numPointsY; ++jxY) {
                    ix3Dj = ixt*env.numPointsA*env.numPointsY + ixA*env.numPointsY + jxY;
                    sln.Evalue[ix3D] += env.Q[ixY*env.numPointsY + jxY] * sln.value[ix3Dj];
                    sln.EmargUtil[ix3D] += env.Q[ixY*env.numPointsY + jxY] * sln.margUtil[ix3Dj];
                }
            }

        }
        // cout << "Period " << ixt << " complete" << endl;
    }
    
    
}

// Plot solution
void Model::slnPlot(int ixt, char outcomeChar, char const *filename, char const *graphname) {

    double *outcome;
    char const *outcometxt;
    int ixY, ixA, ix3D;
    FILE *fp;
    char shellcmd[256];
    
    switch (outcomeChar) {
        case 'v':
            outcome = sln.value;
            outcometxt = "Value";
            break;
        case 'E':
            outcome = sln.Evalue;
            outcometxt = "Evalue";
            break;
        case 'C':
            outcome = sln.policyC;
            outcometxt = "PolicyC";
            break;
        case 'A':
            outcome = sln.policyA1;
            outcometxt = "PolicyA1";
            break;
        case 'm':
            outcome = sln.margUtil;
            outcometxt = "MargUtil";
            break;
        case 'U':
            outcome = sln.EmargUtil;
            outcometxt = "EmargUtil";
            break;
    }

    fp = fopen(filename, "w");
    if (!fp) {
        printf("File %s could not be opened", filename);
        exit(-1);
    }

    for (ixY=0; ixY<env.numPointsY; ++ixY) {
        fprintf(fp, "#m=1,S=1\n");
        for (ixA=0; ixA<env.numPointsA; ++ixA) {
            ix3D = ixt*env.numPointsA*env.numPointsY + ixA*env.numPointsY + ixY;
            fprintf(fp, "%g %g\n", env.Agrid[ixt*env.numPointsA + ixA], outcome[ix3D]);
        }
    }

    fclose(fp);
    
    strcpy(shellcmd, "graph -T ps -X \"Assets\" -Y \"");
    strcat(shellcmd, outcometxt);
    strcat(shellcmd, "\" -L \"");
    strcat(shellcmd, outcometxt);
    strcat(shellcmd, " across assets and income\" -F Helvetica < ");
    strcat(shellcmd, filename);
    strcat(shellcmd, " > ");
    strcat(shellcmd, graphname);
    system(shellcmd);

}




// Simulation function
void Model::simulate(void) {

    int ixi, ixistart;
    int ixt;
    int ixA, ixY;
    int s;
    double *policyA1;
    double *value;

    // Other arrays that will be used
    double e[env.numSims*env.T];
    double lny1[env.numSims];
    double lny[env.numSims*env.T];
    
    // Find std. dev. (ignores truncation)
    double sigmalny = env.sigma / pow((1. - pow(env.rho, 2)), 0.5);

    mt19937 eng(42);
    normal_distribution<double> eDist(0.0, env.sigma), lnyDist(env.mu, sigmalny);

    for(ixi = 0; ixi < env.numSims*env.T; ++ixi)
        e[ixi] = eDist(eng);
    
    for(s = 0; s < env.numSims; ++s)
        lny1[s] = lnyDist(eng);

    for(s = 0; s < env.numSims; ++s) {
    
        ixistart = s * env.T;
        sim.A[ixistart] = env.startA;

        for(ixt = 0; ixt < env.T; ++ixt) {

            ixi = ixistart + ixt;

            // tretire is the first period of retirement
            if (ixt < env.tretire) {
            
                // Calculate lny
                if (ixt == 0) lny[ixi] = lny1[s];
                else lny[ixi] = ((1.0 - env.rho) * env.mu) + (env.rho * lny[ixi - 1]) + e[ixi];
                
                // Truncate (or is this censoring?!!)
                // (Why do we want to do this? Perhaps it's because we truncated in the solution)
                if (lny[ixi] < env.mu - (env.truncAt*sigmalny)) lny[ixi] = env.mu - (env.truncAt*sigmalny);
                if (lny[ixi] > env.mu + (env.truncAt*sigmalny)) lny[ixi] = env.mu + (env.truncAt*sigmalny);

                sim.y[ixi] = exp(lny[ixi]);

                // 2D interpolation
                // Assets is j (or y), Income is i (or x)
                // Signature: double interp2d(double *xgrid, double *ygrid, double *zgrid, int xlen, int ylen, double xval, double yval)
                sim.A[ixi + 1] = interp2d(&env.Ygrid[ixt*env.numPointsY + 0], &env.Agrid[ixt*env.numPointsA + 0],
                    &sln.policyA1[ixt*env.numPointsA*env.numPointsY + 0 + 0], env.numPointsY, env.numPointsA, sim.y[ixi], sim.A[ixi]);
                sim.v[ixi] = interp2d(&env.Ygrid[ixt*env.numPointsY + 0], &env.Agrid[ixt*env.numPointsA + 0],
                    &sln.value[ixt*env.numPointsA*env.numPointsY + 0 + 0], env.numPointsY, env.numPointsA, sim.y[ixi], sim.A[ixi]);
                
            }
            
            else {
            
                // Set ixY to bottom of grid
                ixY = 0;

                // Copy policyA1 to new 1D array
                policyA1 = new double[env.numPointsA];
                for (ixA = 0; ixA < env.numPointsA; ++ixA)
                    policyA1[ixA] = sln.policyA1[ixt*env.numPointsA*env.numPointsY + ixA*env.numPointsY + ixY];

                // Copy value to new 1D array
                value = new double[env.numPointsA];
                for (ixA = 0; ixA < env.numPointsA; ++ixA)
                    value[ixA] = sln.value[ixt*env.numPointsA*env.numPointsY + ixA*env.numPointsY + ixY];
            
                sim.y[ixi] = 0.0;
                // 1D interpolation
                // We do this just linearly. Should it be done on linearised values?
                sim.A[ixi + 1] = interp(&env.Agrid[ixt*env.numPointsA + 0], policyA1, env.numPointsA, sim.A[ixi]);
                sim.v[ixi] = interp(&env.Agrid[ixt*env.numPointsA + 0], value, env.numPointsA, sim.A[ixi]);

            }

            sim.c[ixi] = sim.A[ixi] + sim.y[ixi] - (sim.A[ixi + 1] / (1.0 + env.r));

            if (!((sim.A[ixi + 1] >= env.Agrid[(ixt + 1)*env.numPointsA + 0])
                && (sim.A[ixi + 1] <= env.Agrid[(ixt + 1)*env.numPointsA + env.numPointsA - 1]))) {
                cout << "Next-period assets outside the grid" << endl;
                cout << "s: " << s << endl;
                cout << "t: " << ixt << endl;
                cout << "A1: " << sim.A[ixi + 1] << endl;
                cout << "c: " << sim.c[ixi] << endl;
                cout << "Agridlo: " << env.Agrid[(ixt + 1)*env.numPointsA + 0] << endl;
                cout << "Agridhi: " << env.Agrid[(ixt + 1)*env.numPointsA + env.numPointsA - 1] << endl;
                exit(-1);
            }
            

        }

    }

}

// Print simulation values to plot
void Model::simPlot(int startSim, int endSim, char outcomeChar, char const *filename, char const *graphname) {

    double *outcome;
    char const *outcometxt;
    int s;
    int ixt;
    int ixistart, ixi;
    FILE *fp;
    char shellcmd[256];
    
    switch (outcomeChar) {
        case 'v':
            outcome = sim.v;
            outcometxt = "Value";
            break;
        case 'y':
            outcome = sim.y;
            outcometxt = "Income";
            break;
        case 'A':
            outcome = sim.A;
            outcometxt = "Assets";
            break;
        case 'c':
            outcome = sim.c;
            outcometxt = "Consumption";
            break;
    }

    fp = fopen(filename, "w");
    if (!fp) {
        printf("File %s could not be opened", filename);
        exit(-1);
    }
    
    for (s=startSim; s<=endSim; ++s) {
        ixistart = s * env.T;
        fprintf (fp, "#m=1,S=1\n");
            for (ixt=0; ixt<env.T; ++ixt) {
                ixi = ixistart + ixt;
                fprintf (fp, "%d %g\n", ixt, outcome[ixi]);
            }
    }
    
    fclose(fp);

    strcpy(shellcmd, "graph -T ps -X \"Age\" -Y \"");
    strcat(shellcmd, outcometxt);
    strcat(shellcmd, "\" -L \"");
    strcat(shellcmd, outcometxt);
    strcat(shellcmd, " across life\" -F Helvetica < ");
    strcat(shellcmd, filename);
    strcat(shellcmd, " > ");
    strcat(shellcmd, graphname);
    system(shellcmd);


}



int main() {

    Model model;
    model.init();
    
    //model.printSomeStuff();
    model.solveUsingEuler();
    model.simulate();

    model.slnPlot(0, 'v', "solution.dat", "solution.ps");
    model.simPlot(0, 10, 'A', "simulation.dat", "simulation.ps");

    model.fin();
   
}




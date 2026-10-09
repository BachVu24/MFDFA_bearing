// Fused ADMM arithmetic, mathematically identical to code/core/vmd.py.
// FFT/mirror extension remain in NumPy; no fast-math compiler option is used.
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <limits>
using C=std::complex<double>;
extern "C" __declspec(dllexport)
int vmd_admm(const double* target_data,const double* frequencies,int bins,int K,
             const double* penalties,double tau,double tol,int max_iter,int dc,
             double* mode_data,double* center_data,double* history,double* metrics) {
    const C* target=reinterpret_cast<const C*>(target_data);
    C* modes=reinterpret_cast<C*>(mode_data);
    std::vector<C> dual(bins,C(0,0)),total(bins,C(0,0));
    double target_power=0;
    for(int j=0;j<bins;++j) target_power+=std::norm(target[j]);
    for(int k=0;k<K;++k) history[k]=center_data[k];
    metrics[0]=0; metrics[1]=0; metrics[2]=(target_power==0);
    if(target_power==0) return 0;
    const double floor=std::numeric_limits<double>::epsilon()*target_power;
    for(int iteration=1;iteration<=max_iter;++iteration) {
        std::fill(total.begin(),total.end(),C(0,0));
        for(int k=0;k<K;++k) for(int j=0;j<bins;++j) total[j]+=modes[k*bins+j];
        double change=0;
        for(int k=0;k<K;++k) {
            double numerator=0,denominator=0,power_sum=0,moment_sum=0;
            const double center=center_data[k],penalty=2*penalties[k];
            C* mode=modes+k*bins;
            for(int j=0;j<bins;++j) {
                const C old=mode[j];
                total[j]-=old;
                const double delta=frequencies[j]-center;
                const C next=(target[j]-total[j]+dual[j]/2.0)/(1+penalty*delta*delta);
                total[j]+=next;
                numerator+=std::norm(next-old); denominator+=std::norm(old);
                mode[j]=next;
                const double power=std::norm(next);
                power_sum+=power; moment_sum+=frequencies[j]*power;
            }
            if(!(dc && k==0) && power_sum>0) center_data[k]=moment_sum/power_sum;
            change+=numerator/std::max(denominator,floor);
        }
        double error_power=0;
        for(int j=0;j<bins;++j) {
            const C error=target[j]-total[j];
            dual[j]+=tau*error; error_power+=std::norm(error);
        }
        const double constraint=std::sqrt(error_power/target_power);
        for(int k=0;k<K;++k) history[iteration*K+k]=center_data[k];
        metrics[0]=change; metrics[1]=constraint;
        if(change<=tol && (tau==0 || constraint<=std::sqrt(tol))) {
            metrics[2]=1; return iteration;
        }
    }
    return max_iter;
}

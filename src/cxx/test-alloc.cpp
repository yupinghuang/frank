
#include "idg-common.h"

#include <casacore/casa/Arrays/Array.h>
#include "common/Proxy.h"
#include "CUDA/Generic/Generic.h"

#include <complex>
#include <assert.h>

using std::complex; using casacore::IPosition;

int main (int argc, char *argv[]) {
    int nr_baselines= 5;
    int nr_timesteps = 1;
    int nr_channels = 1;

    std::clog << "Testing with 1 timestep and 1 channel. The ordering of channel and timesteps \
    are reversed in casacore and in idg so that's a caveat"
              << std::endl;
    std::clog << "Real part is the index, imaginary is the corr {0, 1, 2, 3} -> {XX, XY, YX, YY}" << std::endl;
    std::clog << "Make a casacore::Array and iterate through it:" << std::endl;
    // The IPosition constructor goes like (n_dim, n_corr, n_elem)
    casacore::Array<complex<float>> cc_data(IPosition(2, 4, nr_baselines));
    for (unsigned int corr=0; corr < 4; ++corr) {
        for (unsigned int bl=0; bl < nr_baselines; ++ bl) {
            cc_data(IPosition(2, corr, bl)) = complex<float>(bl, corr);
        }
    }

    // prints 0+0j 0+1j 0+2j, 0+3j, 1+0j etc...
    std::clog << "Iterator iterate: " << std::endl;
    for (auto elem : cc_data) {
        std::clog << elem.real() << "+" << elem.imag() << "j" <<std::endl;
    }

    std::clog << "Iterate with a pointer: " << std::endl;
    for (complex<float>* p = cc_data.data(); p < (cc_data.data() + cc_data.size()); ++p) {
        std::clog << p->real() << "+" << p->imag() << "j" <<std::endl;
    }

    std::clog << "Make an idg::Array3D<Matrix<2x2>>:" << std::endl;
    
    idg::proxy::cuda::Generic proxy;
    idg::Array3D<idg::Visibility<complex<float>>> visibilities =
            proxy.allocate_array3d<idg::Matrix2x2<complex<float>>>(nr_baselines, nr_timesteps, nr_channels);
    
    for (unsigned int bl=0; bl < 5; ++ bl) {
        visibilities(bl, 0, 0) = {
            complex<float>(bl, 0),
            complex<float>(bl, 1),
            complex<float>(bl, 2),
            complex<float>(bl, 3)};
    }

    std::clog << "Iterate through idg:Array3D<Matrix2x2> with a pointer" << std::endl;
    // stuck: Matrix2x2 is a struct.
    complex<float> *visibilities_begin_p = &(visibilities(0, 0, 0).xx);
    {
        std::cout << visibilities.size() << std::endl;
        auto end_p = visibilities_begin_p + visibilities.size() * 4;
        for (complex<float> *p = visibilities_begin_p; p < end_p; ++p)
        {
            std::clog << p->real() << "+" << p->imag() << "j" << std::endl;
        }
    }

    casacore::Array<complex<float>> cc_data_shared(IPosition(4, 4, nr_baselines, nr_timesteps, nr_channels),
    visibilities_begin_p,
    casacore::StorageInitPolicy::SHARE,
    std::allocator<complex<float>>());

    std::clog << "Iterator iterate for casacore Array sharing the IDG allocated memory: " << std::endl;
    for (auto elem : cc_data_shared) {
        std::clog << elem.real() << "+" << elem.imag() << "j" <<std::endl;
    }
    assert(visibilities_begin_p == cc_data_shared.data());
    /**
    step 3 is to read from a measurement set and make sure that
    data are in the right order.
    **/
    
}
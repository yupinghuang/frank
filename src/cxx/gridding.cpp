#include "CUDA/Generic/Generic.h"
#include "common/KernelTypes.h"
#include "common/Proxy.h"
#include "idg-util.h" // Don't need if not using the test Data class
#include "idg-common.h"

#include <casacore/casa/Arrays/Array.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/tables/Tables/ScalarColumn.h>
#include <casacore/ms/MeasurementSets/MeasurementSet.h>

#include <iostream>
#include <iomanip>
#include <memory>
#include <cstdlib>  // size_t
#include <complex>
#include <limits>
#include <stdexcept>
#include <chrono>

#include <omp.h>

using std::complex; using std::string;

void read_from_ms(idg::UVW<float>& uvw, idg::Array1D<float>& freqs,
                  idg::Array1D<std::pair<unsigned int, unsigned  int>>& baselines,
                  idg::Array3D<idg::Visibility<complex<float>>>& vis, const string& ms_path) {
    //construct valueholder from the allocated arrays

    // vis_arr_casa = casacore::Array<complex>(shape, vis_ptr, casacore::StorageInitPolicy::SHARE)
    //
    // oh wait I need to reorder the data.
    1 + 1;
}


int main (int argc, char *argv[]) {
    string ms_path = "/fastpool/data/20210226M-2GHz-1chan-600int.ms";
    casacore::MeasurementSet ms(ms_path);
    casacore::ROArrayColumn<casacore::Complex> data_column(
            ms, casacore::MS::columnName(casacore::MSMainEnums::DATA));
    casacore::ROArrayColumn<double> uvw_column(
            ms, casacore::MS::columnName(casacore::MSMainEnums::UVW));
    casacore::ROScalarColumn<int> ant1(
            ms, casacore::MS::columnName(casacore::MSMainEnums::ANTENNA1));
    casacore::ROScalarColumn<int> ant2(
            ms, casacore::MS::columnName(casacore::MSMainEnums::ANTENNA2));

    const unsigned int nr_correlations = 4;
    const unsigned int nr_rows = data_column.nrow();
    const unsigned int nr_stations = ms.antenna().nrow();
    const unsigned int nr_baselines = (nr_stations * (nr_stations - 1)) / 2;
    if (nr_rows % nr_baselines != 0 )
        throw std::length_error("Data column should be divisible by nr_correlations.");
    const unsigned int nr_timesteps = nr_rows / nr_baselines;

    std::clog << "nr_baselines = " << nr_baselines << std::endl;
    std::clog << "nr_rows = " << nr_rows << std::endl;
    std::clog << "nr_stations = " << nr_stations << std::endl;
    std::clog << "nr_timesteps = " << nr_timesteps << std::endl;

    const float integration_time = 1.5f;

    casacore::ROScalarColumn<int> numChanCol(ms.spectralWindow(), casacore::MSSpectralWindow::columnName(
                      casacore::MSSpectralWindowEnums::NUM_CHAN));
    casacore::ROArrayColumn<double> freqs(ms.spectralWindow(), casacore::MSSpectralWindow::columnName(
                    casacore::MSSpectralWindowEnums::CHAN_FREQ));
    int nr_channels;
    numChanCol.get(0, nr_channels);
    std::clog << "nr_channels = " << nr_channels << std::endl;
    std::clog << "freqs.nrow() = " << freqs.nrow() << std::endl;
    std::clog << "freqs row 0 shape = " << freqs.shape(0) << std::endl;
    return 0;
    const unsigned int nr_timeslots = 1; // timeslot for a-term
    const unsigned int subgrid_size = 32;
    const unsigned int kernel_size = 13;


    // hard-coding for now
    unsigned int grid_size = 8192;
    float image_size = 0.083367;
    float cell_size = image_size / grid_size;
    std::clog << "grid_size = " << grid_size << std::endl;
    std::clog << "image_size = " << image_size << std::endl;
    std::clog << "cell_size = " << cell_size << std::endl;

    std::clog << ">>> Initialize IDG data structures" << std::endl;
    idg::proxy::cuda::Generic proxy;

    // Allocate memory with Proxy
    idg::Array1D<float> frequencies = proxy.allocate_array1d<float>(nr_channels);
    idg::Array2D<idg::UVW<float>> uvw =
            proxy.allocate_array2d<idg::UVW<float>>(nr_baselines, nr_timesteps);
    idg::Array3D<idg::Visibility<std::complex<float>>> visibilities =
            proxy.allocate_array3d<idg::Visibility<std::complex<float>>>(nr_baselines, nr_timesteps, nr_correlations);

    idg::Array1D<std::pair<unsigned int, unsigned int>> baselines =
            idg::get_example_baselines(proxy, nr_stations, nr_baselines);
    idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms =
            idg::get_example_aterms(proxy, nr_timeslots, nr_stations, subgrid_size,
                                    subgrid_size);
    idg::Array1D<unsigned int> aterms_offsets =
            idg::get_example_aterms_offsets(proxy, nr_timeslots, nr_timesteps);
    idg::Array2D<float> spheroidal =
            idg::get_example_spheroidal(proxy, subgrid_size, subgrid_size);
    idg::Array1D<float> shift = idg::get_zero_shift();
    auto grid = proxy.allocate_grid(1, nr_correlations, grid_size, grid_size);
    proxy.set_grid(grid);

    std::chrono::_V2::system_clock::time_point start = std::chrono::high_resolution_clock::now();
    std::chrono::_V2::system_clock::time_point stop;
    std::chrono::seconds duration;

    std::clog << "Reading measurement set." << std::endl;

    /**
     * TODO store data on disk in different order.
     * I tried a few simple things: reading all baselines for given timestep then reorder;
     * reading one row at a time then reorder.
     * casacore doesn't seem to like it within omp parallel.
    **/

    const casacore::Array<std::complex<float>> rows = data_column.getColumn();
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    start = std::chrono::high_resolution_clock::now();
    std::clog << "Done reading measurement set in " << duration.count() << "s" << std::endl;

    // TODO use rows::const_iterator?
    #pragma omp parallel for default(none) shared(visibilities, nr_channels)
    for (unsigned int bl=0; bl < nr_baselines; ++bl) {
        for (unsigned int t=0; t < nr_timesteps; ++t) {
            for (unsigned int chan=0; chan < nr_channels; ++chan) {
                unsigned row_nr = bl + t * nr_baselines;
                casacore::IPosition xx(3, chan, 0, row_nr);
                casacore::IPosition xy(3, chan, 1, row_nr);
                casacore::IPosition yx(3, chan, 2, row_nr);
                casacore::IPosition yy(3, chan, 3, row_nr);
                idg::Matrix2x2<std::complex<float>> vis = {rows(xx), rows(xy), rows(yx), rows(yy)};
                visibilities(bl, t, chan) = vis;
            }
        }
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::clog << "Reordered visibilities in " << duration.count() << "s" << std::endl;

    // not using w_step
    proxy.init_cache(subgrid_size, cell_size, 0.0, shift);

    // Create plan
    std::clog << ">>> Create plan" << std::endl;
    idg::Plan::Options options;
    options.plan_strict = true;
    const std::unique_ptr<idg::Plan> plan = proxy.make_plan(
            kernel_size, frequencies, uvw, baselines, aterms_offsets, options);
    std::clog << std::endl;

    std::clog << ">>> Run gridding" << std::endl;
    proxy.gridding(*plan, frequencies, visibilities, uvw, baselines, aterms,
                   aterms_offsets, spheroidal);
    proxy.get_final_grid();
}

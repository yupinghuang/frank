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
#include <assert.h>
#include <vector>

#include <omp.h>

#define SPEED_OF_LIGHT 299792458.0

using std::complex; using std::string; using casacore::IPosition;


void get_metadata(const string &ms_path, float &integration_time,
                  unsigned int &nr_rows, unsigned int &nr_stations,
                  unsigned int &nr_baselines, unsigned int &nr_timesteps,
                  unsigned int &nr_channels, unsigned int &nr_correlations) {
    casacore::MeasurementSet ms(ms_path);
    casacore::ROArrayColumn<casacore::Complex> data_column(
            ms, casacore::MS::columnName(casacore::MSMainEnums::DATA));
    nr_correlations = 4;
    nr_rows = data_column.nrow();
    nr_stations = ms.antenna().nrow();
    nr_baselines = (nr_stations * (nr_stations - 1)) / 2;
    // assume there's no autocorrelation in the data
    assert(nr_rows % nr_baselines == 0);
    nr_timesteps = nr_rows / nr_baselines;

    casacore::ROScalarColumn<double> exposureCol(ms, casacore::MS::columnName(casacore::MSMainEnums::EXPOSURE));
    integration_time = static_cast<float>(exposureCol.get(0));

    casacore::ROScalarColumn<int> numChanCol(ms.spectralWindow(), casacore::MSSpectralWindow::columnName(
                      casacore::MSSpectralWindowEnums::NUM_CHAN));
    nr_channels = static_cast<unsigned int>(numChanCol.get(0));
    std::clog << "integration_time = " << integration_time << std::endl;
    std::clog << "nr_rows = " << nr_rows << std::endl;
    std::clog << "nr_stations = " << nr_stations << std::endl;
    std::clog << "nr_baselines = " << nr_baselines << std::endl;
    std::clog << "nr_timesteps = " << nr_timesteps << std::endl;
    std::clog << "nr_channels = " << nr_channels << std::endl;
    std::clog << "nr_correlations = " << nr_correlations << std::endl;
}

void get_data(const string &ms_path, const unsigned int nr_channels,
              const unsigned int nr_baselines, const unsigned int nr_timesteps,
              idg::Array2D<idg::UVW<float>> &uvw, idg::Array1D<float> &frequencies,
              idg::Array1D<std::pair<unsigned int, unsigned int>> &baselines,
              idg::Array3D<idg::Visibility<complex<float>>> &visibilities)
{
    casacore::MeasurementSet ms(ms_path);
    casacore::ROArrayColumn<casacore::Complex> data_column(
            ms, casacore::MS::columnName(casacore::MSMainEnums::DATA));
    casacore::ROArrayColumn<double> uvw_column(
            ms, casacore::MS::columnName(casacore::MSMainEnums::UVW));
    casacore::ROScalarColumn<int> ant1(
            ms, casacore::MS::columnName(casacore::MSMainEnums::ANTENNA1));
    casacore::ROScalarColumn<int> ant2(
            ms, casacore::MS::columnName(casacore::MSMainEnums::ANTENNA2));
    casacore::ROArrayColumn<double> freqs(ms.spectralWindow(), casacore::MSSpectralWindow::columnName(
                    casacore::MSSpectralWindowEnums::CHAN_FREQ));

    std::clog << "Reading baseline pairs and frequencies data from the measurement set." << std::endl;
    casacore::Array<double> src_freqs = freqs(0);
    assert(src_freqs.size() == nr_channels);
    for (unsigned int i = 0; i < nr_channels; ++i)
        frequencies(i) = float(src_freqs(IPosition(1, i)));
    std::clog << "done with reading frequencies." << std::endl;

    casacore::Slicer first_int_rows(IPosition(1, 0), IPosition(1, nr_baselines)); 
    casacore::Vector<int> ant1_vec = ant1.getColumnRange(first_int_rows); 
    casacore::Vector<int> ant2_vec = ant2.getColumnRange(first_int_rows); 
    #pragma omp parallel for default(none) shared(baselines, ant1_vec, ant2_vec)
    for (unsigned int i = 0; i < nr_baselines; ++i) {
        std::pair<unsigned int, unsigned int> curr_pair = {ant1_vec(i), ant2_vec(i)};
        baselines(i) = curr_pair;
    }

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

    const casacore::Array<complex<float>> data_rows = data_column.getColumn();
    const casacore::Array<double> uvw_rows = uvw_column.getColumn();
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::clog << "Done reading measurement set in " << duration.count() << "s" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    // TODO use one of the specialization of Array to iterate.
    #pragma omp parallel for default(none) shared(visibilities, uvw_column, uvw)
    for (unsigned int t=0; t < nr_timesteps; ++t) {
        for (unsigned int bl=0; bl < nr_baselines; ++bl) {
            unsigned int row_i = bl + t * nr_baselines;

            idg::UVW<float> idg_uvw = {
                float(uvw_rows(IPosition(2, 0, row_i))),
                float(uvw_rows(IPosition(2, 1, row_i))),
                float(uvw_rows(IPosition(2, 2, row_i)))};
            uvw(bl, t) = idg_uvw;
            
            for (unsigned int chan=0; chan < nr_channels; ++chan) {
                idg::Matrix2x2<complex<float>> vis = {
                        data_rows(IPosition(3, 0, chan, row_i)),
                        data_rows(IPosition(3, 1, chan, row_i)),
                        data_rows(IPosition(3, 2, chan, row_i)),
                        data_rows(IPosition(3, 3, chan, row_i))};
                visibilities(bl, t, chan) = vis;
            }
        }
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::clog << "Reordered visibilities in " << duration.count() << "s" << std::endl;
}

float compute_image_size(unsigned long grid_size, float end_frequency) {
    const float grid_padding = 1.20;
    grid_size /= (2 * grid_padding);
    float max_uv = 15392.2;
    return grid_size / max_uv * (SPEED_OF_LIGHT / end_frequency);
}
int main (int argc, char *argv[]) {
    string ms_path = "/fastpool/data/W-10int-200chan.ms";

    // TODO: a struct?
    unsigned int nr_correlations;
    unsigned int nr_rows;
    unsigned int nr_stations;
    unsigned int nr_baselines;
    float integration_time;
    unsigned int nr_timesteps;
    unsigned int nr_channels;
    get_metadata(ms_path, integration_time, nr_rows, nr_stations,
                 nr_baselines, nr_timesteps, nr_channels, nr_correlations);
    
    std::clog << ">>> Initialize IDG proxy." << std::endl;
    idg::proxy::cuda::Generic proxy;

    std::clog << ">>> Allocating metadata arrays" << std::endl;
    idg::Array1D<float> frequencies = proxy.allocate_array1d<float>(nr_channels);
    idg::Array1D<std::pair<unsigned int, unsigned int>> baselines =
        proxy.allocate_array1d <std::pair<unsigned int, unsigned int>>(nr_baselines);
    idg::Array2D<idg::UVW<float>> uvw =
        proxy.allocate_array2d<idg::UVW<float>>(nr_baselines, nr_timesteps);
    std::clog << ">>> Allocating vis" << std::endl;
    idg::Array3D<idg::Visibility<complex<float>>> visibilities =
            proxy.allocate_array3d<idg::Visibility<complex<float>>>(nr_baselines, nr_timesteps, nr_channels);

    std::clog << ">>> Reading data" << std::endl;
    get_data(ms_path, nr_channels, nr_baselines, nr_timesteps, uvw, frequencies, baselines, visibilities);

    float end_frequency = frequencies(frequencies.size() - 1);
    std::clog << "end frequency = " << end_frequency << std::endl;
    // hard-coding for now
    unsigned int grid_size = 8192;
    float image_size = compute_image_size(grid_size, end_frequency);
    float cell_size = image_size / grid_size;
    std::clog << "grid_size = " << grid_size << std::endl;
    std::clog << "image_size = " << image_size << std::endl;
    std::clog << "cell_size = " << cell_size << std::endl;

    // A-terms
    const unsigned int nr_timeslots = 1; // timeslot for a-term
    const unsigned int subgrid_size = 32;
    const unsigned int kernel_size = 13;
    idg::Array4D<idg::Matrix2x2<complex<float>>> aterms =
        idg::get_example_aterms(proxy, nr_timeslots, nr_stations, subgrid_size,
                                subgrid_size);
    idg::Array1D<unsigned int> aterms_offsets =
            idg::get_example_aterms_offsets(proxy, nr_timeslots, nr_timesteps);

    idg::Array2D<float> spread =
            idg::get_example_spheroidal(proxy, subgrid_size, subgrid_size);
    idg::Array1D<float> shift = idg::get_zero_shift();

    std::shared_ptr<idg::Grid> grid = proxy.allocate_grid(1, nr_correlations, grid_size, grid_size);
    proxy.set_grid(grid);
    // no w-tiling, i.e. not using w_step

    proxy.init_cache(subgrid_size, cell_size, 0.0, shift);

    // Create plan
    std::clog << ">>> Creating plan" << std::endl;
    idg::Plan::Options options;
    options.plan_strict = true;
    const std::unique_ptr<idg::Plan> plan = proxy.make_plan(
            kernel_size, frequencies, uvw, baselines, aterms_offsets, options);
    std::clog << std::endl;

    std::clog << ">>> Run gridding" << std::endl;
    proxy.gridding(*plan, frequencies, visibilities, uvw, baselines, aterms,
                   aterms_offsets, spread);
    proxy.get_final_grid();
}

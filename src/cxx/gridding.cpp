#include "CUDA/Generic/Generic.h"
#include "common/KernelTypes.h"
#include "common/Proxy.h"
#include "idg-util.h" // Don't need if not using the test Data class
#include "idg-common.h"

#include <iostream>
#include <iomanip>
#include <cstdlib>  // size_t
#include <complex>
#include <limits>
#include <complex>

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

void grid() {
    /*
     * Call this from python?
     */
}

int main (int argc, char *argv[]) {

    // parameters
    unsigned int nr_correlations = 4;
    unsigned int nr_stations = 9;
    unsigned int nr_channels = 9;
    unsigned int nr_timesteps = 2048;
    unsigned int nr_timeslots = 7;
    unsigned int grid_size = 2048;
    unsigned int subgrid_size = 32;
    unsigned int kernel_size = 9;
    unsigned int nr_baselines = (nr_stations * (nr_stations - 1)) / 2;
    float integration_time = 1.0f;

    idg::Data data = idg::get_example_data(nr_baselines, grid_size,
                                           integration_time, nr_channels);

    float image_size = data.compute_image_size(grid_size, nr_channels);
    float cell_size = image_size / grid_size;

    std::clog << ">>> Initialize data structures" << std::endl;
    idg::proxy::cuda::Generic proxy;

    // Try to allocate aligned memory
    idg::Array1D<float> frequencies = proxy.allocate_array1d<float>(nr_channels);
    data.get_frequencies(frequencies, image_size);

    idg::Array2D<idg::UVW<float>> uvw =
            proxy.allocate_array2d<idg::UVW<float>>(nr_baselines, nr_timesteps);
    data.get_uvw(uvw);

    idg::Array3D<idg::Visibility<std::complex<float>>> visibilities =
            idg::get_dummy_visibilities(proxy, nr_baselines, nr_timesteps,
                                        nr_channels);

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
    std::clog << std::endl;

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
    // result is at grid->data()
    // TODO: remove taper
}

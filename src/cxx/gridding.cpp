#include <assert.h>
#include <casacore/casa/Arrays/Array.h>
#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/tables/Tables/ScalarColumn.h>
#include <omp.h>

#include <chrono>
#include <complex>
#include <cstdlib>  // size_t
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "CUDA/Generic/Generic.h"
#include "common/KernelTypes.h"
#include "common/Proxy.h"
#include "idg-common.h"
#include "idg-util.h"  // Don't need if not using the test Data class
#include "npy.hpp"

const float SPEED_OF_LIGHT = 299792458.0;
const float MAX_BL_M = 15392.2;

using casacore::IPosition;
using std::complex;
using std::string;

void getMetadata(const string &ms_path, float &integration_time,
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

  casacore::ROScalarColumn<double> exposure_col(
      ms, casacore::MS::columnName(casacore::MSMainEnums::EXPOSURE));
  integration_time = static_cast<float>(exposure_col.get(0));

  casacore::ROScalarColumn<int> num_chan_col(
      ms.spectralWindow(), casacore::MSSpectralWindow::columnName(
                               casacore::MSSpectralWindowEnums::NUM_CHAN));
  nr_channels = static_cast<unsigned int>(num_chan_col.get(0));
  std::clog << "integration_time = " << integration_time << std::endl;
  std::clog << "nr_rows = " << nr_rows << std::endl;
  std::clog << "nr_stations = " << nr_stations << std::endl;
  std::clog << "nr_baselines = " << nr_baselines << std::endl;
  std::clog << "nr_timesteps = " << nr_timesteps << std::endl;
  std::clog << "nr_channels = " << nr_channels << std::endl;
  std::clog << "nr_correlations = " << nr_correlations << std::endl;
}

void reorderData(const unsigned int nr_timesteps,
                 const unsigned int nr_baselines,
                 const unsigned int nr_channels,
                 const casacore::Array<double> &uvw_rows,
                 const casacore::Array<complex<float>> &data_rows,
                 idg::Array2D<idg::UVW<float>> &uvw,
                 idg::Array4D<complex<float>> &visibilities) {
// TODO use one of the specialization of Array to iterate.
#pragma omp parallel for default(none) shared(visibilities, uvw, uvw_rows, data_rows)
  for (unsigned int t = 0; t < nr_timesteps; ++t) {
    for (unsigned int bl = 0; bl < nr_baselines; ++bl) {
      unsigned int row_i = bl + t * nr_baselines;

      idg::UVW<float> idg_uvw = {float(uvw_rows(IPosition(2, 0, row_i))),
                                 float(uvw_rows(IPosition(2, 1, row_i))),
                                 float(uvw_rows(IPosition(2, 2, row_i)))};
      uvw(bl, t) = idg_uvw;

      for (unsigned int chan = 0; chan < nr_channels; ++chan) {
        visibilities(bl, t, chan, 0) = data_rows(IPosition(3, 0, chan, row_i));
        visibilities(bl, t, chan, 1) = data_rows(IPosition(3, 1, chan, row_i));
        visibilities(bl, t, chan, 2) = data_rows(IPosition(3, 2, chan, row_i));
        visibilities(bl, t, chan, 3) = data_rows(IPosition(3, 3, chan, row_i));
      }
    }
  }
}

void getData(const string &ms_path, const unsigned int nr_channels,
             const unsigned int nr_baselines, const unsigned int nr_timesteps,
             idg::Array2D<idg::UVW<float>> &uvw,
             idg::Array1D<float> &frequencies,
             idg::Array1D<std::pair<unsigned int, unsigned int>> &baselines,
             idg::Array4D<complex<float>> &visibilities) {
  casacore::MeasurementSet ms(ms_path);
  casacore::ROArrayColumn<casacore::Complex> data_column(
      ms, casacore::MS::columnName(casacore::MSMainEnums::DATA));
  casacore::ROArrayColumn<double> uvw_column(
      ms, casacore::MS::columnName(casacore::MSMainEnums::UVW));
  casacore::ROScalarColumn<int> ant1(
      ms, casacore::MS::columnName(casacore::MSMainEnums::ANTENNA1));
  casacore::ROScalarColumn<int> ant2(
      ms, casacore::MS::columnName(casacore::MSMainEnums::ANTENNA2));
  casacore::ROArrayColumn<double> freqs(
      ms.spectralWindow(), casacore::MSSpectralWindow::columnName(
                               casacore::MSSpectralWindowEnums::CHAN_FREQ));

  std::clog
      << "Reading baseline pairs and frequencies data from the measurement set."
      << std::endl;
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
    std::pair<unsigned int, unsigned int> curr_pair = {ant1_vec(i),
                                                       ant2_vec(i)};
    baselines(i) = curr_pair;
  }

  std::chrono::_V2::system_clock::time_point start =
      std::chrono::high_resolution_clock::now();
  std::chrono::_V2::system_clock::time_point stop;
  std::chrono::seconds duration;

  std::clog << "Reading measurement set." << std::endl;

  /**
   * TODO store data on disk in different order.
   * I tried a few simple things: reading all baselines for given timestep then
   *reorder; reading one row at a time then reorder. casacore doesn't seem to
   *like it within omp parallel.
   **/

  const casacore::Array<complex<float>> data_rows = data_column.getColumn();
  const casacore::Array<double> uvw_rows = uvw_column.getColumn();
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
  std::clog << "Done reading measurement set in " << duration.count() << "s"
            << std::endl;
  start = std::chrono::high_resolution_clock::now();
  reorderData(nr_timesteps, nr_baselines, nr_channels, uvw_rows, data_rows, uvw,
              visibilities);
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
  std::clog << "Reordered visibilities in " << duration.count() << "s"
            << std::endl;
}

float computeImageSize(unsigned long grid_size, float end_frequency) {
  const float grid_padding = 1.20;
  grid_size /= (2 * grid_padding);
  return grid_size / MAX_BL_M * (SPEED_OF_LIGHT / end_frequency);
}

int main(int argc, char *argv[]) {
  string ms_path = "/fastpool/data/W-snapshot-64chan-5deg.ms";

  // TODO: a struct?
  unsigned int nr_correlations;
  unsigned int nr_rows;
  unsigned int nr_stations;
  unsigned int nr_baselines;
  float integration_time;
  unsigned int nr_timesteps;
  unsigned int nr_channels;
  getMetadata(ms_path, integration_time, nr_rows, nr_stations, nr_baselines,
              nr_timesteps, nr_channels, nr_correlations);

  std::clog << ">>> Initialize IDG proxy." << std::endl;
  idg::proxy::cuda::Generic proxy;

  std::clog << ">>> Allocating metadata arrays" << std::endl;
  idg::Array1D<float> frequencies = proxy.allocate_array1d<float>(nr_channels);
  idg::Array1D<std::pair<unsigned int, unsigned int>> baselines =
      proxy.allocate_array1d<std::pair<unsigned int, unsigned int>>(
          nr_baselines);
  idg::Array2D<idg::UVW<float>> uvw =
      proxy.allocate_array2d<idg::UVW<float>>(nr_baselines, nr_timesteps);
  std::clog << ">>> Allocating vis" << std::endl;
  auto visibilities = proxy.allocate_array4d<complex<float>>(
      nr_baselines, nr_timesteps, nr_channels, nr_correlations);

  std::clog << ">>> Reading data" << std::endl;
  getData(ms_path, nr_channels, nr_baselines, nr_timesteps, uvw, frequencies,
          baselines, visibilities);
  float end_frequency = frequencies(frequencies.size() - 1);
  std::clog << "end frequency = " << end_frequency << std::endl;

  unsigned int grid_size = 8192;
  float image_size = computeImageSize(grid_size, end_frequency);
  float cell_size = image_size / grid_size;
  std::clog << "grid_size = " << grid_size << std::endl;
  std::clog << "image_size = " << image_size << " (radians?)" << std::endl;
  std::clog << "pixel_size (idg's cell_size) = " << cell_size << " (radians?)" << std::endl;

  // A-terms
  const unsigned int nr_timeslots = 1;  // timeslot for a-term
  const unsigned int subgrid_size = 32;
  const unsigned int kernel_size = 13;
  idg::Array4D<idg::Matrix2x2<complex<float>>> aterms = idg::get_example_aterms(
      proxy, nr_timeslots, nr_stations, subgrid_size, subgrid_size);
  idg::Array1D<unsigned int> aterms_offsets =
      idg::get_example_aterms_offsets(proxy, nr_timeslots, nr_timesteps);

  idg::Array2D<float> spread =
      idg::get_example_spheroidal(proxy, subgrid_size, subgrid_size);
  idg::Array1D<float> shift = idg::get_zero_shift();

  std::shared_ptr<idg::Grid> grid =
      proxy.allocate_grid(1, nr_correlations, grid_size, grid_size);
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
  std::clog << "Run FFT" << std::endl;
  proxy.transform(idg::FourierDomainToImageDomain);
  auto image_corr = proxy.get_final_grid();
  auto image_iquv = proxy.allocate_array3d<double>(4, grid_size, grid_size);
  for (unsigned int i = 0; i < grid_size; ++i) {
    for (unsigned int j = 0; j < grid_size; ++j) {
      image_iquv(0, i, j) =
          static_cast<double>(0.5 * ((*image_corr)(0, 0, i, j).real() +
                                     (*image_corr)(0, 3, i, j).real()));
      image_iquv(1, i, j) =
          static_cast<double>(0.5 * ((*image_corr)(0, 0, i, j).real() -
                                     (*image_corr)(0, 3, i, j).real()));
      image_iquv(2, i, j) =
          static_cast<double>(0.5 * ((*image_corr)(0, 1, i, j).real() -
                                     (*image_corr)(0, 2, i, j).real()));
      image_iquv(3, i, j) =
          static_cast<double>(0.5 * (-(*image_corr)(0, 1, i, j).imag() +
                                     (*image_corr)(0, 2, i, j).imag()));
    }
  }
  const long unsigned imshape[] = {4, grid_size, grid_size};
  npy::SaveArrayAsNumpy(
      "image.npy", false, 3, imshape,
      std::vector<double>(image_iquv.data(),
                          image_iquv.data() + image_iquv.size()));
}

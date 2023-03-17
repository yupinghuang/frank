import argparse

from casatasks import deconvolve, importfits, exportfits

def casa_deconvolve(input_fits, output_fits, psf_fits):
    prefix = input_fits.rstrip('.fits')
    importfits(fitsimage=input_fits, imagename=prefix + '.residual', overwrite=True)
    importfits(fitsimage=psf_fits, imagename=prefix + '.psf', overwrite=True)
    deconvolve(imagename=prefix, deconvolver="multiscale",
        niter=5000000, gain=0.2,smallscalebias=0.5, interactive=True, scales=[0,3,9,27])
    exportfits(imagename=prefix + '.image', fitsimage=output_fits, overwrite=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deconvolve a dirty image')
    parser.add_argument('input_fits', help='Input image')
    parser.add_argument('output_fits', help='Output image')
    parser.add_argument('psf_fits', help='PSF image')
    args = parser.parse_args()
    casa_deconvolve(args.input_fits, args.output_fits, args.psf_fits)

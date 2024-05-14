# Pre-trained Networks

The following pre-trained networks are provided with PlantSeg package out-of-the box and can be specified in the config file or chosen in the GUI.

* `generic_confocal_3D_unet` - alias for `confocal_3D_unet_ovules_ds2x` see below
* `generic_light_sheet_3D_unet` - alias for `lightsheet_3D_unet_root_ds1x` see below
* `confocal_3D_unet_ovules_ds1x` - a variant of 3D U-Net trained on confocal images of *Arabidopsis* ovules on original resolution, voxel size: (0.235x0.075x0.075 µm^3) (ZYX) with BCEDiceLoss
* `confocal_3D_unet_ovules_ds2x` - a variant of 3D U-Net trained on confocal images of *Arabidopsis* ovules on 1/2 resolution, voxel size: (0.235x0.150x0.150 µm^3) (ZYX) with BCEDiceLoss
* `confocal_3D_unet_ovules_ds3x` - a variant of 3D U-Net trained on confocal images of *Arabidopsis* ovules on 1/3 resolution, voxel size: (0.235x0.225x0.225 µm^3) (ZYX) with BCEDiceLoss
* `confocal_2D_unet_ovules_ds2x` - a variant of 2D U-Net trained on confocal images of *Arabidopsis* ovules. Training the 2D U-Net is done on the Z-slices (1/2 resolution, pixel size: 0.150x0.150 µm^3) with BCEDiceLoss
* `confocal_3D_unet_ovules_nuclei_ds1x` - a variant of 3D U-Net trained on confocal images of *Arabidopsis* ovules nuclei stain on original resolution, voxel size: (0.35x0.1x0.1 µm^3) (ZYX) with BCEDiceLoss
* `lightsheet_3D_unet_root_ds1x` - a variant of 3D U-Net trained on light-sheet images of *Arabidopsis* lateral root on original resolution, voxel size: (0.25x0.1625x0.1625 µm^3) (ZYX) with BCEDiceLoss
* `lightsheet_3D_unet_root_ds2x` - a variant of 3D U-Net trained on light-sheet images of *Arabidopsis* lateral root on 1/2 resolution, voxel size: (0.25x0.325x0.325 µm^3) (ZYX) with BCEDiceLoss
* `lightsheet_3D_unet_root_ds3x` - a variant of 3D U-Net trained on light-sheet images of *Arabidopsis* lateral root on 1/3 resolution, voxel size: (0.25x0.4875x0.4875 µm^3) (ZYX) with BCEDiceLoss
* `lightsheet_2D_unet_root_ds1x` - a variant of 2D U-Net trained on light-sheet images of *Arabidopsis* lateral root. Training the 2D U-Net is done on the Z-slices (pixel size: 0.1625x0.1625 µm^3) with BCEDiceLoss
* `lightsheet_3D_unet_root_nuclei_ds1x` - a variant of 3D U-Net trained on light-sheet images *Arabidopsis* lateral root nuclei on original resolution, voxel size: (0.25x0.1625x0.1625 µm^3) (ZYX) with BCEDiceLoss
* `lightsheet_2D_unet_root_nuclei_ds1x` - a variant of 2D U-Net trained on light-sheet images *Arabidopsis* lateral root nuclei. Training the 2D U-Net is done on the Z-slices (pixel size: 0.1625x0.1625 µm^3) with BCEDiceLoss.
* `confocal_3D_unet_sa_meristem_cells` - a variant of 3D U-Net trained on confocal images of shoot apical meristem dataset from: Jonsson, H., Willis, L., & Refahi, Y. (2017). Research data supporting Cell size and growth regulation in the Arabidopsis thaliana apical stem cell niche. <https://doi.org/10.17863/CAM.7793>. voxel size: (0.25x0.25x0.25 µm^3) (ZYX)
* `confocal_2D_unet_sa_meristem_cells` - a variant of 2D U-Net trained on confocal images of shoot apical meristem dataset from: Jonsson, H., Willis, L., & Refahi, Y. (2017). Research data supporting Cell size and growth regulation in the Arabidopsis thaliana apical stem cell niche. <https://doi.org/10.17863/CAM.7793>.  pixel size: (25x0.25 µm^3) (YX)
* `lightsheet_3D_unet_mouse_embryo_cells` - A variant of 3D U-Net trained to predict the cell boundaries in live light-sheet images of ex-vivo developing mouse embryo. Voxel size: (0.2×0.2×1 µm^3) (XYZ)
* `confocal_3D_unet_mouse_embryo_nuclei` - A variant of 3D U-Net trained to predict the cell boundaries in live light-sheet images of ex-vivo developing mouse embryo. Voxel size: (0.2×0.2×1 µm^3) (XYZ)

Selecting a given network name (either in the config file or GUI) will download the network into the `~/.plantseg_models` directory.
Detailed description of network training can be found in our [paper](https://doi.org/10.7554/eLife.57613).

The PlantSeg home directory can be configured with the `PLANTSEG_HOME` environment variable.

```bash
export PLANTSEG_HOME="/path/to/plantseg/home"
```

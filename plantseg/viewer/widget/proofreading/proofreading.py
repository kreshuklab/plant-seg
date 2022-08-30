from functools import partial

import napari
import numpy as np
from magicgui import magicgui
from napari.layers import Labels, Image
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info

from plantseg.viewer.widget.proofreading.utils import get_bboxes, get_idx_slice
from skimage.segmentation import watershed

"""
try:
    import SimpleITK as sitk
    from plantseg.segmentation.functional.segmentation import simple_itk_watershed_from_markers as watershed

except ImportError:
    from skimage.segmentation import watershed as skimage_ws
    watershed = partial(skimage_ws, compactness=0.01)
    
"""


current_label_layer = None


def _merge_from_seeds(segmentation, region_slice, region_bbox, bboxes, all_idx, max_label):
    region_segmentation = segmentation[region_slice]

    mask = [region_segmentation == idx for idx in all_idx]

    new_label = 0 if 0 in all_idx else all_idx[0]

    mask = np.logical_or.reduce(mask)
    region_segmentation[mask] = new_label
    bboxes[new_label] = region_bbox
    show_info('merge complete')
    return region_segmentation, region_slice, {'bboxes': bboxes, 'max_label': max_label}


def _split_from_seed(segmentation, sz, sx, sy, region_slice, all_idx, offsets, bboxes, image, seeds_values, max_label):
    local_sz, local_sx, local_sy = sz - offsets[0], sx - offsets[1], sy - offsets[2]

    region_image = image[region_slice]
    region_segmentation = segmentation[region_slice]

    region_seeds = np.zeros_like(region_segmentation)

    seeds_values += max_label
    region_seeds[local_sz, local_sx, local_sy] = seeds_values

    mask = [region_segmentation == idx for idx in all_idx]
    mask = np.logical_or.reduce(mask)

    new_seg = watershed(region_image, region_seeds, mask=mask, compactness=0.001)
    new_seg[~mask] = region_segmentation[~mask]

    new_bboxes = get_bboxes(new_seg)
    for idx in np.unique(seeds_values):
        values = new_bboxes[idx]
        values = values + offsets[None, :]
        bboxes[idx] = values

    show_info('split_complete')
    return new_seg, region_slice, {'bboxes': bboxes, 'max_label': max_label}


def split_merge_from_seeds(seeds, segmentation, image, bboxes, max_label):
    # find seeds location ad label value
    sz, sx, sy = np.nonzero(seeds)

    seeds_values = seeds[sz, sx, sy]
    seeds_idx = np.unique(seeds_values)

    all_idx = segmentation[sz, sx, sy]
    all_idx = np.unique(all_idx)

    region_slice, region_bbox, offsets = get_idx_slice(all_idx, bboxes_dict=bboxes)

    if len(seeds_idx) == 1:
        return _merge_from_seeds(segmentation,
                                 region_slice,
                                 region_bbox,
                                 bboxes,
                                 all_idx,
                                 max_label)
    else:
        return _split_from_seed(segmentation,
                                sz, sx, sy,
                                region_slice,
                                all_idx,
                                offsets,
                                bboxes,
                                image,
                                seeds_values,
                                max_label)


@magicgui(call_button='Clean scribbles - < c >')
def widget_clean_scribble(viewer):
    if current_label_layer is None:
        show_info('Scribble not defined. Run the proofreading widget tool once first')
        return None

    viewer.layers[current_label_layer].data = np.zeros_like(viewer.layers[current_label_layer].data)
    viewer.layers[current_label_layer].refresh()


@magicgui(call_button='Split/Merge from scribbles - < p >',
          scribbles={'label': 'Scribbles'},
          segmentation={'label': 'Segmentation'},
          image={'label': 'Image'})
def widget_split_and_merge_from_scribbles(viewer: napari.Viewer,
                                          scribbles: Labels,
                                          segmentation: Labels,
                                          image: Image) -> None:

    if scribbles.name == segmentation.name:
        show_info('Scribbles layer and segmentation layer cannot be the same')
        return None

    if 'bboxes' in segmentation.metadata.keys():
        bboxes = segmentation.metadata['bboxes']
    else:
        bboxes = get_bboxes(segmentation.data)

    if 'max_label' in segmentation.metadata.keys():
        max_label = segmentation.metadata['max_label']
    else:
        max_label = np.max(segmentation.data)

    global current_label_layer
    current_label_layer = scribbles.name

    @thread_worker
    def func():
        new_seg, region_slice, meta = split_merge_from_seeds(scribbles.data,
                                                             segmentation.data,
                                                             image=image.data,
                                                             bboxes=bboxes,
                                                             max_label=max_label)
        viewer.layers[segmentation.name].data[region_slice] = new_seg
        viewer.layers[segmentation.name].metadata = meta
        viewer.layers[segmentation.name].refresh()

    worker = func()
    worker.start()

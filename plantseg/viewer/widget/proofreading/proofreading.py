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

current_label_layer: str = '__undefined__'
default_key_binding_split_merge = 'n'
default_key_binding_clean = 'b'


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

    show_info('split complete')
    return new_seg, region_slice, {'bboxes': bboxes, 'max_label': max_label}


def split_merge_from_seeds(seeds, segmentation, image, bboxes, max_label, correct_labels):
    # find seeds location ad label value
    sz, sx, sy = np.nonzero(seeds)

    seeds_values = seeds[sz, sx, sy]
    seeds_idx = np.unique(seeds_values)

    all_idx = segmentation[sz, sx, sy]
    all_idx = np.unique(all_idx)

    region_slice, region_bbox, offsets = get_idx_slice(all_idx, bboxes_dict=bboxes)

    for idx in all_idx:
        if idx in correct_labels:
            return segmentation, region_slice, {'bboxes': bboxes, 'max_label': max_label}

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


@magicgui(call_button=f'Clean scribbles - < {default_key_binding_clean} >')
def widget_clean_scribble(viewer: napari.Viewer):
    if 'scribbles' not in viewer.layers:
        show_info('Scribble Layer not defined. Run the proofreading widget tool once first')
        return None

    viewer.layers['scribbles'].data = np.zeros_like(viewer.layers['scribbles'].data)
    viewer.layers['scribbles'].refresh()


def widget_add_label_to_corrected(viewer: napari.Viewer):
    if 'corrected' not in viewer.layers:
        show_info('Corrected Layer not defined. Run the proofreading widget tool once first')
        return None


def initialize_proofreading(viewer: napari.Viewer, shapes, layers_kwargs) -> bool:
    init_was_required = False
    if 'scribbles' not in viewer.layers:
        init_was_required = True
        viewer.add_labels(np.zeros(shapes, dtype=np.uint16), name='scribbles', **layers_kwargs)

    if 'corrected' not in viewer.layers:
        init_was_required = True
        viewer.add_labels(np.zeros(shapes, dtype=np.uint16),
                          name='corrected',
                          **layers_kwargs,
                          metadata={'correct_labels': set()})

    return init_was_required


@magicgui(call_button=f'Split/Merge from scribbles - < {default_key_binding_split_merge} >',
          segmentation={'label': 'Segmentation'},
          image={'label': 'Image'})
def widget_split_and_merge_from_scribbles(viewer: napari.Viewer,
                                          segmentation: Labels,
                                          image: Image) -> None:

    if initialize_proofreading(viewer, segmentation.data.shape, {'scale': segmentation.scale}):
        show_info('Proofreading widget initialized')
        return None

    if 'bboxes' in segmentation.metadata.keys():
        bboxes = segmentation.metadata['bboxes']
    else:
        bboxes = get_bboxes(segmentation.data)

    if 'max_label' in segmentation.metadata.keys():
        max_label = segmentation.metadata['max_label']
    else:
        max_label = np.max(segmentation.data)

    scribbles = viewer.layers['scribbles'].data
    correct_labels = viewer.layers['corrected'].metadata['correct_labels']

    @thread_worker
    def func():
        new_seg, region_slice, meta = split_merge_from_seeds(scribbles,
                                                             segmentation.data,
                                                             image=image.data,
                                                             bboxes=bboxes,
                                                             max_label=max_label,
                                                             correct_labels=correct_labels)
        print(np.allclose(new_seg, segmentation.data[region_slice]))
        print(np.sum(scribbles))
        viewer.layers[segmentation.name].data[region_slice] = new_seg
        viewer.layers[segmentation.name].metadata = meta
        viewer.layers[segmentation.name].refresh()

    worker = func()
    worker.start()

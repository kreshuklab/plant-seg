from plantseg.pipeline.steps import AbstractSegmentationStep
from plantseg.segmentation.functional.segmentation import simple_itk_watershed


class SimpleITKWatershed(AbstractSegmentationStep):
    def __init__(
        self,
        predictions_paths,
        key=None,
        channel=None,
        save_directory="ITKWatershed",
        ws_2D=True,
        ws_threshold=0.4,
        ws_minsize=50,
        ws_sigma=0.3,
        n_threads=8,
        state=True,
        **kwargs,
    ):
        super().__init__(
            input_paths=predictions_paths,
            save_directory=save_directory,
            file_suffix='_itkws',
            state=state,
            input_key=key,
            input_channel=channel,
        )

        self.ws_threshold = ws_threshold
        self.ws_minsize = ws_minsize
        self.ws_sigma = ws_sigma

    def process(self, pmaps):
        segmentation = simple_itk_watershed(
            pmaps, threshold=self.ws_threshold, sigma=self.ws_sigma, minsize=self.ws_minsize
        )
        return segmentation

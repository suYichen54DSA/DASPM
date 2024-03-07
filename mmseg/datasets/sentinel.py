# from .gaofen import gaofenDataset
from .builder import DATASETS
from .custom import CustomDataset
from ..core.evaluation.class_names import gaofen2m_class,gaofen2m_palette
import os.path as osp

import mmcv
import numpy as np
from PIL import Image

@DATASETS.register_module()
class sentinelDataset(CustomDataset):
    # CLASSES = CustomDataset.CLASSES
    # PALETTE = CustomDataset.PALETTE
    # CLASSES = [
    #     'Paddy field', 'Dryland', 'Orchard', 'Forest land', 'Grassland',
    #     'Urban area', 'Roads', 'Impervious surface', 'Agricultural greenhouse',
    #     'Bare land', 'Water', 'Ice and snow accumulation', 'Others'
    # ]
    # PALETTE = [[160,120,0],[200,140,0],[0,139,139],[0,139,0],[0,238,0],
    #         [205,20,20],[255,255,0],[196,196,0],[153,20,20],[200,200,200],
    #         [2,2,141],[160,160,255],[0,255,255]]
    # CLASSES = ['Others','green area', 
    #     'Urban area', 'Roads', 'Impervious surface',
    #     'Water',
    # ]
    
    # PALETTE = [[0,255,255],[0,238,0],
    #         [205,20,20],[255,255,0],[196,196,0],
    #         [2,2,141]]
    CLASSES = ['green area', 
        'Urban area', 'Roads', 'Impervious surface',
        'Water',
    ]
    PALETTE = [[0,238,0],
            [205,20,20],[255,255,0],[196,196,0],
            [2,2,141]]
    def __init__(self, **kwargs):
        # assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(sentinelDataset, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='.tif',
            split=None,
            **kwargs)
    def results2img(self, results, imgfile_prefix, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 6.
            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self, results, imgfile_prefix, indices=None):
        """Format the results into dir (standard format for LoveDA evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """
        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, indices)

        return result_files
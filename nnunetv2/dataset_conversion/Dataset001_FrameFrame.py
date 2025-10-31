import multiprocessing
import shutil
from multiprocessing import Pool

from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
from acvl_utils.morphology.morphology_helper import generic_filter_components
from scipy.ndimage import binary_fill_holes
from os.path import join


def load_and_convert_case(
    input_image: str,
    input_seg: str,
    output_image: str,
    output_seg: str,
    min_component_size: int = 50,
):
    # seg = io.imread(input_seg)
    # image = io.imread(input_image)
    # image = image.sum(2)
    # the dataset has large white areas in which road segmentations can exist but no image information is available.
    # Remove the road label in these areas
    # io.imsave(output_seg, seg, check_contrast=False)
    shutil.copy(input_seg, output_seg)
    shutil.copy(input_image, output_image)


if __name__ == "__main__":
    # extracted archive from https://www.kaggle.com/datasets/insaff/massachusetts-roads-dataset?resource=download
    source = "/root/autodl-tmp/dataset/frame"
    nnUNet_raw = "/root/autodl-tmp/data/_raw"
    dataset_name = "Dataset001_FrameFrame"

    imagestr = join(nnUNet_raw, dataset_name, "imagesTr")
    imagests = join(nnUNet_raw, dataset_name, "imagesTs")
    labelstr = join(nnUNet_raw, dataset_name, "labelsTr")
    labelsts = join(nnUNet_raw, dataset_name, "labelsTs")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_source = join(source, "train")
    test_source = join(source, "test")

    with multiprocessing.get_context("spawn").Pool(8) as p:

        # not all training images have a segmentation
        valid_ids = subfiles(
            join(train_source, "mask", "frame"), join=False, suffix="png"
        )
        num_train = len(valid_ids)
        r = []
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    (
                        (
                            join(train_source, "image", v),
                            join(train_source, "mask", "frame", v),
                            join(imagestr, v[:-4] + "_0000.png"),
                            join(labelstr, v),
                            50,
                        ),
                    ),
                )
            )

        # test set
        valid_ids = subfiles(
            join(test_source, "mask", "frame"), join=False, suffix="png"
        )
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    (
                        (
                            join(test_source, "image", v),
                            join(test_source, "mask", "frame", v),
                            join(imagests, v[:-4] + "_0000.png"),
                            join(labelsts, v),
                            50,
                        ),
                    ),
                )
            )
        _ = [i.get() for i in r]

    generate_dataset_json(
        join(nnUNet_raw, dataset_name),
        {0: "R", 1: "G", 2: "B"},
        {"background": 0, "road": 1},
        num_train,
        ".png",
        dataset_name=dataset_name,
    )


## Event-based attention and tracking on neuromorphic hardware

This is a brian2 implementation of:

Renner, A., Evanusa, M., & Sandamirskaya, Y. (2019).
Event-based attention and tracking on neuromorphic hardware.
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 2019.
arXiv:1907.04060. [pdf](https://openaccess.thecvf.com/content_CVPRW_2019/papers/EventVision/Renner_Event-Based_Attention_and_Tracking_on_Neuromorphic_Hardware_CVPRW_2019_paper.pdf)

    @InProceedings{Renner_2019_CVPR_Workshops,
    author = {Renner, Alpha and Evanusa, Matthew and Sandamirskaya, Yulia},
    title = {Event-Based Attention and Tracking on Neuromorphic Hardware},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {June},
    year = {2019}
    }

## Install
Clone the repository.

Then download the data here:
https://polybox.ethz.ch/index.php/s/oFZhVdO9Jdcq3lm
and copy the contents into the subdirectory ./data/

Then create and environment and install the dependencies

    conda create --name dvs_tracker cython
    conda activate dvs_tracker
    pip install -r requirements.txt

## Run
    python dvs_tracker/main_brian2.py


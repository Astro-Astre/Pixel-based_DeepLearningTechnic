import os
import numpy as np
import torch
from torchvision import models
from torchvision import transforms
from grad_cam_utils import GradCAM, show_cam_on_image
from Xception import *
from DenseNet import *
from args import *
from data_handle import *
import io
import cv2
from sklearn.model_selection import train_test_split


def get_grad_cam(model, target_layers, img, true_class):
    img_tensor = torch.Tensor(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    grayscale_cam, pred = cam(input_tensor=input_tensor, target_category=true_class)
    grayscale_cam = grayscale_cam[0]
    result = show_cam_on_image(img.astype(dtype=np.float32),
                               grayscale_cam,
                               use_rgb=True)
    return result, pred


if __name__ == '__main__':
    # model = eval(data_config.model_name)(**data_config.model_parm)
    PATH = "/data/renhaoye/decals_2022/trained_model/x_ception-LR_0.0001-LS_focal_loss-CLS_7-BSZ_32-OPT_AdamW-BEST_2/model_6.pt"
    device = torch.device('cuda:0')
    with open(PATH, 'rb') as f:
        buffer = io.BytesIO(f.read())
        model = torch.load(buffer, map_location=device)
    target_layers = [model.exit_flow.conv]
    path = "/data/renhaoye/decals_2022/in_decals/decals_best/"
    files = os.listdir(path)
    i = 0
    # img = load_img(path + "122.541511_20.486553.fits", None)
    files = [x for x in files if not "rotated.fits" in x]
    files = [x for x in files if not "shifted.fits" in x]
    files = [x for x in files if not "flipped.fits" in x]

    df_auto = pd.read_csv("/data/renhaoye/decals_2022/fits.csv")

    threshold = 0.5
    threshold2 = 0.6
    merger = major = df_auto.query('merging_minor_disturbance_fraction > %f '
                                   '| merging_major_disturbance_fraction > %f '
                                   '| merging_merger_fraction > %f '
                                   % (threshold2, threshold2, threshold2))
    smoothRounded = df_auto.query('smooth_or_featured_smooth_fraction >  %f '
                                  '& how_rounded_round_fraction > %f' % (0.7, 0.8))
    smoothInBetween = df_auto.query('smooth_or_featured_smooth_fraction >  %f '
                                    '& how_rounded_in_between_fraction > %f' % (0.7, 0.85))
    smoothCigarShaped = df_auto.query('smooth_or_featured_smooth_fraction > %f '
                                      '& how_rounded_cigar_shaped_fraction > %f' % (threshold, threshold2))
    edgeOn = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
                           '& disk_edge_on_yes_fraction > %f'
                           % (threshold, 0.7))
    diskNoBar = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
                              '& disk_edge_on_no_fraction > %f '
                              '& bar_no_fraction > %f '
                              % (threshold, threshold, 0.7))
    diskStrongBar = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
                                  '& disk_edge_on_no_fraction > %f '
                                  '&bar_strong_fraction > %f '
                                  % (threshold, threshold, threshold2))
    merger_train, merger_test = train_test_split(merger, test_size=0.1, random_state=1926)
    merger_train, merger_valid = train_test_split(merger_train, test_size=0.1, random_state=1926)

    smoothRounded_train, smoothRounded_test = train_test_split(smoothRounded, test_size=0.1, random_state=1926)
    smoothRounded_train, smoothRounded_valid = train_test_split(smoothRounded_train, test_size=0.1, random_state=1926)

    smoothInBetween_train, smoothInBetween_test = train_test_split(smoothInBetween, test_size=0.1, random_state=1926)
    smoothInBetween_train, smoothInBetween_valid = train_test_split(smoothInBetween_train, test_size=0.1,
                                                                    random_state=1926)

    smoothCigarShaped_train, smoothCigarShaped_test = train_test_split(smoothCigarShaped, test_size=0.1,
                                                                       random_state=1926)
    smoothCigarShaped_train, smoothCigarShaped_valid = train_test_split(smoothCigarShaped_train, test_size=0.1,
                                                                        random_state=1926)

    edgeOn_train, edgeOn_test = train_test_split(edgeOn, test_size=0.1, random_state=1926)
    edgeOn_train, edgeOn_valid = train_test_split(edgeOn_train, test_size=0.1, random_state=1926)

    diskNoBar_train, diskNoBar_test = train_test_split(diskNoBar, test_size=0.1, random_state=1926)
    diskNoBar_train, diskNoBar_valid = train_test_split(diskNoBar_train, test_size=0.1, random_state=1926)

    diskStrongBar_train, diskStrongBar_test = train_test_split(diskStrongBar, test_size=0.1, random_state=1926)
    diskStrongBar_train, diskStrongBar_valid = train_test_split(diskStrongBar_train, test_size=0.1, random_state=1926)
    num = 20
    row_gap = np.ones((4, (data_config.num_class + 1) * 256 + (data_config.num_class + 1 - 1) * 4, 3))
    col_gap = np.ones((256, 4, 3))
    row_output = None
    poi = (0, 30)
    save = None

    df_list = [merger_test, smoothRounded_test, smoothInBetween_test, smoothCigarShaped_test, edgeOn_test, diskNoBar_test, diskStrongBar_test]
    save_name = ["merger", "smoothRounded", "smoothInBetween", "smoothCigarShaped", "edgeOn", "diskNoBar", "diskStrongBar"]
    for index in range(len(df_list)):
        for row in range(num):
            ra, dec = df_list[index].iloc[row].ra, df_list[index].iloc[row].dec
            img = load_img(path + str(ra) + "_" + str(dec) + ".fits", None)
            col_output = None
            _, pred = get_grad_cam(model, target_layers, img, None)
            pred = F.softmax(torch.Tensor(pred), dim=1)
            pred = 100 * pred.numpy()
            true = np.argmax(pred, axis=-1)
            for col in range(data_config.num_class):
                col_result, _ = get_grad_cam(model, target_layers, img, col)
                if true == col:
                    cv2.putText(col_result, "pred", (0, 250), cv2.FONT_ITALIC, 1., (1, 1, 1), 1.1)
                cv2.putText(col_result, str(pred[0][col])[:5] + "%", poi, cv2.FONT_ITALIC, 1., (1, 1, 1), 1.1)
                if col == 0:
                    col_output = np.concatenate((chw2hwc(img), col_gap, col_result), axis=1)
                else:
                    col_output = np.concatenate((col_output, col_gap, col_result), axis=1)
            if row == 0:
                row_output = col_output
            else:
                row_output = np.concatenate((row_output, row_gap, col_output), axis=0)
        plt.imsave("/data/renhaoye/decals_2022/%s.jpg" % save_name[index], row_output)

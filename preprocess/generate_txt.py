# -*- coding: utf-8-*-
import os


# noinspection PyTypeChecker
def generateTxt(txt_path, img_dir):
    zero, one, two, three, four, five, six = 0, 0, 0, 0, 0, 0, 0
    f = open(txt_path, 'w')
    for root, sub_dir, _ in os.walk(img_dir, topdown=True):  # 获取 train文件下各文件夹名称
        for label in sub_dir:
            # if label in ["0", "4", "5", "6", "7"]:
            i_dir = os.path.join(root, label)
            img_list = os.listdir(i_dir)
            for i in range(len(img_list)):
                if label == "0":
                    img_path = os.path.join(i_dir, img_list[i])
                    line = img_path + " " + str(0) + "\n"
                    zero += 1
                    f.write(line)
                if label == "1":
                    img_path = os.path.join(i_dir, img_list[i])
                    line = img_path + " " + str(1) + "\n"
                    f.write(line)
                    one += 1
                if label == "2":
                    img_path = os.path.join(i_dir, img_list[i])
                    line = img_path + " " + str(2) + "\n"
                    f.write(line)
                    two += 1
                if label == "3":
                    img_path = os.path.join(i_dir, img_list[i])
                    line = img_path + " " + str(3) + "\n"
                    f.write(line)
                    three += 1
                if label == "4":
                    img_path = os.path.join(i_dir, img_list[i])
                    line = img_path + " " + str(4) + "\n"
                    f.write(line)
                    four += 1
                if label == "5":
                    img_path = os.path.join(i_dir, img_list[i])
                    line = img_path + " " + str(4) + "\n"
                    f.write(line)
                    four += 1
                if label == "6":
                    img_path = os.path.join(i_dir, img_list[i])
                    line = img_path + " " + str(5) + "\n"
                    f.write(line)
                    five += 1
                if label == "7":
                    img_path = os.path.join(i_dir, img_list[i])
                    line = img_path + " " + str(5) + "\n"
                    f.write(line)
                    five += 1
                if label == "8":
                    img_path = os.path.join(i_dir, img_list[i])
                    line = img_path + " " + str(5) + "\n"
                    f.write(line)
                    five += 1
                if label == "9":
                    img_path = os.path.join(i_dir, img_list[i])
                    line = img_path + " " + str(6) + "\n"
                    f.write(line)
                    six += 1
    f.close()
    print(zero, one, two, three, four, five, six)


if __name__ == '__main__':
    generateTxt("/data/renhaoye/decals_2022/training_mask_7.txt",
                "/data/renhaoye/decals_2022/in_decals/masked_dataset/trainingSet")
    generateTxt("/data/renhaoye/decals_2022/test_mask_7.txt",
                "/data/renhaoye/decals_2022/in_decals/masked_dataset/testSet")
    generateTxt("/data/renhaoye/decals_2022/validation_mask_7.txt",
                "/data/renhaoye/decals_2022/in_decals/masked_dataset/validationSet")

from functools import partial
from preprocess.data_handle import *
import multiprocessing


def normalize_out_decals(i, rows):
    save_dir = "/data/renhaoye/decals_2022/out_decals/normalized_dat/"
    mkdir(save_dir)
    path = rows[i][0]
    save_name = save_dir + path.split("/")[-1].split(".fits")[0] + "raw.dat"
    if os.path.getsize(path) == 792000:
        hdul = fits.open(path)
        img, remove = normalization(hdul[0].data)
        hdul.close()
        if not remove:
            save_dat(img, save_name)


if __name__ == '__main__':
    raw_out_decals_dir = "/data/GZ_Decals/MGS_out_DECaLS/"
    empty_file = raw_out_decals_dir + "emptyfile.txt"
    raw_out_decals = pd.DataFrame(os.listdir(raw_out_decals_dir), columns=["loc"]) \
        .apply(lambda x: raw_out_decals_dir + x)
    pd.set_option('max_colwidth', 100)
    empty = pd.read_csv(empty_file, names=["loc"]).apply(lambda x: raw_out_decals_dir + x)
    out_decals = np.array(pd.merge(raw_out_decals, empty, on="loc", how="left")).tolist()
    # for i in range(1):
    #     normalize_out_decals(i, out_decals)
    index = []
    # noinspection PyTypeChecker
    for i in range(len(out_decals)):
        index.append(i)
    p = multiprocessing.Pool(20)
    p.map(partial(normalize_out_decals, rows=out_decals), index)
    p.close()
    p.join()
import json
import numpy as np
import sys
import os

folders = []
t_err = []
v_err = []
flags = []

results_dir = 'results/retrain_logs/'
for subdir, dirs, files in os.walk(results_dir):
    if "2020" not in subdir:
        continue
    for f in files:
        if "performance.json" in f:
            try:
                fpath = os.path.join(subdir, f)
                d = json.load(open(fpath))
                t_err.append(d['test'][-1])
                v_err.append(d['val'][-1])
                flags.append(d['flags'])
                folders.append(subdir.replace(results_dir, '../'))
            except Exception as e:
                print("Failed in ", subdir)

sort_idxs = np.argsort(folders)
print("test \t val  \t units \t beta \t lif \t w    str \t thrs \t dir")
for i in sort_idxs:
    if 'n_thr_spikes' not in flags[i]:
        flags[i]['n_thr_spikes'] = None
    # print("test {:.4f} val {:.4f} beta {} lif {} w{}str{} dir {}".format(
    print("{:.3f} \t {:.3f} \t {} \t {} \t {}\t{} {}\t{}\t{}".format(
          t_err[i], v_err[i], flags[i]['n_hidden'], flags[i]['beta'], flags[i]['n_lif_frac'],
          flags[i]['window_size_ms'], flags[i]['window_stride_ms'], flags[i]['n_thr_spikes'], folders[i]))
    # print(t_err[i], v_err[i], flags[i]['beta'], flags[i]['n_lif_frac'], folders[i])
# print('------------ MEAN --------------')
# print("train {:.4f} +- {:.4f} val {:.4f} +- {:.4f}".format(np.mean(t_err), np.std(t_err), np.mean(v_err), np.std(v_err)))



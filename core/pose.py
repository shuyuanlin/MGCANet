from six.moves import xrange
import numpy as np
import os
def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs

def auc(cur_err_q, cur_err_t, res_path):
    #  AUC compute
    pose_errors = []
    for idx in range(len(cur_err_q)):
        pose_error = np.maximum(cur_err_q[idx], cur_err_t[idx])
        pose_errors.append(pose_error)
    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds)
    aucs = [100. * yy for yy in aucs]
    # print('Evaluation Results (mean over {} pairs):'.format(len(measure_list[0])))
    # print('AUC@5\t AUC@10\t AUC@20\t')
    # print('{:.2f}\t {:.2f}\t {:.2f}\t'.format(aucs[0], aucs[1], aucs[2]))
    # ofn = os.path.join(res_path, "auc5_10_20.txt")
    # np.savetxt(ofn, aucs)
    auc_result = 'AUC@5\t AUC@10\t AUC@20\t\n{:.2f}\t {:.2f}\t {:.2f}\t\n'.format(aucs[0], aucs[1], aucs[2])
    auc_file = os.path.join(res_path, "auc.txt")
    with open(auc_file, "w") as f:
        f.write(auc_result)

def map(qt_acc, ths, res_path, tag):
    map = []
    for _idx_th in xrange(1, len(ths)):
        map.append(np.mean(qt_acc[:_idx_th] * 100))
        # save qt result
    ofn = os.path.join(res_path, "acc_qt_auc_{}.txt".format(tag))
    with open(ofn, "w") as ofp:
        for _idx_th in xrange(1, len(ths)):
            idx_th = "acc_qt_auc" + str(ths[_idx_th]) + "_" + str(tag) + ":\n"
            ofp.write(idx_th)
            ofp.write("{}\n\n".format(np.mean(qt_acc[:_idx_th])))

    map_result = 'map@5\t map@10\t map@15\t map@20\t\n{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t\n'.format(map[0],
                                                                                                    map[1],
                                                                                                    map[2],
                                                                                                    map[3])
    map_file = os.path.join(res_path, "map.txt")
    with open(map_file, "w") as f:
        f.write(map_result)
    # print('map@5\t map@10\t map@15\t map@20\t')
    # print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(map[0], map[1], map[2], map[3]))

    # #----------------计算map与auc----------------
    # from pose import auc,map
    # auc(cur_err_q, cur_err_t, res_path)
    # map(qt_acc, ths, res_path, tag)
    # #-------------------------------------------

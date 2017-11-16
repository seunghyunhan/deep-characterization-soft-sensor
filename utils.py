import tensorflow as tf
import numpy as np
import time


plot_time = time.strftime('%m%d-%H-%M', time.localtime(time.time()))


FLAGS = tf.app.flags.FLAGS

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Array format to Location
def array2loc(in_arr):
    location = 0
    for i in range(FLAGS.num_class):
        if in_arr[i] == 1:
            location = i + 1
            break
    return location

# Location to Array format
def loc2array(num_array, idx):
    arr = [0] * num_array
    arr[idx-1] = 1
    return arr

# Randomize the indexes
def preprocessing(data):
    len_data = np.shape(data)[0]
    batchdataindex = range(FLAGS.seq_length, len_data)
    permindex = np.array(batchdataindex)
    rng = np.random.RandomState(FLAGS.random_seed)
    rng.shuffle(permindex)
    return permindex


def print_result(options, est_f, ref_f, est_loc, ref_loc, test_seq):
    total_force_truth_results = [[], [], []]
    est_results = [[], [], []]

    each_force_truth_results = [[], [], [], [], []]
    force_est_results = [[], [], [], [], []]

    interval = 50
    for i in range(len(options)):
        x_location = int(options[i][1])-1
        est_location = np.argmax(est_loc[i])
        ref_force = -float(ref_f[i])
        est_force = -float(est_f[i])

        total_force_truth_results[x_location].append(ref_force)
        est_results[x_location].append(est_force)

        ref_force_kPa = ref_force / (25*np.pi) *1000

        each_force_truth_results[int(ref_force_kPa/(interval))].append(x_location)
        force_est_results[int(ref_force_kPa/(interval))].append(est_location)


    # Total Regression Results
    RMSE = rmse(np.array(est_f), np.array(ref_f))
    NRMSE = RMSE / (np.max(est_f) - np.min(est_f))
    RMSE_kPa = RMSE / (25*np.pi) * 1000
    print("== Regression Result ==")
    print("Overall  RMSE: {:.4}	| NRMSE %: {:.3}".format(RMSE_kPa, NRMSE * 100))
    print("=========================================")

    # Each Regression Result
    for c in range(FLAGS.num_class):
        RMSE = rmse(np.array(est_results[c]), np.array(total_force_truth_results[c]))
        NRMSE = RMSE / (np.max(est_results[c]) - np.min(est_results[c]))
        RMSE_kPa = RMSE / (25 * np.pi) * 1000
        print("Loc {} |  RMSE: {:.4}	| NRMSE %: {:.3}".format(c, RMSE_kPa, NRMSE * 100))
    print("")


    TOTAL = 0
    SUCCESS = 1
    accurate_cnt = 0
    accurate_cnts = np.zeros((FLAGS.num_class, 2))
    loc_results = np.zeros((FLAGS.num_class, FLAGS.num_class))
    for i in range(len(est_loc)):
        ref = int(ref_loc[i]) - 1
        loc = np.argmax(est_loc[i])
        loc_results[ref][loc] += 1
        accurate_cnts[ref][TOTAL] += 1  # count total
        if ref == loc:
            accurate_cnts[ref][SUCCESS] += 1  # count success
            accurate_cnt += 1
        # print ("{:5}: REF {} / EST {}".format(int(merge[0]), merge[1], merge[2]))

    # Total Location Result
    print("== Localization Result ==")
    print("Overall {}/{} : {:.2f}%".format(accurate_cnt, len(est_loc), accurate_cnt / len(est_loc) * 100))
    print("=========================================")

    # Each Location Result
    for i in range(FLAGS.num_class):
        print("Loc {} | {:4}/{:4} : {:.2f}%   {:4}|{:4}|{:4}".format(i + 1, int(accurate_cnts[i][SUCCESS]),
                                                                    int(accurate_cnts[i][TOTAL]), round(
                accurate_cnts[i][SUCCESS] / accurate_cnts[i][TOTAL] * 100, 4)
                                                                    , int(loc_results[i][0]), int(loc_results[i][1]),
                                                                    int(loc_results[i][2])))

    #loc_result_cnts = np.zeros((FLAGS.num_class, len(force_truth_results), FLAGS.num_class))
    loc_accurate_cnts = np.zeros((FLAGS.num_class, len(each_force_truth_results), 2))
    for i in range(len(each_force_truth_results)):
        for j in range(len(each_force_truth_results[i])):
            x_location = each_force_truth_results[i][j]
            #loc_result_cnts[x_location][force_est_results[i][j]] += 1
            loc_accurate_cnts[x_location][i][TOTAL] += 1
            if x_location == force_est_results[i][j]:
                loc_accurate_cnts[x_location][i][SUCCESS] +=1

    for r in range(len(each_force_truth_results)):
        loc_overall_success = loc_accurate_cnts[0][r][SUCCESS] + loc_accurate_cnts[1][r][SUCCESS] + loc_accurate_cnts[2][r][SUCCESS]
        loc_overall_total = loc_accurate_cnts[0][r][TOTAL] + loc_accurate_cnts[1][r][TOTAL] +  loc_accurate_cnts[2][r][TOTAL]
        if loc_overall_total != 0:
             loc_overall = loc_overall_success / loc_overall_total * 100
        else: loc_overall = 0
        if loc_accurate_cnts[0][r][TOTAL] != 0 :
            loc1 = loc_accurate_cnts[0][r][SUCCESS] / loc_accurate_cnts[0][r][TOTAL] * 100
        else : loc1 = 0
        if loc_accurate_cnts[1][r][TOTAL] != 0:
            loc2 = loc_accurate_cnts[1][r][SUCCESS] / loc_accurate_cnts[1][r][TOTAL] * 100
        else : loc2 = 0
        if loc_accurate_cnts[2][r][TOTAL] != 0:
            loc3 = loc_accurate_cnts[2][r][SUCCESS] / loc_accurate_cnts[2][r][TOTAL] * 100
        else : loc3 = 0
        print("Range {:2}<=kPa<{:2} | {:3.2f}% | {:3.2f}% | {:3.2f}% | {:3.2f}%".format(r*interval, (r+1)*interval, loc_overall, loc1, loc2, loc3))

import numpy as np
import copy
import argoverse
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
from argoverse.utils.centerline_utils import get_nt_distance

def compute_metrics(predictions, agent_traj):
    #target_traj = np.asarray(test_seq.agent_traj)
    distances = predictions-agent_traj
    error = np.linalg.norm(distances[:,20:,:],axis=2)
    final_error = error[:,-1]
    fde_idx = np.argmin(final_error)
    fde = final_error[fde_idx]
    ade = np.mean(error[fde_idx])
    misses = len(final_error[final_error > 2.0])
    mr = misses/predictions.shape[0]
    return fde_idx, fde, ade, mr

def traj_pts_inside_da(agent_traj, city_map, city_transform):
    traj_h = np.concatenate((agent_traj, np.ones((len(agent_traj),1))), axis=1)
    traj_t = np.matmul(city_transform, traj_h.T)
    count = 0
    for x,y in zip(traj_t[0], traj_t[1]):
        y,x = int(y),int(x)
        count += city_map[y][x]
    return count
def normalize_trajectory(traj):
    norm_traj = traj - traj[0]
    theta = -np.arctan2(norm_traj[19][1],norm_traj[19][0])
    rot_mat = ((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta)))
    norm_traj = np.matmul(rot_mat, norm_traj.T).T
    return norm_traj

def build_lookup(afl, output_path):
    if os.path.exists(output_path):
        sys.exit(f'output path \"{output_path}\" already exists')
    num_seqs = len(afl)
    lookup = [None] * num_seqs
    #print(f'Processing {num_seqs} sequences')
    for idx in tqdm(range(num_seqs)):
        seq = afl[idx]
        lookup[idx] = normalize_trajectory(seq.agent_traj)
    lookup = np.asarray(lookup)
    np.save(output_path, lookup)
    
def get_centerline_nt_attributes(agent_traj, city, avm):
    candidate_centerlines = avm.get_candidate_centerlines_for_traj(agent_traj, city)
    nt_distances = [get_nt_distance(agent_traj,centerline) for centerline in candidate_centerlines]
    return (candidate_centerlines, nt_distances)
    
def build_lookup_map(afl, avm, output_path):
    if os.path.exists(output_path):
        sys.exit(f'output path \"{output_path}\" already exists')
    num_seqs = len(afl)
    lookup = [None] * num_seqs
    print(f'Processing {num_seqs} sequences')
    for idx in tqdm(range(num_seqs)):
        seq = afl[idx]
        lookup[idx] = get_centerline_nt_attributes(seq, avm)
        np.save(output_path, lookup)

def get_top_k(target, lookup, k=6):
    axis = tuple([i for i in range(1,len(lookup.shape))])
    distances = np.subtract(lookup[:,:20], target[:20])
    distances = np.linalg.norm(distances, axis=axis)
    top_k_idxs = np.argsort(distances)
    return distances, top_k_idxs[:k]



def get_norms(nt_distances):
    norms = [None] * len(nt_distances)
    for idx, example in enumerate(nt_distances):
        best_centerline_nt = example[0]
        best_centerline_norm = best_centerline_nt[:,0]
        norms[idx] = best_centerline_norm
    norms = np.asarray(norms)
    return norms

def smooth_prediction(predict_traj, agent_traj, num_pts=8):
    eps = 1e-3
    agent_shift = np.asarray([0,0]).astype(float)
    for idx in range(num_pts):
        agent_shift += agent_traj[19-idx] - agent_traj[19-idx-1]
    agent_shift /= max(np.linalg.norm(agent_shift), eps)
    agent_dx, agent_dy = agent_shift
    target_x = agent_traj[19,0] + agent_dx
    target_y = agent_traj[19,1] + agent_dy
    shift_x = target_x - predict_traj[20,0]
    shift_y = target_y - predict_traj[20,1]
    predict_traj += (shift_x, shift_y)
    
    pred_shift = np.asarray([0,0]).astype(float)
    for idx in range(num_pts):
        pred_shift += predict_traj[19-idx] - predict_traj[19-idx-1]
    pred_shift /= max(np.linalg.norm(pred_shift), eps)
    pred_dx, pred_dy = pred_shift
    
    #agent_dx, agent_dy = agent_traj[19] - agent_traj[18]
    #target_x = agent_traj[19,0] + agent_dx
    #target_y = agent_traj[19,1] + agent_dy
    #shift_x = target_x - predict_traj[20,0]
    #shift_y = target_y - predict_traj[20,1]
    #predict_traj += shift
    
    
    #pred_dx, pred_dy = predict_traj[19] - predict_traj[18]
    #pred_dx /= pred_dx + pred_dy
    #pred_dy /= pred_dx + pred_dy
    #agent_dx /= agent_dx + agent_dy
    #agent_dy /= agent_dx + agent_dy
    rot_mat = ((pred_dx*agent_dx + pred_dy*agent_dy, -(pred_dx*agent_dy-pred_dy*agent_dx)),
               (pred_dx*agent_dy - agent_dx*pred_dy, pred_dx*agent_dx + pred_dy*agent_dy))
    offset = copy.deepcopy(predict_traj[19])
    predict_traj -= offset
    #rot_mat = ((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta)))
    predict_traj = np.matmul(rot_mat, predict_traj.T).T
    predict_traj += offset
    return predict_traj

def get_multiple_forecasts(agent_traj, lookup, k=6, plot=False, is_test=False):
    norm_agent_traj = normalize_trajectory(agent_traj)
    distances, top_k_idxs = get_top_k(norm_agent_traj, lookup, k=6)
    predictions = [None] * k
    origin_agent_traj = agent_traj - agent_traj[0]
    agent_dir = origin_agent_traj[19] - origin_agent_traj[0]
    agent_dir /= np.linalg.norm(agent_dir)
    agent_dx, agent_dy = agent_dir
    for idx, k in enumerate(top_k_idxs):
        predict_traj = lookup[k]
        # using cosine dot product relation to determine angular difference between trajectories
        pred_dir = predict_traj[19] - predict_traj[0]
        pred_dir /= np.linalg.norm(pred_dir)
        pred_dx, pred_dy = pred_dir
        rot_mat = ((pred_dx*agent_dx + pred_dy*agent_dy, -(pred_dx*agent_dy-pred_dy*agent_dx)),
                   (pred_dx*agent_dy - agent_dx*pred_dy, pred_dx*agent_dx + pred_dy*agent_dy))
        rot_mat = np.asarray(rot_mat)
        '''
        arg = np.dot(predict_traj[19], origin_agent_traj[19])/ np.linalg.norm(predict_traj[19])/np.linalg.norm(origin_agent_traj[19])
        theta = np.arccos(np.clip(arg, -1.0, 1.0))
        if origin_agent_traj[19,1] < 0:
            # rotate in negative angle direction (cw)
            theta *= -1
        rot_mat = ((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta)))
        '''
        t_predict_traj = np.matmul(rot_mat, predict_traj.T).T + agent_traj[0]
        #if np.linalg.norm(t_predict_traj[20]-agent_traj[19]) > 1:
        t_predict_traj = smooth_prediction(t_predict_traj, agent_traj)
        t_predict_traj[:20] = agent_traj[:20]
        predictions[idx] = t_predict_traj
    predictions = np.asarray(predictions)
    
    #metrics = compute_metrics(predictions, test_seq)
    if not is_test:
        metrics = compute_metrics(predictions, agent_traj)
        fde_idx, fde, ade, mr = metrics
    else:
        metrics = None
    if plot:
        boldwidth=3
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
        #ax = plt.gca().set_aspect('equal')
        # plot starting point
        ax[0].set_title('predictions')
        ax[0].plot(agent_traj[0,0], agent_traj[0,1],'-o', c='r')
        ax[0].plot(agent_traj[:,0], agent_traj[:,1],c='r',linewidth=boldwidth)
        for idx, prediction in enumerate(predictions):
            if not is_test:
                linewidth = boldwidth if idx == fde_idx else 1
                linecolor = 'b' if idx == fde_idx else np.random.random(3,)
            else:
                linewidth = 1
                linecolor = np.random.random(3,)
            #ax[0].plot(prediction[19,0], prediction[19,1],'-o',c=linecolor)
            ax[0].plot(prediction[19:,0], prediction[19:,1],c=linecolor, linewidth=linewidth)
        ax[1].set_title('normalized predictions')
        ax[1].plot(norm_agent_traj[0,0], norm_agent_traj[0,1],'-o', c='r')
        ax[1].plot(norm_agent_traj[:,0], norm_agent_traj[:,1],c='r',linewidth=boldwidth)
        for idx, k in enumerate(top_k_idxs):
            if not is_test:
                linewidth = boldwidth if idx == fde_idx else 1
                linecolor = 'b' if idx == fde_idx else np.random.random(3,)
            else:
                linewidth = 1
                linecolor = np.random.random(3,)
            norm_prediction = lookup[k]
            #ax[1].plot(norm_prediction[19,0], norm_prediction[19,1],'-o',c=linecolor)
            ax[1].plot(norm_prediction[:,0], norm_prediction[:,1],c=linecolor, linewidth=linewidth)
    return top_k_idxs, predictions, metrics

def get_multiple_forecasts_norm(agent_traj, city, norm_train_trajs, train_nt_dists, agent_nt, avm, map_info, abs_k=-1, norm_k=6, plot=False, is_test=False):
    # init predictions and get top k matches
    predictions = list()
    norm_agent_traj = normalize_trajectory(agent_traj)
    norm_dists, top_k_abs_idxs = get_top_k(norm_agent_traj, norm_train_trajs, k=abs_k)
    normals = get_norms(train_nt_dists[top_k_abs_idxs])
    agent_n = agent_nt[0][:,0]
    if abs_k == -1:
        abs_k = len(normals)
    nt_dists, top_k_norm_idxs = get_top_k(agent_n, normals, k=abs_k)
    top_k_idxs = top_k_abs_idxs[top_k_norm_idxs]
    city_map, city_transform = map_info[city]['map'], map_info[city]['transform']
    
    # get direction information for alignment
    origin_agent_traj = agent_traj - agent_traj[0]
    agent_dir = origin_agent_traj[19] - origin_agent_traj[0]
    agent_dir /= np.linalg.norm(agent_dir)
    agent_dx, agent_dy = agent_dir
    for idx, k in enumerate(top_k_idxs):
        predict_traj = norm_train_trajs[k]
        # using cosine dot product relation to determine angular difference between trajectories
        pred_dir = predict_traj[19] - predict_traj[0]
        pred_dir /= np.linalg.norm(pred_dir)
        pred_dx, pred_dy = pred_dir
        rot_mat = ((pred_dx*agent_dx + pred_dy*agent_dy, -(pred_dx*agent_dy-pred_dy*agent_dx)),
                   (pred_dx*agent_dy - agent_dx*pred_dy, pred_dx*agent_dx + pred_dy*agent_dy))
        rot_mat = np.asarray(rot_mat)
        '''
        arg = np.dot(predict_traj[19], origin_agent_traj[19])/ np.linalg.norm(predict_traj[19])/np.linalg.norm(origin_agent_traj[19])
        theta = np.arccos(np.clip(arg, -1.0, 1.0))
        if origin_agent_traj[19,1] < 0:
            # rotate in negative angle direction (cw)
            theta *= -1
        rot_mat = ((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta)))
        '''
        t_predict_traj = np.matmul(rot_mat, predict_traj.T).T + agent_traj[0]
        #if np.linalg.norm(t_predict_traj[20]-agent_traj[19]) > 1:
        t_predict_traj = smooth_prediction(t_predict_traj, agent_traj)
        t_predict_traj[:20] = agent_traj[:20]
        if traj_pts_inside_da(t_predict_traj, city_map, city_transform) > 45:
            
            predictions.append(t_predict_traj)
            if len(predictions) == norm_k:
                break
                
        if idx > 1000:
            if len(predictions) > 0:
                diff = norm_k - len(predictions)
                for i in range(diff):
                    predictions.append(predictions[0])
                break
    predictions = np.asarray(predictions)
    
    #metrics = compute_metrics(predictions, test_seq)
    if not is_test:
        metrics = compute_metrics(predictions, agent_traj)
        fde_idx, fde, ade, mr = metrics
    else:
        metrics = None
    if plot:
        boldwidth=3
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
        #ax = plt.gca().set_aspect('equal')
        # plot starting point
        ax[0].set_title('predictions')
        ax[0].plot(agent_traj[0,0], agent_traj[0,1],'-o', c='r')
        ax[0].plot(agent_traj[:,0], agent_traj[:,1],c='r',linewidth=boldwidth)
        for idx, prediction in enumerate(predictions):
            if not is_test:
                linewidth = boldwidth if idx == fde_idx else 1
                linecolor = 'b' if idx == fde_idx else np.random.random(3,)
            else:
                linewidth = 1
                linecolor = np.random.random(3,)
            #ax[0].plot(prediction[19,0], prediction[19,1],'-o',c=linecolor)
            ax[0].plot(prediction[19:,0], prediction[19:,1],c=linecolor, linewidth=linewidth)
        ax[1].set_title('normalized predictions')
        ax[1].plot(norm_agent_traj[0,0], norm_agent_traj[0,1],'-o', c='r')
        ax[1].plot(norm_agent_traj[:,0], norm_agent_traj[:,1],c='r',linewidth=boldwidth)
        for idx, k in enumerate(top_k_idxs[:norm_k]):
            if not is_test:
                linewidth = boldwidth if idx == fde_idx else 1
                linecolor = 'b' if idx == fde_idx else np.random.random(3,)
            else:
                linewidth = 1
                linecolor = np.random.random(3,)
            norm_prediction = norm_train_trajs[k]
            #ax[1].plot(norm_prediction[19,0], norm_prediction[19,1],'-o',c=linecolor)
            ax[1].plot(norm_prediction[:,0], norm_prediction[:,1],c=linecolor, linewidth=linewidth)
    return top_k_idxs, predictions, metrics
import os
import pdb
import cv2
import time
import itertools
import os.path as osp
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import ops

from models.unitrack.model import AppearanceModel, partial_load
from models.unitrack.utils.log import logger
from models.unitrack.core.association import matching
from models.unitrack.core.propagation import propagate
from models.unitrack.core.motion.kalman_filter import KalmanFilter

from models.unitrack.utils.box import *
from models.unitrack.utils.mask import *
from models.unitrack.basetrack import *

from models.unitrack.data.query_feat_tracklet import QueryFeatTube




class AssociationTracker(object):
    def __init__(self, tracker_cfg):
        self.tracker_cfg = tracker_cfg
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []     # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        
        self.query_feat_tubes = [] # where to store all query feat tubes of this video - always sorted by track_id!

        self.frame_id = 0
        self.det_thresh = tracker_cfg.mots.conf_thres
        self.buffer_size = tracker_cfg.mots.track_buffer
        self.max_time_lost = self.buffer_size

        self.kalman_filter = KalmanFilter()

        self.app_model = AppearanceModel(tracker_cfg).to(tracker_cfg.common.device)
        self.app_model.eval()
        
        if not self.tracker_cfg.mots.asso_with_motion:
            self.tracker_cfg.mots.motion_lambda = 1
            self.tracker_cfg.mots.motion_gated = False
        
    def extract_emb(self, img, obs):
        raise NotImplementedError

    def prepare_obs(self, img, img0, obs, embs=None):
        raise NotImplementedError

    def update(self, img, img0, obs, query_feats, total_num_tubes_previous, yembs=None):
        torch.cuda.empty_cache()
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
 
        t1 = time.time()
        detections = self.prepare_obs(img, img0, obs, embs=None)

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        tracks = joint_stracks(tracked_stracks, self.lost_stracks)
        dists, recons_ftrk = matching.reconsdot_distance(tracks, detections)
        if self.tracker_cfg.mots.use_kalman: 
            # Predict the current location with KF
            STrack.multi_predict(tracks)
            dists = matching.fuse_motion(self.kalman_filter, dists, tracks, detections, 
                    lambda_=self.tracker_cfg.mots.motion_lambda, gate=self.tracker_cfg.mots.motion_gated)
        if obs.shape[1] == 6:
            dists = matching.category_gate(dists, tracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            track = tracks[itracked]
            det = detections[idet] # "detections" - in input order of masks
            # update qf_tube ------------------------------------------------------------------------
            query_feat = query_feats[idet]
            self.query_feat_tubes[track.track_id - 1 - total_num_tubes_previous].update(query_feat, self.frame_id)
            # ----------------------------------------------------------------------------------------
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        if self.tracker_cfg.mots.use_kalman:
            '''(tracker_cfgional) Step 3: Second association, with IOU'''
            tracks = [tracks[i] for i in u_track if tracks[i].state==TrackState.Tracked]
            detections = [detections[i] for i in u_detection]
            # update query_fetas, keep only uncomfirmed ones ------------------------------------
            query_feats = [query_feats[i] for i in u_detection]
            # -----------------------------------------------------------------------------------
            dists = matching.iou_distance(tracks, detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
            
            for itracked, idet in matches:
                track = tracks[itracked]
                det = detections[idet]
                # update qf_tube ------------------------------------------------------------------------
                query_feat = query_feats[idet]
                self.query_feat_tubes[track.track_id - 1 - total_num_tubes_previous].update(query_feat, self.frame_id)
                # ----------------------------------------------------------------------------------------
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            detections = [detections[i] for i in u_detection]
            # update query_fetas, keep only uncomfirmed ones --------- ---------------------------
            query_feats = [query_feats[i] for i in u_detection]
            # -----------------------------------------------------------------------------------
            dists = matching.iou_distance(unconfirmed, detections)  # find if any uncomfirmed track can match this u_detection
            matches, u_unconfirmed, u_detection = matching.linear_assignment(
                    dists, thresh=self.tracker_cfg.mots.confirm_iou_thres)
            for itracked, idet in matches:
                unconfirmed[itracked].update(detections[idet], self.frame_id)
                activated_stracks.append(unconfirmed[itracked])
                # update qf_tube ------------------------------------------------------------------------
                query_feat = query_feats[idet]
                self.query_feat_tubes[unconfirmed[itracked].track_id - 1 - total_num_tubes_previous].update(query_feat, self.frame_id)
                # ----------------------------------------------------------------------------------------
            for it in u_unconfirmed:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)  # do we need to do something here ?

        for it in u_track:
            track = tracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            # init new query_feat_tube --------------------------------------------------------------------
            query_feat = query_feats[inew]
            query_feat_tube = QueryFeatTube(self.frame_id, track.track_id, query_feat)
            self.query_feat_tubes.append(query_feat_tube)
            #----------------------------------------------------------------------------------------------
            activated_stracks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
                self.tracked_stracks, self.lost_stracks, ioudist=self.tracker_cfg.mots.dup_iou_thres)   #???

        # sort query_feat_tubes by track_id
        self.query_feat_tubes = sorted(self.query_feat_tubes, key=lambda qf_tube: qf_tube.track_id)
        num_tubes_this_video = len(self.query_feat_tubes)

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks, num_tubes_this_video

    def reset_all(self, ):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0
import numpy as np
from collections import deque

class QueryFeatTube(object):
    def __init__(self, start_frame_id, track_id, query_feat):
        self.track_id = track_id
        self.start_frame_id = start_frame_id
        self.end_frame_id = start_frame_id
        self.len = 1
        qf_tube = self._init_previous_tube()
        qf_tube.append(query_feat)
        self.qf_tube = qf_tube
    
    def __repr__(self):
        return 'QFT_{}_({}_{})'.format(self.track_id, self.start_frame_id, self.end_frame_id)
        
    def _init_previous_tube(self):
        # called when a new detection is found - fill previous position with "None"
        qf_tube = [None for i in range(self.start_frame_id - 1)]
        return qf_tube


    def update(self, query_feat, cur_frame_id):
        if self.end_frame_id < cur_frame_id:
            self.qf_tube.extend([None for i in range(cur_frame_id - self.end_frame_id - 1)])
        self.qf_tube.append(query_feat)
        self.end_frame_id = cur_frame_id
        self.len += 1

    def complete_empty_postfix(self, last_frame_idx):
        if len(self.qf_tube) == last_frame_idx + 1:
            return self
        self.qf_tube.extend([None for i in range(last_frame_idx + 1 - self.end_frame_id)])
        # self.end_frame_id = last_frame_idx + 1
        return self


        
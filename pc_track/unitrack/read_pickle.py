import pickle
import sys
sys.path.append("/mnt/lustre/jkyang/wxpeng/CVPR23/PVSG_Image")

pickle_file = pickle.load(open('/mnt/lustre/jkyang/PSG4D/OpenPVSG/models/unitrack/query_feats.pickle','rb'))
print(pickle_file[0])
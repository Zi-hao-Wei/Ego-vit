import os
import pdb

import tqdm
import random
from abc import abstractmethod

import av
import cv2
import decord
import ffmpeg
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms
from torchvision.transforms import ToPILImage
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs

def sample_frames_clips(start, end, vlen, acc_samples):
    start = max(0, start)
    end = min(vlen, end)

    intervals = np.linspace(start=start, stop=end, num=int(acc_samples) + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges
                      ]
    return frame_idxs

def sample_frames_start_end(num_frames, start, end, sample='uniform', fix_start=None):
    acc_samples = min(num_frames, end)
    intervals = np.linspace(start=start, stop=end, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs

def read_frames_cv2(video_path, num_frames, sample='uniform', fix_start=None):
    cap = cv2.VideoCapture(video_path)
    assert (cap.isOpened())
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get indexes of sampled frames
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = []
    success_idxs = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass
            # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')

    frames = torch.stack(frames).float() / 255
    cap.release()
    return frames, success_idxs

def read_frames_av(video_path, num_frames, sample='uniform', fix_start=None):
    reader = av.open(video_path)
    try:
        frames = []
        frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
    except (RuntimeError, ZeroDivisionError) as exception:
        print('{}: WEBM reader cannot open {}. Empty '
              'list returned.'.format(type(exception).__name__, video_path))
    vlen = len(frames)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = torch.stack([frames[idx] for idx in frame_idxs]).float() / 255
    frames = frames.permute(0, 3, 1, 2)
 
def get_video_len(video_path):
    cap = cv2.VideoCapture(video_path)
    if not (cap.isOpened()):
        return False
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return vlen


video_reader = {
    'av': read_frames_av,
    'cv2': read_frames_cv2,
}


class MSRVTT_test(Dataset):
    def __init__(self,
                    video_params={
                    "input_res": 224,
                    "num_frames": 16,
                    "loading": "lax"
                    },
                 tsfms=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 neg_param=None,
                 ):
        self.video_params = video_params
        self.split = "train"
        norm_mean=(0.485, 0.456, 0.406)
        norm_std=(0.229, 0.224, 0.225)
        normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
        self.transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0, saturation=0, hue=0),
            normalize,
        ])
        self.cut = cut
        self.subsample = subsample
        self.sliding_window_stride = sliding_window_stride
        self.video_reader = video_reader[reader]
        self.label_type = 'caption'
        self.neg_param = neg_param
        data = pd.read_csv("./MSRVTT_test.csv",sep=",",on_bad_lines="warn")
        data["path"]="./TestVideo/"+data["video_id"]+".mp4"
        self.metadata = data

        if self.sliding_window_stride != -1:
            if self.split != 'test':
                raise ValueError('Fixing frame sampling is for test time only. can remove but...')
            self._fix_temporal_samples()



    def _get_video_lens(self):
        vlen_li = []
        for idx, row in self.metadata.iterrows():
            video_path = self._get_video_path(row)[0]
            vlen_li.append(get_video_len(video_path))

        return vlen_li

    def _fix_temporal_samples(self):
        self.metadata['vlen'] = self._get_video_lens()
        self.metadata['frame_intervals'] = self.metadata['vlen'].apply(
            lambda x: np.linspace(start=0, stop=x, num=min(x, self.video_params['num_frames']) + 1).astype(int))
        self.metadata['fix_start'] = self.metadata['frame_intervals'].apply(
            lambda x: np.arange(0, int(x[-1] / len(x - 1)), self.sliding_window_stride)
        )
        self.metadata = self.metadata.explode('fix_start')

    def __len__(self):
        return self.metadata.__len__()

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp = sample["path"]
        caption = sample["caption"]
        
        # print(video_fp,caption)
        video_loading = self.video_params.get('loading', 'strict')
        frame_sample = 'uniform'
        fix_start = None


        try:
            if os.path.isfile(video_fp):
                imgs, idxs = self.video_reader(video_fp, self.video_params['num_frames'], frame_sample,
                                               fix_start=fix_start)
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == 'strict':
                raise ValueError(
                    f'Video loading failed for {video_fp}, video loading for this dataset is strict.') from e
            else:
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                imgs = transforms.ToTensor()(imgs).unsqueeze(0)

        # ToPILImage()(imgs[0]).show()

        import matplotlib.pyplot as plt

        imgs = self.transforms(imgs)

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs

        data = {'video': final, 'text': caption}
        return data



# class MSVDLoader(DataLoader):
#     """
#     Base class for all data loaders
#     """
#     def __init__(self, dataset=MSVDDataset(), batch_size=16, shuffle=True, num_workers=1, collate_fn=default_collate):
#         self.shuffle = shuffle

#         self.batch_idx = 0
#         self.n_samples = dataset.__len__()

#         self.init_kwargs = {
#             'dataset': dataset,
#             'batch_size': batch_size,
#             'shuffle': self.shuffle,
#             'collate_fn': collate_fn,
#             'num_workers': num_workers,
#             'pin_memory': True
#         }
#         super().__init__(**self.init_kwargs)



if __name__=="__main__":
    d = MSVDDataset()
    print(len(d))
    loader =DataLoader(MSVDDataset(), batch_size=4, shuffle=False)

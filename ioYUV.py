import numpy as np
from tqdm.notebook import tqdm

def write_video(path, vid, w=960, h=540, ch=3):
    buf = []
    for frame in tqdm(vid):
        print(frame.shape)
        buf.extend(list(frame[:, :, 0].flatten()))
        buf.extend(list(frame[::2, ::2, 1].flatten()))
        buf.extend(list(frame[::2, ::2, 2].flatten()))
    print(len(buf))
    buf = np.asarray(buf)
    with open(path, "wb") as f:
        vid = f.write(buf.tobytes())

def read_frame(path, w=1920, h=1080, subsampling='420', bit_depth=8):
    vid = None
    with open(path, "rb") as f:
        vid = f.read()
    if vid != None:
        w_c = w
        h_c = h
        if subsampling == '420':
            w_c = w // 2
            h_c = h // 2
            
        fr_size_y = w * h
        fr_size_uv = w_c * h_c
        
        dtp = np.uint8
        if bit_depth > 8:
            dtp = np.uint16
            fr_size_y *= 2
            fr_size_uv *= 2
        
        for fr_beg in tqdm(range(0, len(vid), fr_size_y + 2 * fr_size_uv)):
            y = np.frombuffer(vid[fr_beg:fr_beg + fr_size_y], dtype=dtp).reshape((h, w))
            u = np.frombuffer(vid[fr_beg + fr_size_y:fr_beg + fr_size_y + fr_size_uv], dtype=dtp).reshape((h_c, w_c))
            v = np.frombuffer(vid[fr_beg + + fr_size_y + fr_size_uv:fr_beg + fr_size_y + 2 * fr_size_uv], dtype=dtp).reshape((h_c, w_c))
            
            if h_c == h // 2:
                u = np.repeat(u, 2, axis=0)
                v = np.repeat(v, 2, axis=0)
                
            if w_c == w // 2:
                u = np.repeat(u, 2, axis=1)
                v = np.repeat(v, 2, axis=1)

            
            yield np.stack((y,u,v), axis=2)
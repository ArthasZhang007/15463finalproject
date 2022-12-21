from matplotlib.pyplot import show
from helper import *

# foreground energy computation given label x and ratio r
def E_f(x, r, lamb = 0.2):
    return (1 - x) * (np.maximum(r, lamb) - lamb)/ (1 - lamb)

# background energy computation given label x and probability matrix p
def E_m(x, p):
    return x * (2 * np.maximum(p, 0.5) - 1)

def foreground(nf_color, f_color):
    nf = gray(nf_color)
    f = gray(f_color)
    dim = np.shape(f)
    print(dim)

    numbins = 16
    eps = 1e-3

    nf_hist, range_val = np.histogram(nf, bins=numbins, range=(0, 1))
    f_hist, _ = np.histogram(f, bins=numbins, range=(0, 1))

    # print(nf_hist)
    # print(f_hist)
    # print(range_val)
    step = range_val[1]
    ids_nf = ((nf - eps)/step).astype(int)
    ids_f = ((f - eps)/step).astype(int)
    h_nf = nf_hist[ids_nf.reshape(1,-1)].reshape(dim)
    h_f = f_hist[ids_f.reshape(1,-1)].reshape(dim)


    # for i in range(len(range_val) - 1):
    #     print(i)
    #     l = range_val[i]
    #     r = range_val[i+1]
    #     h_nf += ((nf >= l) & (nf < r)) * (nf_hist[i])
    #     h_f += ((f >= l) & (f < r)) * (f_hist[i])

    # h_nf+=eps
    # h_f+=eps

    r_f = np.maximum((h_f - h_nf), 0)/(h_f)
    r_nf = np.maximum((h_nf - h_f), 0)/(h_nf)

    writeimage(outputpath + "arthas_nf_ratio.png", r_nf)
    writeimage(outputpath + "arthas_f_ratio.png", r_f)
    # showimage_raw(r_nf)
    # showimage_raw(r_f)
    # print(r_f)
    # print(r_nf)
    return r_nf, r_f


def background(nf_color, f_color, sigma=0.1):

    nf = gray(nf_color) 
    f = gray(f_color)
    dim = np.shape(f)
    flow = cv2.calcOpticalFlowFarneback(nf, f, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    dy = flow[..., 0] * flow.shape[0]
    dx = flow[..., 1] * flow.shape[1]

    y,x = np.indices(dim)
    ny,nx = (y + dy + 0.5).astype(np.int32), (x + dx + 0.5).astype(np.int32)

    I_diff = f[ny,nx] - nf

    mu = np.mean(I_diff)
    sigma_b = np.log(2)/sigma/sigma/9
    p = np.exp(-sigma_b * (I_diff - mu) * (I_diff - mu))

    # showimage(p)
    #showmaker(p)

    return p
    


    # showimage(I_diff)
    # showmaker(I_diff)
    
    #print(np.max(y))
    #print(np.max(x))


    # print(np.max(flow))
    # # a = np.sum(flow == flow)
    # # print(a)
    # # print(np.sum(flow > 0.5)/np.sum(flow == flow))

    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # hsv = np.zeros_like(nf_color)
    # hsv[..., 0] = ang*180/np.pi/2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # showimage(bgr)

def solve(r, p, gamma_f = 10, gamma_m = 10):
    def compute_E_d(x):
        return gamma_f * E_f(x, r) + gamma_m * E_m(x, p)
    E_0,E_1 = compute_E_d(0),compute_E_d(1)
    E_min = np.minimum(E_0, E_1)
    label = (E_1 == E_min)
    return label

def post_process(I, mask):
    print("labeled ratio: ", np.sum(mask)/np.sum(mask == mask))
    B,G,R = I[...,0],I[...,1],I[...,2]
    B = (B * mask) + 1.0 * (1 - mask) # blue hue
    G = G * mask
    R = R * mask 
    res = np.dstack([B,G,R])
    # print(res.dtype)
    return res.astype(np.float32)
    
    

def main():
    prefix,suffix = 'arthas','JPG'
    # prefix,suffix = 'peter','png'
    nf = readimage(inputpath + prefix + '_i.'+ suffix)
    f = readimage(inputpath + prefix + '_f.'+ suffix)

    # global parameters
    cp_factor = 1.2  #increase contrast
    sigma = 0.10    #background probability distribution standard deviation
    gamma_f = 14    #weight of the foreground energy term
    gamma_m = 7    #weight of the background energy term
    
    # nf_int = readimage_int(inputpath + prefix + '_i.'+ suffix)
    # f_int = readimage_int(inputpath + prefix + '_i.'+ suffix)

    onf = np.clip(nf * cp_factor, 0, 1)
    of = np.clip(f * cp_factor, 0, 1)
    # showimage(nf)

    # downsample
    # nf = nf[::4,::4]
    # f = f[::4, ::4]

    r_nf,r_f = foreground(onf,of)
    p = background(onf,of, sigma)

    label_nf = solve(r_nf, p, gamma_f, gamma_m)
    label_f = solve(r_f, p,gamma_f, gamma_m)


    final_f = post_process(f, label_f)
    final_nf = post_process(nf, label_nf) * cp_factor
    
    writeimage(outputpath + 'arthas_final_f_{:.1f}_{:.2f}_{:d}_{:d}.{:s}'.format(cp_factor, sigma, gamma_f, gamma_m,suffix), final_f)
    writeimage(outputpath + 'arthas_final_nf_{:.1f}_{:.2f}_{:d}_{:d}.{:s}'.format(cp_factor, sigma, gamma_f, gamma_m,suffix), final_nf)


# a = np.array([(2,3),(5,4)])
# print(a, np.max(a))
main()
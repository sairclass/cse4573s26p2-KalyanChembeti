'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def prepare_img(img: torch.Tensor) -> torch.Tensor:
    if img.dim() == 4:
        img = img.squeeze(0)
    img = img.float()
    if img.max() > 1.0:
        img = img / 255.0
    return img.clamp(0.0, 1.0)

def to_gray(img: torch.Tensor) -> torch.Tensor:
    return K.color.rgb_to_grayscale(img.unsqueeze(0)).squeeze(0)

def harris_resp(gray: torch.Tensor) -> torch.Tensor:
    g = gray.unsqueeze(0)
    grads = K.filters.spatial_gradient(g, mode='sobel', order=1)
    ix = grads[:, :, 0]
    iy = grads[:, :, 1]

    ixx = K.filters.gaussian_blur2d(ix * ix, (5, 5), (1.0, 1.0))
    iyy = K.filters.gaussian_blur2d(iy * iy, (5, 5), (1.0, 1.0))
    ixy = K.filters.gaussian_blur2d(ix * iy, (5, 5), (1.0, 1.0))

    k = 0.04
    det = ixx * iyy - ixy * ixy
    trace = ixx + iyy
    r = det - k * trace * trace
    return r.squeeze(0).squeeze(0)

def select_keypoints(response: torch.Tensor, max_points: int = 1200, border: int = 12) -> torch.Tensor:
    h, w = response.shape
    resp = response.clone()

    resp[:border, :] = 0
    resp[-border:, :] = 0
    resp[:, :border] = 0
    resp[:, -border:] = 0

    pooled = torch.nn.functional.max_pool2d(
        resp.unsqueeze(0).unsqueeze(0), kernel_size=7, stride=1, padding=3
    )
    keep = (resp.unsqueeze(0).unsqueeze(0) == pooled) & (resp.unsqueeze(0).unsqueeze(0) > 0)
    coords = torch.nonzero(keep.squeeze(0).squeeze(0), as_tuple=False)

    if coords.shape[0] == 0:
        return torch.empty((0, 2), device=response.device, dtype=torch.float32)

    scores = resp[coords[:, 0], coords[:, 1]]
    num = min(max_points, coords.shape[0])
    _, idx = torch.topk(scores, k=num, largest=True)
    pts = coords[idx][:, [1, 0]].float()
    return pts

def ext_patch_descriptors(gray: torch.Tensor, kpts: torch.Tensor, patch_size: int = 11) -> torch.Tensor:
    if kpts.shape[0] == 0:
        return torch.empty((0, patch_size * patch_size), device=gray.device, dtype=gray.dtype)

    _, h, w = gray.shape
    n = kpts.shape[0]
    r = patch_size // 2

    ys = torch.linspace(-r, r, patch_size, device=gray.device, dtype=gray.dtype)
    xs = torch.linspace(-r, r, patch_size, device=gray.device, dtype=gray.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    base = torch.stack([xx, yy], dim=-1)

    grid = base.unsqueeze(0).repeat(n, 1, 1, 1)
    grid[..., 0] += kpts[:, 0].view(n, 1, 1)
    grid[..., 1] += kpts[:, 1].view(n, 1, 1)

    grid[..., 0] = (grid[..., 0] / max(w - 1, 1)) * 2.0 - 1.0
    grid[..., 1] = (grid[..., 1] / max(h - 1, 1)) * 2.0 - 1.0

    src = gray.unsqueeze(0).repeat(n, 1, 1, 1)
    patches = torch.nn.functional.grid_sample(
        src, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )

    desc = patches.view(n, -1)
    desc = desc - desc.mean(dim=1, keepdim=True)
    desc = desc / (desc.norm(dim=1, keepdim=True) + 1e-8)
    return desc

def detect_describe(img: torch.Tensor, max_points: int = 1200):
    gray = to_gray(img)
    resp = harris_resp(gray)
    kpts = select_keypoints(resp, max_points=max_points, border=12)
    desc = ext_patch_descriptors(gray, kpts, patch_size=11)
    return kpts, desc

def match_desc(desc1: torch.Tensor, desc2: torch.Tensor, max_matches: int = 400):
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        device = desc1.device if desc1.numel() > 0 else desc2.device
        return torch.empty((0, 2), dtype=torch.long, device=device)

    dmat = torch.cdist(desc1, desc2, p=2)
    nn12 = torch.argmin(dmat, dim=1)
    nn21 = torch.argmin(dmat, dim=0)

    ids1 = torch.arange(desc1.shape[0], device=desc1.device)
    mutual = nn21[nn12] == ids1
    idx1 = ids1[mutual]
    idx2 = nn12[mutual]

    if idx1.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=desc1.device)

    scores = dmat[idx1, idx2]
    order = torch.argsort(scores)
    order = order[:min(max_matches, order.numel())]

    matches = torch.stack([idx1[order], idx2[order]], dim=1)
    return matches

def normalize_points(pts: torch.Tensor):
    mean = pts.mean(dim=0)
    centered = pts - mean
    dist = torch.sqrt((centered ** 2).sum(dim=1) + 1e-8)
    scale = torch.sqrt(torch.tensor(2.0, device=pts.device, dtype=pts.dtype)) / (dist.mean() + 1e-8)

    t = torch.eye(3, device=pts.device, dtype=pts.dtype)
    t[0, 0] = scale
    t[1, 1] = scale
    t[0, 2] = -scale * mean[0]
    t[1, 2] = -scale * mean[1]

    ones = torch.ones((pts.shape[0], 1), device=pts.device, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=1)
    pts_n = (t @ pts_h.t()).t()
    return pts_n[:, :2], t

def dlt_homography(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    if src.shape[0] < 4:
        return torch.eye(3, device=src.device, dtype=src.dtype)

    src_n, t_src = normalize_points(src)
    dst_n, t_dst = normalize_points(dst)

    n = src.shape[0]
    a = torch.zeros((2 * n, 9), device=src.device, dtype=src.dtype)

    x = src_n[:, 0]
    y = src_n[:, 1]
    u = dst_n[:, 0]
    v = dst_n[:, 1]

    a[0::2, 0:3] = torch.stack([x, y, torch.ones_like(x)], dim=1)
    a[0::2, 6:9] = -torch.stack([u * x, u * y, u], dim=1)
    a[1::2, 3:6] = torch.stack([x, y, torch.ones_like(x)], dim=1)
    a[1::2, 6:9] = -torch.stack([v * x, v * y, v], dim=1)

    _, _, vh = torch.linalg.svd(a)
    h = vh[-1].view(3, 3)

    h = torch.linalg.inv(t_dst) @ h @ t_src
    h = h / (h[2, 2] + 1e-8)
    return h

def transform_points(h: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((pts.shape[0], 1), device=pts.device, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=1)
    warped = (h @ pts_h.t()).t()
    warped = warped[:, :2] / (warped[:, 2:3] + 1e-8)
    return warped

def ransac_hg(src: torch.Tensor, dst: torch.Tensor, num_iter: int = 800, thresh: float = 3.0):
    if src.shape[0] < 4:
        return None, None

    best_h = None
    best_inliers = None
    best_count = 0

    n = src.shape[0]
    for _ in range(num_iter):
        perm = torch.randperm(n, device=src.device)[:4]
        h = dlt_homography(src[perm], dst[perm])

        proj = transform_points(h, src)
        err = torch.sqrt(((proj - dst) ** 2).sum(dim=1))
        inliers = err < thresh
        count = int(inliers.sum().item())

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_h = h

    if best_h is None or best_count < 4:
        return None, None

    refined_h = dlt_homography(src[best_inliers], dst[best_inliers])
    return refined_h, best_inliers

def output_canvas(img1: torch.Tensor, img2: torch.Tensor, h21: torch.Tensor):
    _, h1, w1 = img1.shape
    _, h2, w2 = img2.shape

    corners1 = torch.tensor(
        [[0.0, 0.0], [w1 - 1.0, 0.0], [w1 - 1.0, h1 - 1.0], [0.0, h1 - 1.0]],
        device=img1.device, dtype=img1.dtype
    )
    corners2 = torch.tensor(
        [[0.0, 0.0], [w2 - 1.0, 0.0], [w2 - 1.0, h2 - 1.0], [0.0, h2 - 1.0]],
        device=img2.device, dtype=img2.dtype
    )

    warped2 = transform_points(h21, corners2)
    all_pts = torch.cat([corners1, warped2], dim=0)

    min_xy = torch.floor(all_pts.min(dim=0).values)
    max_xy = torch.ceil(all_pts.max(dim=0).values)

    tx = -min_xy[0]
    ty = -min_xy[1]

    out_w = int((max_xy[0] - min_xy[0] + 1).item())
    out_h = int((max_xy[1] - min_xy[1] + 1).item())

    t = torch.eye(3, device=img1.device, dtype=img1.dtype)
    t[0, 2] = tx
    t[1, 2] = ty

    return t, out_h, out_w

def warp_img_mask(img: torch.Tensor, h: torch.Tensor, out_h: int, out_w: int):
    img_b = img.unsqueeze(0)
    warped = K.geometry.transform.warp_perspective(
        img_b, h.unsqueeze(0), dsize=(out_h, out_w),
        mode='bilinear', padding_mode='zeros', align_corners=True
    ).squeeze(0)

    mask = torch.ones((1, img.shape[1], img.shape[2]), device=img.device, dtype=img.dtype).unsqueeze(0)
    warped_mask = K.geometry.transform.warp_perspective(
        mask, h.unsqueeze(0), dsize=(out_h, out_w),
        mode='bilinear', padding_mode='zeros', align_corners=True
    ).squeeze(0)
    warped_mask = (warped_mask > 0.5).float()
    return warped, warped_mask

def blend_bg(img1: torch.Tensor, mask1: torch.Tensor, img2: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    valid1 = mask1 > 0.5
    valid2 = mask2 > 0.5

    only1 = valid1 & (~valid2)
    only2 = valid2 & (~valid1)
    overlap = valid1 & valid2

    out = torch.zeros_like(img1)

    out = out + img1 * only1.float()
    out = out + img2 * only2.float()

    if overlap.any():
        diff = torch.mean(torch.abs(img1 - img2), dim=0, keepdim=True)  # 1xHxW

        stable = overlap & (diff < 0.06)
        moving = overlap & (diff >= 0.06)

        avg = 0.5 * (img1 + img2)
        out = out + avg * stable.float()

        if moving.any():
            gray1 = to_gray(img1)
            gray2 = to_gray(img2)

            score1 = torch.exp((gray1 - gray2) / 0.08)
            score2 = torch.exp((gray2 - gray1) / 0.08)

            wsum = score1 + score2 + 1e-8
            w1 = score1 / wsum
            w2 = score2 / wsum

            w1 = K.filters.gaussian_blur2d(w1.unsqueeze(0), (17, 17), (3.0, 3.0)).squeeze(0)
            w2 = K.filters.gaussian_blur2d(w2.unsqueeze(0), (17, 17), (3.0, 3.0)).squeeze(0)

            wsum = w1 + w2 + 1e-8
            w1 = w1 / wsum
            w2 = w2 / wsum

            soft_pref = img1 * w1 + img2 * w2
            mx = torch.maximum(img1, img2)

            conf = ((diff - 0.14) / 0.12).clamp(0.0, 1.0)
            motion_part = (1.0 - conf) * soft_pref + conf * mx

            out = out + motion_part * moving.float()

    valid = (valid1 | valid2).float()
    out = out * valid
    return out.clamp(0.0, 1.0)

def side_by_side(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    _, h1, w1 = img1.shape
    _, h2, w2 = img2.shape
    h = max(h1, h2)
    w = w1 + w2
    out = torch.zeros((3, h, w), device=img1.device, dtype=img1.dtype)
    out[:, :h1, :w1] = img1
    out[:, :h2, w1:w1 + w2] = img2
    return out

def finalize_img(img: torch.Tensor) -> torch.Tensor:
    if img.dim() == 4:
        img = img.squeeze(0)
    img = img.clamp(0.0, 1.0)
    img = (img * 255.0).round().to(torch.uint8)
    return img

def safe_inv_hg(h: torch.Tensor) -> torch.Tensor:
    try:
        inv_h = torch.linalg.inv(h)
        inv_h = inv_h / (inv_h[2, 2] + 1e-8)
        return inv_h
    except:
        return None

def valid_overlap(matches: torch.Tensor, inliers: torch.Tensor) -> bool:
    if matches.shape[0] < 12:
        return False
    if inliers is None:
        return False
    inlier_count = int(inliers.sum().item())
    if inlier_count < 10:
        return False
    if matches.shape[0] > 0:
        ratio = inlier_count / float(matches.shape[0])
        if ratio < 0.20:
            return False
        
    return True

def largest_conn_comp(overlap: torch.Tensor):
    n = overlap.shape[0]
    visited = [False] * n
    components = []

    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        comp = []
        visited[start] = True

        while stack:
            u = stack.pop()
            comp.append(u)
            for v in range(n):
                if overlap[u, v] > 0 and not visited[v]:
                    visited[v] = True
                    stack.append(v)

        components.append(comp)

    if len(components) == 0:
        return []
    components.sort(key=lambda x: len(x), reverse=True)

    return components[0]

def choose_ref_img(component, overlap: torch.Tensor) -> int:
    best_idx = component[0]
    best_degree = -1
    for idx in component:
        degree = int(overlap[idx].sum().item())
        if degree > best_degree:
            best_degree = degree
            best_idx = idx

    return best_idx

def global_homographies(component, ref_idx: int, pairwise_h: dict, device, dtype):
    global_h = {}
    global_h[ref_idx] = torch.eye(3, device=device, dtype=dtype)

    queue = [ref_idx]
    visited = set([ref_idx])

    while queue:
        u = queue.pop(0)
        for v in component:
            if v in visited:
                continue
            key = (v, u)
            if key in pairwise_h:
                global_h[v] = global_h[u] @ pairwise_h[key]
                visited.add(v)
                queue.append(v)

    return global_h

def panorama_canvas(images: list, global_h: dict):
    device = images[0].device
    dtype = images[0].dtype

    all_pts = []
    for idx, img in enumerate(images):
        if idx not in global_h:
            continue
        _, h, w = img.shape
        corners = torch.tensor(
            [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
            device=device, dtype=dtype
        )
        warped = transform_points(global_h[idx], corners)
        all_pts.append(warped)

    if len(all_pts) == 0:
        t = torch.eye(3, device=device, dtype=dtype)
        return t, 256, 256

    all_pts = torch.cat(all_pts, dim=0)
    min_xy = torch.floor(all_pts.min(dim=0).values)
    max_xy = torch.ceil(all_pts.max(dim=0).values)

    tx = -min_xy[0]
    ty = -min_xy[1]

    out_w = int((max_xy[0] - min_xy[0] + 1).item())
    out_h = int((max_xy[1] - min_xy[1] + 1).item())

    t = torch.eye(3, device=device, dtype=dtype)
    t[0, 2] = tx
    t[1, 2] = ty

    return t, out_h, out_w

def blend_panorama_avg(warped_imgs: list, warped_masks: list) -> torch.Tensor:
    if len(warped_imgs) == 0:
        return torch.zeros((3, 256, 256), dtype=torch.float32)

    acc = torch.zeros_like(warped_imgs[0])
    weight = torch.zeros_like(warped_masks[0])

    for img, mask in zip(warped_imgs, warped_masks):
        acc = acc + img * mask
        weight = weight + mask

    pano = acc / (weight + 1e-8)
    pano = pano.clamp(0.0, 1.0)
    return pano

# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.

    keys = list(imgs.keys())
    if len(keys) != 2:
        vals = [prepare_img(imgs[k]) for k in keys]
        if len(vals) == 0:
            return finalize_img(img)
        if len(vals) == 1:
            return finalize_img(vals[0])
        return finalize_img(side_by_side(vals[0], vals[1]))

    img1 = prepare_img(imgs[keys[0]])
    img2 = prepare_img(imgs[keys[1]])

    kpts1, desc1 = detect_describe(img1, max_points=1200)
    kpts2, desc2 = detect_describe(img2, max_points=1200)

    matches = match_desc(desc1, desc2, max_matches=400)

    if matches.shape[0] < 4:
        return finalize_img(side_by_side(img1, img2))

    pts1 = kpts1[matches[:, 0]]
    pts2 = kpts2[matches[:, 1]]
    h21, inliers = ransac_hg(pts2, pts1, num_iter=800, thresh=3.0)

    if h21 is None:
        return finalize_img(side_by_side(img1, img2))

    t, out_h, out_w = output_canvas(img1, img2, h21)
    h1_canvas = t
    h2_canvas = t @ h21

    warped1, mask1 = warp_img_mask(img1, h1_canvas, out_h, out_w)
    warped2, mask2 = warp_img_mask(img2, h2_canvas, out_h, out_w)

    img = finalize_img(blend_bg(warped1, mask1, warped2, mask2))
    
    return img

# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama, 
        overlap: torch.Tensor of the output image. 
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.
    overlap = torch.empty((3, 256, 256)) # assumed empty 256*256 overlap. Update this as per your logic.

    keys = list(imgs.keys())
    n = len(keys)

    if n == 0:
        img = finalize_img(img)
        overlap = torch.zeros((0, 0), dtype=torch.int64)
        return img, overlap

    images = [prepare_img(imgs[k]) for k in keys]
    device = images[0].device
    dtype = images[0].dtype
    all_kpts = []
    all_desc = []

    for im in images:
        kpts, desc = detect_describe(im, max_points=1200)
        all_kpts.append(kpts)
        all_desc.append(desc)

    overlap = torch.zeros((n, n), dtype=torch.int64, device=device)
    pairwise_h = {}

    for i in range(n):
        overlap[i, i] = 1

    for i in range(n):
        for j in range(i + 1, n):
            matches = match_desc(all_desc[i], all_desc[j], max_matches=400)

            if matches.shape[0] < 4:
                continue

            pts_i = all_kpts[i][matches[:, 0]]
            pts_j = all_kpts[j][matches[:, 1]]

            h_ji, inliers = ransac_hg(pts_j, pts_i, num_iter=800, thresh=3.0)

            if h_ji is None:
                continue

            if not valid_overlap(matches, inliers):
                continue

            h_ij = safe_inv_hg(h_ji)
            if h_ij is None:
                continue

            overlap[i, j] = 1
            overlap[j, i] = 1

            pairwise_h[(j, i)] = h_ji
            pairwise_h[(i, j)] = h_ij

    component = largest_conn_comp(overlap)

    if len(component) == 0:
        img = finalize_img(images[0])
        overlap = overlap.cpu()
        return img, overlap

    if len(component) == 1:
        img = finalize_img(images[component[0]])
        overlap = overlap.cpu()
        return img, overlap

    ref_idx = choose_ref_img(component, overlap)
    global_h = global_homographies(component, ref_idx, pairwise_h, device, dtype)
    connected_component = [idx for idx in component if idx in global_h]

    if len(connected_component) == 0:
        img = finalize_img(images[ref_idx])
        overlap = overlap.cpu()
        return img, overlap

    filtered_images = [images[i] for i in range(n)]
    t, out_h, out_w = panorama_canvas(filtered_images, global_h)
    warped_imgs = []
    warped_masks = []

    for idx in connected_component:
        h_canvas = t @ global_h[idx]
        warped_img, warped_mask = warp_img_mask(images[idx], h_canvas, out_h, out_w)
        warped_imgs.append(warped_img)
        warped_masks.append(warped_mask)

    pano = blend_panorama_avg(warped_imgs, warped_masks)
    img = finalize_img(pano)
    overlap = overlap.cpu()

    return img, overlap
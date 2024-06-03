import torch


def fftconv(u, k, D, is_causal=True):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    kernellen = k.shape[-1]

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")

    if not is_causal:
        y = y[..., kernellen // 2 : seqlen + kernellen // 2]
    else:
        y = y[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


def fftconv2d(u, k, D):
    original_dtype = u.dtype
    u = u.to(torch.float32)
    k = k.to(torch.float32)
    D = D.to(torch.float32)
    b, d, h, w = u.shape
    _, _, kh, kw = k.shape
    hh = max(2 * h, kh)
    ww = max(2 * w, kw)

    k_f = torch.fft.rfft2(k, s=(hh, ww)) / (hh * ww)
    u_f = torch.fft.rfft2(u, s=(hh, ww))

    y = torch.fft.irfft2(u_f * k_f, s=(hh, ww), norm="forward")
    y = y[..., kh // 2 : h + kh // 2, kw // 2 : w + kw // 2]

    out = y + u * D
    return out.to(dtype=original_dtype)

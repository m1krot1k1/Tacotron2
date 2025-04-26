import numpy as np
from scipy.io.wavfile import read
import torch
import cv2
import math


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len) if torch.cuda.is_available() else torch.LongTensor(max_len) )
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def get_drop_frame_mask_from_lengths(lengths, drop_frame_rate):
    batch_size = lengths.size(0)
    max_len = torch.max(lengths).item()
    mask = get_mask_from_lengths(lengths).float()
    drop_mask = torch.empty([batch_size, max_len], device=lengths.device).uniform_(0., 1.) < drop_frame_rate
    drop_mask = drop_mask.float() * mask
    return drop_mask


def dropout_frame(mels, global_mean, mel_lengths, drop_frame_rate):
    drop_mask = get_drop_frame_mask_from_lengths(mel_lengths, drop_frame_rate)
    dropped_mels = (mels * (1.0 - drop_mask).unsqueeze(1) +
                    global_mean[None, :, None] * drop_mask.unsqueeze(1))
    return dropped_mels

def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

def guide_attention_slow(text_lengths, mel_lengths, max_txt=None, max_mel=None):
    b = len(text_lengths)
    if max_txt is None:
        max_txt= np.max(text_lengths)
    if max_mel is None:
        max_mel = np.max(mel_lengths)
    guide = np.ones((b, max_txt, max_mel), dtype=np.float32)
    mask = np.zeros((b, max_txt, max_mel), dtype=np.float32)
    for i in range(b):
        W = guide[i]
        M = mask[i]
        N = float(text_lengths[i])
        T = float(mel_lengths[i])
        for n in range(max_txt):
            for t in range(max_mel):
                if n < N and t < T:
                    W[n][t] = 1.0 - np.exp(-(float(n) / N - float(t) / T) ** 2 / (2.0 * (0.2 ** 2)))
                    M[n][t] = 1.0
                elif t >= T and n < N:
                    W[n][t] = 1.0 - np.exp(-((float(n - N - 1) / N)** 2 / (2.0 * (0.2 ** 2))))
    if len(guide) == 1:
        cv2.imwrite('messigray2.png',(guide[0]*255).astype(np.uint8))
        return guide[0], mask[0]
    return guide, mask

def rotate_image(image, angle, center=(0,25)):
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def diagonal_guide(text_len, mel_len, g=0.2):
    grid_text = torch.linspace(0., 1. - 1. / text_len, text_len)  # (T)
    grid_mel = torch.linspace(0., 1. - 1. / mel_len, mel_len)  # (M)
    grid = grid_text.view(1, -1) - grid_mel.view(-1, 1)  # (M, T)
    W = 1 - torch.exp(-grid ** 2 / (2 * g ** 2))
    return W.numpy()

def linear_guide(text_len, mel_len, g=0.2):
    a = np.linspace(-1., -1./text_len, text_len)  # (T)
    W = 1 - np.exp(-a ** 2 / (2 * g ** 2))
    return W

# Измененная guide_attention_fast: теперь принимает только актуальные длины сэмпла
# Возвращает маску формы (txt_len, mel_len)
def guide_attention_fast(txt_len, mel_len, g=0.20): # <-- УДАЛЕНЫ max_txt, max_mel из параметров
    # Инициализируем маску с актуальными размерами сэмпла
    mask = np.ones((txt_len, mel_len), dtype=np.float32) # <-- ИСПРАВЛЕНО

    # Рассчитываем значения диагональной маски для фактических длин сэмпла
    # diag имеет форму (mel_len, txt_len)
    diag = diagonal_guide(txt_len, mel_len, g=g)

    # Транспонируем diag
    transposed_diag = np.transpose(diag, (1, 0))

    # Присваиваем транспонированную диагональ всей маске. Размеры уже совпадают.
    mask[:, :] = transposed_diag # <-- ИСПРАВЛЕНО. Обрезка больше не нужна здесь.
    # Часть с линейным гидом, если она должна применяться только к актуальной части сэмпла
    # Предполагаем, что linear_guide применяется к части маски в пределах актуальных длин
    # Если эта часть предназначена для паддинга, ее нужно удалить или пересмотреть.
    # Оставляем ее как есть, предполагая, что она должна работать с актуальными размерами.
    # linear = linear_guide(txt_len, mel_len).reshape(-1, 1) # форма (txt_len, 1)
    # Если linear_guide должен влиять на всю маску формы (txt_len, mel_len), то
    # mask[:txt_len, mel_len:] = np.repeat(linear, mel_len - mel_len, axis=-1) - эта строка из оригинала всегда пустая mel_len:mel_len
    # Это подозрительно. Возможно, линейный гид должен применяться иначе, или он не нужен в этой версии.
    # Для устранения ошибки Runtime, пока оставляем только диагональную часть как основную.
    # Если линейный гид действительно нужен, его логику нужно прояснить.
    # Комментируем строку с linear_guide пока что, как потенциальный источник будущих проблем.
    # linear = linear_guide(txt_len,mel_len).reshape(-1,1)
    # mask[:txt_len,mel_len:] = linear.repeat(mel_len-mel_len,axis=-1) # Эта строка не имеет эффекта


    return mask


# res = guide_attention_fast(150,700) # <-- Изменен вызов для теста
# cv2.imwrite('test.png', (res*255).astype(np.uint8))


import pickle
import torch
import numpy as np
import torch.utils.data as Data

def Load_Dataset_SAG(logger):
    file_pointer = './data/RML2016.10a_dict.pkl'
    classes = {b'QAM16': 0, b'QAM64': 1, b'8PSK': 2, b'WBFM': 3, b'BPSK': 4,
               b'CPFSK': 5, b'AM-DSB': 6, b'GFSK': 7, b'PAM4': 8, b'QPSK': 9, b'AM-SSB': 10}

    Set = pickle.load(open(file_pointer, 'rb'), encoding='bytes')
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Set.keys())))), [1, 0])

    Signals, Labels, SNRs = [], [], []
    for mod in mods:
        for snr in snrs:
            d = Set[(mod, snr)]
            Signals.append(d)
            for i in range(d.shape[0]):
                Labels.append(classes[mod])
                SNRs.append(snr)

    Signals  = torch.from_numpy(np.vstack(Signals).astype(np.float32))
    Labels   = torch.tensor(np.array(Labels, dtype=np.int64))
    SNRs_norm = torch.tensor(
        [(s - (-20)) / (18 - (-20)) for s in SNRs], dtype=torch.float32)

    logger.info(f'Signals: {list(Signals.shape)}')
    logger.info(f'Labels:  {list(Labels.shape)}')

    return Signals, Labels, SNRs_norm, snrs, mods


def Dataset_Split_SAG(Signals, Labels, SNRs, snrs, mods, logger):
    n = Signals.shape[0]
    n_train = int(n * 0.6)
    n_val   = int(n * 0.2)

    Slices = np.linspace(0, n, num=len(mods)*len(snrs)+1)
    train_idx, val_idx, test_idx = [], [], []

    per_class = len(mods) * len(snrs)
    for k in range(len(Slices)-1):
        start, end = int(Slices[k]), int(Slices[k+1])
        idx = np.arange(start, end)
        np.random.shuffle(idx)
        n_tr = int(n_train / per_class)
        n_vl = int(n_val   / per_class)
        train_idx.extend(idx[:n_tr])
        val_idx.extend(idx[n_tr:n_tr+n_vl])
        test_idx.extend(idx[n_tr+n_vl:])

    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)
    test_idx  = np.array(test_idx)

    logger.info(f'Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')

    return (Signals[train_idx], Labels[train_idx], SNRs[train_idx]), \
           (Signals[val_idx],   Labels[val_idx],   SNRs[val_idx]), \
           (Signals[test_idx],  Labels[test_idx],  SNRs[test_idx]), \
           test_idx


def Create_Data_Loader_SAG(train_set, val_set, cfg, logger):
    train_loader = Data.DataLoader(
        Data.TensorDataset(*train_set),
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=0)  # num_workers sabit 0
    val_loader = Data.DataLoader(
        Data.TensorDataset(*val_set),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=0)
    logger.info(f'train batches: {len(train_loader)}, val batches: {len(val_loader)}')
    return train_loader, val_loader

"""
Speech Classification Demo: spectrogram + hierarchical pooling.

Demonstrates cross-modal generality:
1. Audio -> 1D amplitude envelope (pure Python)
2. Amplitude -> 2D spectrogram-like tensor (time vs frequency bands)
3. 3-level hierarchical pooling + active cosine matching

Data loading separated from processing:
- load_audio(): file -> 1D sample array
- audio_to_spectrogram(): 1D -> 2D tensor
- GroupHierarchicalClassifier: tensor -> classification

Pure standard library only. No numpy/scipy required.
The principle primitives run on the installed cos_comparison engine
(core.pool, core.cos_comparison_passive, core.cos); audio I/O and the
spectrogram transform plus structural helpers remain local.
"""
import os
import sys
import wave
import struct

from cos_comparison import core


def cosine_similarity(a, b):
    """Whole-tensor cosine similarity via the formal engine."""
    return core.cos(a, b, algorithm=core._cos)


def down_sample(tensor, factor):
    """Average pooling via the formal engine: coarsen by factor."""
    h, w = len(tensor), len(tensor[0])
    nh, nw = h // factor, w // factor
    output = core.create_void_list((nh, nw))
    core.pool(tensor, window_size=(factor, factor),
              step=(factor, factor), output=output)
    return output


def passive_extract(tensor):
    """Passive d-shift boundary map via the formal engine (gap space)."""
    h, w = len(tensor), len(tensor[0])
    output = core.create_void_list((h - 1, w))
    core.cos_comparison_passive(tensor, window_size=(1, 1),
                                d=(1, 0), output=output)
    return output


def active_extract(tensor):
    return flatten_2d(tensor)


def flatten_2d(tensor):
    return [x for row in tensor for x in row]


class _DefaultInMemoryBackend:
    def store(self, key, value):
        return True

    def retrieve(self, key):
        return None


class GroupHierarchicalClassifier:
    """Group-theoretic hierarchical classifier (local, self-contained)."""

    def __init__(self, pool_factors=None, use_passive=True,
                 similarity_func=None, memory_backend=None):
        self.pool_factors = pool_factors or [1, 2, 4]
        self.use_passive = use_passive
        self.similarity_func = similarity_func or cosine_similarity
        self.memory_backend = memory_backend or _DefaultInMemoryBackend()
        self.prototypes = {}

    def _extract_levels(self, tensor):
        active_levels, passive_levels = [], []
        current = tensor
        for factor in self.pool_factors:
            if factor > 1:
                current = down_sample(current, factor)
            active_levels.append(active_extract(current))
            if self.use_passive:
                passive_levels.append(flatten_2d(passive_extract(current)))
        return active_levels, passive_levels

    def fit(self, images, labels):
        self.prototypes = {}
        for img, label in zip(images, labels):
            active_levels, passive_levels = self._extract_levels(img)
            if label not in self.prototypes:
                self.prototypes[label] = []
            entry = {'active': active_levels}
            if self.use_passive:
                entry['passive'] = passive_levels
            self.prototypes[label].append(entry)
        self.memory_backend.store('prototypes', self.prototypes)
        return self

    def predict(self, image):
        q_active, q_passive = self._extract_levels(image)
        best_label, best_score = None, -1.0
        for label, proto_list in self.prototypes.items():
            for proto in proto_list:
                score = 0.0
                for lvl in range(len(self.pool_factors)):
                    sim = self.similarity_func(q_active[lvl], proto['active'][lvl])
                    if sim > score:
                        score = sim
                if self.use_passive and 'passive' in proto:
                    for lvl in range(len(self.pool_factors)):
                        sim = self.similarity_func(q_passive[lvl], proto['passive'][lvl])
                        if sim > score:
                            score = sim
                if score > best_score:
                    best_score, best_label = score, label
        return best_label

    def accuracy(self, images, labels):
        correct = sum(1 for img, lab in zip(images, labels)
                      if self.predict(img) == lab)
        return correct / len(labels) if labels else 0.0


def load_wav(path):
    """Load WAV file as 1D amplitude array (normalized -1..1)."""
    with wave.open(path, 'rb') as wf:
        nframes = wf.getnframes()
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(nframes)

    samples = []
    if sampwidth == 2:
        for i in range(0, len(raw), 2 * nchannels):
            val = struct.unpack('<h', raw[i:i+2])[0]
            samples.append(val / 32768.0)
    elif sampwidth == 1:
        for i in range(0, len(raw), nchannels):
            samples.append((raw[i] - 128) / 128.0)
    else:
        # Fallback: treat as float
        for i in range(0, len(raw), 4):
            samples.append(struct.unpack('<f', raw[i:i+4])[0])

    return samples


def audio_to_spectrogram(samples, n_bands=8, n_time=28):
    """
    Convert 1D audio to 2D spectrogram-like tensor.
    Simple band-pass filtering via amplitude in sub-windows.

    Returns: (n_bands, n_time) 2D tensor.
    """
    n = len(samples)
    if n == 0:
        return [[0.0] * n_time for _ in range(n_bands)]

    # Divide into time windows
    win_size = max(1, n // n_time)
    result = [[0.0] * n_time for _ in range(n_bands)]

    for t in range(n_time):
        start = t * win_size
        end = min(start + win_size, n)
        if start >= end:
            continue
        window = samples[start:end]
        wn = len(window)

        # Simple frequency bands via short-time variance
        # Low band: first quarter, Mid: half, High: last quarter
        quarter = wn // 4
        low = sum(abs(x) for x in window[:quarter]) / max(1, quarter)
        mid = sum(abs(x) for x in window[quarter:3*quarter]) / max(1, 2*quarter)
        high = sum(abs(x) for x in window[3*quarter:]) / max(1, wn - 3*quarter)

        # Fill bands with energy distribution
        for b in range(n_bands):
            frac = b / n_bands
            if frac < 0.3:
                result[b][t] = low * (1.0 - frac * 3)
            elif frac < 0.7:
                result[b][t] = mid * (1.0 - abs(frac - 0.5) * 2)
            else:
                result[b][t] = high * (frac - 0.7) / 0.3

    return result


def load_speech_dataset(data_dir, n_per_class=20):
    """
    Load speech dataset from directory:
        data_dir/
            word1/
                audio1.wav
                ...
            word2/
                ...
    """
    classes = sorted(d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)))
    print(f"Found {len(classes)} classes")

    train_spects, train_labels = [], []
    test_spects, test_labels = [], []

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        files = sorted(f for f in os.listdir(cls_dir) if f.endswith('.wav'))[:n_per_class]
        n_test = max(1, len(files) // 3)
        for f in files[n_test:]:
            samples = load_wav(os.path.join(cls_dir, f))
            spect = audio_to_spectrogram(samples)
            train_spects.append(spect)
            train_labels.append(cls)
        for f in files[:n_test]:
            samples = load_wav(os.path.join(cls_dir, f))
            spect = audio_to_spectrogram(samples)
            test_spects.append(spect)
            test_labels.append(cls)

    print(f"Train: {len(train_spects)}, Test: {len(test_spects)}")
    return train_spects, train_labels, test_spects, test_labels


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo_speech.py <data_dir>")
        print("\nRunning synthetic demo...")
        import random
        random.seed(42)

        def make_word_spect(word, n_time=28, n_bands=8):
            """Create synthetic spectrogram-like patterns with time variation.
            word is the integer class id (0/1/2), matching the call site."""
            spect = [[0.0] * n_time for _ in range(n_bands)]
            # Time envelope: attack, sustain, decay (like real speech)
            envelope = []
            for t in range(n_time):
                if t < 5:  # attack
                    env = t / 5.0
                elif t < 15:  # sustain
                    env = 1.0
                else:  # decay
                    env = max(0, 1.0 - (t - 15) / 13.0)
                envelope.append(env)

            if word == 0:
                # Low frequency stationary vowel
                for t in range(n_time):
                    for b in range(n_bands):
                        spect[b][t] = envelope[t] * 0.5 * (1.0 - abs(b - 2) / n_bands)
            elif word == 1:
                # Mid frequency with rising sweep
                for t in range(n_time):
                    center = 3 + t / n_time * 2  # shifts from b=3 to b=5
                    for b in range(n_bands):
                        spect[b][t] = envelope[t] * 0.5 * (1.0 - abs(b - center) / n_bands)
            else:
                # High frequency with falling sweep
                for t in range(n_time):
                    center = 7 - t / n_time * 3  # shifts from b=7 to b=4
                    for b in range(n_bands):
                        spect[b][t] = envelope[t] * 0.5 * (1.0 - abs(b - center) / n_bands)
            # Add noise
            for b in range(n_bands):
                for t in range(n_time):
                    spect[b][t] += random.gauss(0, 0.02)
            return spect

        train_spects = [make_word_spect(i % 3) for i in range(120)]
        train_labels = [f"word_{i % 3}" for i in range(120)]
        test_spects = [make_word_spect(i % 3) for i in range(36)]
        test_labels = [f"word_{i % 3}" for i in range(36)]
    else:
        data_dir = sys.argv[1]
        train_spects, train_labels, test_spects, test_labels = load_speech_dataset(data_dir)

    print("\nTraining speech classifier...")
    # Note: spectrogram is (n_bands=8, n_time=28), so pool factors are mild
    # to avoid over-compressing the frequency dimension
    clf = GroupHierarchicalClassifier(
        pool_factors=[1, 2],  # Only 2 levels: full res + 2x pool
        use_passive=True  # Passive mode: local difference edges (gap space)
    )
    clf.fit(train_spects, train_labels)

    print("Evaluating...")
    acc = clf.accuracy(test_spects, test_labels)
    print(f"\nAccuracy: {acc * 100:.1f}%")
    print(f"Principle: spectrogram + 3-level pooling + active(element space) + passive(gap space) + K=1 max fusion")


if __name__ == '__main__':
    main()

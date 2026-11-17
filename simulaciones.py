"""Cálculo Monte Carlo de SLE por lotes en CUDA."""

import csv
import ctypes
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent

os.environ.setdefault('CUPY_CACHE_DIR', str(ROOT / '.cupy_cache'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.matplotlib'))
import cupy as cp
import numpy as np
from shapely.geometry import LineString

NS = (250, 500, 1000, 2000, 5000, 10000)
MS = (1, 10, 100, 1000, 10000, 100000)
FINE_N = 10000



def configuraciones_principales():
    constantes = [('constant', float(k), 0, 0, n)
                  for k in (1, 2, 3, 3.9, 4, 5, 6) for n in NS]
    variables = [('variable', float(k), f, 0, 5000)
                 for k in range(1, 7) for f in (1, 5, 10)]
    return sorted(constantes + variables, key=lambda c: (c[4], c))


def brownian_batch(start, size):
    return np.stack([np.random.default_rng(seed).standard_normal(FINE_N)
                     for seed in range(start, start + size)]) / np.sqrt(FINE_N)


def driving_increments(brownian, config):
    kind, k, frequency, phase, n = config
    if kind == 'constant':
        weights = np.sqrt(k)
    else:
        t = np.arange(FINE_N) / FINE_N
        weights = np.sqrt(np.maximum(0, k * (1 + np.cos(
            2 * np.pi * frequency * t + phase * np.pi / 2)) / 2))
    weighted = brownian * weights
    return weighted.reshape(len(brownian), n, FINE_N // n).sum(axis=2)


KERNEL = cp.RawKernel(r'''
#include <cupy/complex.cuh>
extern "C" __global__ void curves(const double* increments, int n,
                                  int batch, complex<double>* result) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= n * batch) return;
    int row = index / n;
    int point_index = index % n;
    complex<double> point(0.0, 0.0);
    for (int j = point_index; j >= 0; --j) {
        point = sqrt(point * point - 4.0 / n);
        if (point.imag() < 0.0) point = -point;
        point += increments[row * n + j];
    }
    result[row * (n + 1) + point_index + 1] = point;
}
''', 'curves')


def curves_cuda(increments):
    batch, n = increments.shape
    device_input = cp.asarray(increments)
    result = cp.zeros((batch, n + 1), dtype=cp.complex128)
    KERNEL(((batch * n + 127) // 128,), (128,),
           (device_input, np.int32(n), np.int32(batch), result))
    return result.get()


def curve_statistics(curve):
    simple = LineString(np.column_stack((curve.real, curve.imag))).is_simple
    return int(simple), float(curve.imag.max()), float(np.ptp(curve.real)), float(curve[-1].real), float(curve[-1].imag)


CHECKPOINT_COLUMNS = ('seed', 'simple', 'height', 'width', 'x', 'y', 'seconds')


class Progress:
    def __init__(self, path):
        self.path = path
        self.count = 0
        self.totals = (0, 0.0, 0.0, 0.0, 0.0)
        self.prefixes = {}
        if not path.exists():
            return
        with path.open('r+b') as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b'\n'):
                    handle.truncate(offset)
                    break
                values = line.decode('ascii').strip().split(',')
                if offset == 0:
                    if tuple(values) != CHECKPOINT_COLUMNS:
                        raise ValueError(f'Cabecera inesperada en {path}')
                    continue
                if len(values) != len(CHECKPOINT_COLUMNS) or int(values[0]) != self.count:
                    raise ValueError(f'Semilla inesperada en {path}: se esperaba {self.count}')
                self.record((int(values[1]), *map(float, values[2:6])))

    def record(self, stats):
        self.count += 1
        self.totals = tuple(total + value for total, value in zip(self.totals, stats))
        if self.count in MS:
            self.prefixes[self.count] = self.totals

    def append(self, stats, seconds):
        with self.path.open('a', encoding='ascii', newline='') as handle:
            writer = csv.writer(handle, lineterminator='\n')
            if handle.tell() == 0:
                writer.writerow(CHECKPOINT_COLUMNS)
            writer.writerows((seed, *row, seconds)
                             for seed, row in enumerate(stats, start=self.count))
        for row in stats:
            self.record(row)


def export_csv(progress, destination):
    """Export completed sample prefixes and replace the summary atomically."""
    temporary = destination.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['kind', 'kappa_or_Kmax', 'f', 'phase_pi_over_2', 'N', 'T', 'M',
                         'simple_count', 'proportion', 'wilson_low', 'wilson_high',
                         'mean_height', 'mean_width', 'mean_final_real', 'mean_final_imag'])
        for config in sorted(progress):
            for m, totals in progress[config].prefixes.items():
                simple, *geometry = totals
                p = simple / m
                z = 1.959963984540054
                denominator = 1 + z*z/m
                center = (p + z*z/(2*m))/denominator
                half = z*np.sqrt(p*(1-p)/m + z*z/(4*m*m))/denominator
                writer.writerow([*config, 1, m, simple, p, center-half, center+half,
                                 *(total / m for total in geometry)])
    os.replace(temporary, destination)


def run(output=ROOT / 'resultados' / 'recalculo', max_m=100_000, batch_size=16, configs=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Windows releases this lock automatically if the process exits or crashes.
    import msvcrt
    lock = (output / 'run.lock').open('a+b')
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close()
        raise RuntimeError('Ya hay una simulacion activa en esta carpeta.')
    if configs is None:
        configs = configuraciones_principales()
    started = time.time()
    # Keep the computer awake while this process runs, allowing the display to sleep.
    awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    print(json.dumps({'pid': os.getpid(), 'configurations': len(configs), 'max_m': max_m,
                      'batch_size': batch_size, 'sleep_prevention': bool(awake)}), flush=True)
    try:
        checkpoints = output / 'trayectorias'
        checkpoints.mkdir(exist_ok=True)
        # Keep summaries for configurations saved by earlier runs, including extras.
        progress = {}
        for path in checkpoints.glob('*.csv'):
            kind, k, f, phase, n = path.stem.split('_')
            config = (kind, float(k), int(f), int(phase), int(n))
            progress[config] = Progress(path)
        for config in configs:
            if config not in progress:
                filename = '_'.join(map(str, config)) + '.csv'
                progress[config] = Progress(checkpoints / filename)

        def save_summary():
            export_csv(progress, output / 'resumen.csv')

        first_seed = min(progress[config].count for config in configs)
        for start in range(first_seed, max_m, batch_size):
            end = min(start + batch_size, max_m)
            brownian = brownian_batch(start, end - start)
            for config in configs:
                saved = progress[config]
                if saved.count >= end:
                    continue
                tick = time.perf_counter()
                curves = curves_cuda(driving_increments(brownian[saved.count - start:], config))
                stats = [curve_statistics(curve) for curve in curves]
                saved.append(stats, (time.perf_counter() - tick) / len(stats))
            if start // 1000 != end // 1000 or any(start < m <= end for m in MS):
                save_summary()
            status = {'completed_per_configuration': end, 'target': max_m,
                      'configurations': len(configs), 'elapsed_seconds_this_run': time.time()-started,
                      'updated': time.strftime('%Y-%m-%d %H:%M:%S'), 'pid': os.getpid()}
            temp = output / 'estado.tmp'
            temp.write_text(json.dumps(status, indent=2), encoding='utf-8')
            os.replace(temp, output / 'estado.json')
            print(json.dumps(status), flush=True)
            if (output / 'STOP').exists():
                save_summary()
                print('Detenido en un punto de control. Retira STOP para reanudar.', flush=True)
                return
        save_summary()
        print('SIMULACIONES COMPLETADAS', flush=True)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


def configuraciones_con_extras():
    configs = set()
    for k in (1, 2, 3, 3.5, 3.8, 3.9, 4, 4.1, 4.2, 4.5, 5, 6):
        for n in NS:
            configs.add(('constant', float(k), 0, 0, n))
    for k in (.5, 1, 1.5, 2, 2.5, 3):
        configs.add(('constant', float(k), 0, 0, 5000))
    for k in range(1, 7):
        for f in (1, 2, 5, 10, 20, 50):
            configs.add(('variable', float(k), f, 0, 5000))
        for f in (1, 5, 10):
            for phase in (1, 2, 3):
                configs.add(('variable', float(k), f, phase, 5000))
    return sorted(configs, key=lambda c: (c[4], c))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Simulaciones Monte Carlo de SLE con CUDA')
    parser.add_argument('--max-m', type=int, default=100_000, choices=MS,
                        help='Trayectorias por combinacion')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', type=Path, default=ROOT / 'resultados' / 'recalculo')
    parser.add_argument('--extras', action='store_true',
                        help='Incluye los experimentos 6 a 10')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size debe ser positivo')
    configs = configuraciones_con_extras() if args.extras else configuraciones_principales()
    run(output=args.output, max_m=args.max_m, batch_size=args.batch_size, configs=configs)

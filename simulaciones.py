"""Cálculo Monte Carlo de SLE por lotes en CUDA."""

import csv
import ctypes
import json
import os
from pathlib import Path
import sqlite3
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


def export_csv(connection, destination):
    """Only export complete sample prefixes. Replace the CSV atomically."""
    temporary = destination.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['kind', 'kappa_or_Kmax', 'f', 'phase_pi_over_2', 'N', 'T', 'M',
                         'simple_count', 'proportion', 'wilson_low', 'wilson_high',
                         'mean_height', 'mean_width', 'mean_final_real', 'mean_final_imag'])
        for key, count in connection.execute('SELECT config, COUNT(*) FROM samples GROUP BY config'):
            for m in MS:
                if m > count:
                    break
                row = connection.execute('SELECT COUNT(*), SUM(simple), AVG(height), AVG(width), AVG(x), AVG(y) FROM samples WHERE config=? AND seed<?', (key, m)).fetchone()
                if row[0] != m:
                    continue
                p = row[1] / m
                z = 1.959963984540054
                denominator = 1 + z*z/m
                center = (p + z*z/(2*m))/denominator
                half = z*np.sqrt(p*(1-p)/m + z*z/(4*m*m))/denominator
                writer.writerow([*json.loads(key), 1, m, row[1], p, center-half, center+half, *row[2:]])
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
    connection = sqlite3.connect(output / 'trayectorias.sqlite')
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('CREATE TABLE IF NOT EXISTS samples (config TEXT, seed INTEGER, simple INTEGER, height REAL, width REAL, x REAL, y REAL, seconds REAL, PRIMARY KEY(config,seed))')
    if configs is None:
        configs = configuraciones_principales()
    started = time.time()
    # Keep the computer awake while this process runs, allowing the display to sleep.
    awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    print(json.dumps({'pid': os.getpid(), 'configurations': len(configs), 'max_m': max_m,
                      'batch_size': batch_size, 'sleep_prevention': bool(awake)}), flush=True)
    try:
        for start in range(0, max_m, batch_size):
            size = min(batch_size, max_m-start)
            brownian = brownian_batch(start, size)
            for config in configs:
                key = json.dumps(config)
                existing = {row[0] for row in connection.execute('SELECT seed FROM samples WHERE config=? AND seed>=? AND seed<?', (key, start, start+size))}
                missing = [i for i in range(size) if start+i not in existing]
                if not missing:
                    continue
                tick = time.perf_counter()
                curves = curves_cuda(driving_increments(brownian[missing], config))
                stats = [curve_statistics(curve) for curve in curves]
                elapsed = time.perf_counter()-tick
                with connection:
                    connection.executemany('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?)',
                        [(key, start+i, *stat, elapsed/len(missing)) for i, stat in zip(missing, stats)])
            if (start // 1000 != (start + size) // 1000 or
                    any(start < m <= start + size for m in MS) or start + size == max_m):
                export_csv(connection, output / 'resumen.csv')
            status = {'completed_per_configuration': start+size, 'target': max_m,
                      'configurations': len(configs), 'elapsed_seconds_this_run': time.time()-started,
                      'updated': time.strftime('%Y-%m-%d %H:%M:%S'), 'pid': os.getpid()}
            temp = output / 'estado.tmp'
            temp.write_text(json.dumps(status, indent=2), encoding='utf-8')
            os.replace(temp, output / 'estado.json')
            print(json.dumps(status), flush=True)
            if (output / 'STOP').exists():
                export_csv(connection, output / 'resumen.csv')
                print('Detenido en un punto de control. Retira STOP para reanudar.', flush=True)
                return
        print('SIMULACIONES COMPLETADAS', flush=True)
    finally:
        connection.close()
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

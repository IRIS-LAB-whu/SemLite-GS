import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np

DEFAULT_RUNTIME = dict(max_seconds=150., stagnation_seconds=30.,
                       stagnation_callbacks=15, improvement_atol=1e-8,
                       improvement_rtol=1e-6)


class FitStopped(Exception):
    pass


class ProgressGuard:
    def __init__(self, policy, clock=time.monotonic):
        self.policy = policy
        self.clock = clock
        self.started = self.last_progress = clock()
        self.reference_value = 0.
        self.calls_without_progress = 0
        self.callbacks = 0

    def check(self, best_value):
        now = self.clock()
        self.callbacks += 1
        threshold = max(self.policy['improvement_atol'],
                        self.policy['improvement_rtol'] * abs(self.reference_value))
        if self.reference_value - best_value > threshold:
            self.reference_value = best_value
            self.last_progress = now
            self.calls_without_progress = 0
        else:
            self.calls_without_progress += 1
        if now - self.started >= self.policy['max_seconds']:
            raise FitStopped('timeout')
        if (self.calls_without_progress >= self.policy['stagnation_callbacks'] or
                now - self.last_progress >= self.policy['stagnation_seconds']):
            raise FitStopped('stagnation')


def save_checkpoint(path, vertices, report):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        np.savez(stream, vertices=vertices, report=np.array(json.dumps(report)))
    temp.replace(path)


def read_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        return data['vertices'].copy(), json.loads(str(data['report']))


def wait_worker(process, seconds):

    try:
        process.wait(timeout=max(.001, seconds))
        return False
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2.)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return True


def guarded_refit(vertices, faces, centers, normals, area, checkpoint_path=None,
                  runtime=None, sdf_context=None):
    policy = dict(DEFAULT_RUNTIME)
    if runtime is not None:
        if set(runtime) - set(policy):
            raise ValueError('Unknown fit runtime options')
        policy.update(runtime)
    for key, value in policy.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Fit runtime options must be finite and positive: ' + key)
    started = time.monotonic()
    from .geometry import audit
    from .plane_fit import PARAMETERS
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if not audit(vertices, faces)['valid']:
        raise ValueError('Invalid pre-fit topology')
    initial = dict(state='complete', parameters=PARAMETERS,
                   plane_fit_records=[], solver_converged=False,
                   solver_message='Retained validated pre-fit mesh',
                   stop_reason='not_started', retained_candidate='initial',
                   max_vertex_shift_m=0., max_plane_displacement_residual_m=0.,
                   accepted_step=0., solver_iterations=0, seconds=0.)
    if sdf_context is not None:
        initial['joint_sdf'] = dict(enabled=True, status='not_started',
                                   field=sdf_context['field_path'])
    with tempfile.TemporaryDirectory(prefix='semlite_plane_fit_') as tmp:
        directory = Path(tmp)
        checkpoint = (Path(checkpoint_path).resolve() if checkpoint_path is not None
                      else directory / 'best.npz')
        save_checkpoint(checkpoint, vertices, initial)
        for name, array in [('vertices', vertices), ('faces', faces), ('centers', centers),
                            ('normals', normals), ('area', area)]:
            np.save(directory / (name + '.npy'), array, allow_pickle=False)
        (directory / 'config.json').write_text(json.dumps(dict(
            runtime=policy, checkpoint=str(checkpoint), sdf_context=sdf_context)))
        env = os.environ.copy()
        package_root = str(Path(__file__).resolve().parents[2])
        env['PYTHONPATH'] = package_root + os.pathsep + env.get('PYTHONPATH', '')
        with (directory / 'worker.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-m',
                'semlite_gs.reconstruction.fit_runtime', str(directory)],
                env=env, cwd=package_root, stdout=log, stderr=subprocess.STDOUT)
            try:
                timed_out = wait_worker(process, policy['max_seconds'] - (time.monotonic() - started))
            except BaseException:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                raise
        if process.returncode != 0 and not timed_out:
            raise RuntimeError('Original plane-fit worker failed:\n' +
                               (directory / 'worker.log').read_text()[-8000:])
        fitted, report = read_checkpoint(checkpoint)
        if fitted.shape != vertices.shape or not np.isfinite(fitted).all():
            raise ValueError('Invalid checkpoint vertex array')
        if not audit(fitted, faces)['valid']:
            raise ValueError('Invalid checkpoint topology')
        old = vertices[faces];new = fitted[faces]
        old_cross = np.cross(old[:, 1]-old[:, 0], old[:, 2]-old[:, 0])
        new_cross = np.cross(new[:, 1]-new[:, 0], new[:, 2]-new[:, 0])
        if not np.all(np.sum(old_cross*new_cross, axis=1) > 0):
            raise ValueError('Checkpoint flips an original face')
        if timed_out:
            report.update(solver_converged=False, stop_reason='timeout',
                solver_message='Time budget exceeded; retained best saved feasible fit')
        report['runtime'] = dict(policy, hard_timeout=timed_out,
            worker_seconds=report.get('seconds'),
            scope='Whole joint-fit stage, including input staging and worker startup; cleanup and final validation may add overhead',
            solver='SLSQP', checkpoint=str(checkpoint) if checkpoint_path is not None else None)
        report['seconds'] = time.monotonic() - started
        report['state'] = 'complete'                                               
        if checkpoint_path is not None:
            save_checkpoint(checkpoint, fitted, report)
        return fitted, report


def main():
    from .plane_fit import _solve_joint_planes
    directory = Path(sys.argv[1])
    config = json.loads((directory / 'config.json').read_text())
    arrays = [np.load(directory / (name + '.npy'), mmap_mode='r', allow_pickle=False)
              for name in ['vertices', 'faces', 'centers', 'normals', 'area']]
    checkpoint = lambda vertices, report: save_checkpoint(config['checkpoint'], vertices, report)
    vertices, report = _solve_joint_planes(*arrays, _checkpoint=checkpoint,
                                            _runtime=config['runtime'],
                                            _sdf=config.get('sdf_context'))
    checkpoint(vertices, report)


if __name__ == '__main__':
    main()

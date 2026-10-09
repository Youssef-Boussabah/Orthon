"""Deterministic measurement contracts and source ownership; no speed gates."""
import ast
import json
import os
from dataclasses import asdict, replace
import pytest
import orthon
from tests.support.source import REPO_ROOT, read_cpp_source
from benchmarks._support import timing
from benchmarks._support.environment import temporary_environment, command
from benchmarks._support.results import document, write_json
from tests.support.compatibility import RETIRED_BINDINGS


def test_quartiles_interpolate_and_statistics_report_median_iqr():
    assert timing.quartiles([4., 1., 3., 2.]) == (1.75, 2.5, 3.25)
    assert timing.quartiles([7.]) == (7., 7., 7.)
    stats = timing.Statistics((4., 1., 3., 2.), 8).as_dict()
    assert stats['median'] == 2.5 and stats['iqr'] == 1.5
    assert stats['sample_count'] == 4 and stats['inner_iterations'] == 8


@pytest.mark.parametrize('values', [[], [float('nan')], [float('inf')], [-1.]])
def test_statistics_reject_invalid_samples(values):
    with pytest.raises(ValueError):
        timing.quartiles(values)


def test_adaptive_inner_shares_calls_and_duration_cap_wins():
    cfg = timing.TimingConfig(
        min_inner=8, max_inner=100, target_ms=40, max_sample_ms=50
    )
    assert timing.choose_inner(1, 2, cfg) == 25
    assert timing.choose_inner(100, 100, cfg) == 1
    assert timing.choose_inner(.01, .02, cfg) == 100
    assert timing.choose_inner(1, 100, timing.TimingConfig(fixed_inner=13)) == 13


def test_pair_alternates_order_using_identical_inner(monkeypatch):
    log = []
    monkeypatch.setattr(timing, 'estimate_ms', lambda fn, sync: 1)
    monkeypatch.setattr(timing, 'batch_ms', lambda fn, n, sync: (fn(), float(n))[1])
    a, b = timing.measure_pair(
        lambda: log.append('a'), lambda: log.append('b'), lambda: None, lambda: None,
        timing.TimingConfig(warmup=0, samples=3, fixed_inner=4),
    )
    assert log == ['a', 'b', 'b', 'a', 'a', 'b']
    assert a.inner == b.inner == 4 and a.samples == b.samples == (1., 1., 1.)


def test_completion_and_timer_order(monkeypatch):
    order = []
    monkeypatch.setattr(
        timing.time, 'perf_counter', lambda: (order.append('clock'), 1.)[1]
    )
    timing.batch_ms(lambda: order.append('op'), 3, lambda: order.append('sync'))
    assert order == ['sync', 'clock', 'op', 'op', 'op', 'sync', 'clock']


def test_correctness_gates_claim_and_overlapping_intervals():
    a = timing.Statistics((1., 2., 3.), 1)
    b = timing.Statistics((1., 2., 3.), 1)
    assert timing.comparison(a, b, True)['verdict'] == 'unreliable'
    invalid = timing.comparison(a, b, False)
    assert invalid['verdict'] == 'invalid' and invalid['speedup'] is None


def test_strict_json_and_quick_schema(tmp_path):
    from benchmarks._support.cli import timing_config
    from argparse import Namespace
    cfg = timing_config(Namespace(quick=True))
    payload = document(
        'fake', [], dict(representative=False, timing=asdict(cfg)), environment={}
    )
    path = tmp_path / 'result.json'
    write_json(path, payload)
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data['schema'] == 'orthon-benchmark' and data['schema_version'] == 1
    assert data['config']['representative'] is False
    with pytest.raises(ValueError):
        write_json(path, dict(payload, bad=float('nan')))
    assert json.loads(path.read_text()) == payload


def test_environment_restores_existing_missing_and_exceptions(monkeypatch):
    monkeypatch.setenv('ORTHON_CUDA_CONV2D_FORWARD', 'naive')
    monkeypatch.delenv('ORTHON_CUDA_CONV2D_GRAD_INPUT', raising=False)
    with pytest.raises(RuntimeError):
        with temporary_environment({
            'ORTHON_CUDA_CONV2D_FORWARD': None,
            'ORTHON_CUDA_CONV2D_GRAD_INPUT': 'optimized',
        }):
            assert 'ORTHON_CUDA_CONV2D_FORWARD' not in os.environ
            raise RuntimeError('fake')
    assert os.environ['ORTHON_CUDA_CONV2D_FORWARD'] == 'naive'
    assert 'ORTHON_CUDA_CONV2D_GRAD_INPUT' not in os.environ


def test_provenance_unavailable_commands_are_null():
    assert command(['/nonexistent/orthon-benchmark-tool']) is None


def test_one_timing_and_json_owner_and_no_path_mutation():
    for path in (REPO_ROOT / 'benchmarks').rglob('*.py'):
        tree = ast.parse(path.read_text())
        attributes = {
            ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Attribute)
        }
        assert not attributes & {'sys.path.insert', 'sys.path.append'}, path
        if '_support' not in path.parts:
            assert not attributes & {
                'time.perf_counter', 'torch.cuda.synchronize',
                'orthon.cuda_synchronize', 'ortho.cuda_synchronize',
                'json.dump', 'json.dumps',
            }, path
            defs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            assert not defs & {
                '_quartiles', '_choose_inner', '_estimate_ms', '_batch_ms', 'summarize_ms'
            }, path
            assert not any(
                isinstance(n, ast.Attribute)
                and n.attr.startswith(('_matmul_', '_memory_pool_', 'cuda_conv2d_profile_'))
                for n in ast.walk(tree)
            ), path
    assert not (REPO_ROOT / 'benchmarks/_bench_utils.py').exists()


def test_examples_have_no_pure_measurement_drivers(tmp_path):
    names = [p.name for p in (REPO_ROOT / 'examples').glob('*.py')]
    assert not [n for n in names if n.startswith(('benchmark_', 'profile_', 'report_', 'compare_mnist'))]
    assert {'cifar10_cnn.py', 'mnist_mlp.py', 'tiny_causal_lm.py'} <= set(names)
    import subprocess
    import sys
    command = [sys.executable, 'examples/cifar10_cnn.py', '--synthetic-data',
               '--no-download', '--epochs', '1', '--batch-size', '2',
               '--channels', '2', '--hidden', '4', '--seed', '7',
               '--max-train-batches', '1', '--max-test-batches', '1']
    rows = []
    for run in range(2):
        output = tmp_path / f'cifar-{run}.json'
        result = subprocess.run(command + ['--output', str(output)], cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(output.read_text())
        assert payload['config']['synthetic_data'] and payload['config']['no_download']
        row = payload['epochs'][0]
        assert row['train_samples'] == row['test_samples'] == 2
        assert row['train_loss'] > 0 and row['test_loss'] > 0
        row.pop('seconds')
        rows.append(row)
    assert rows[0] == rows[1]


def test_diagnostics_have_one_python_owner_and_retirements_are_fenced():
    import orthon._diagnostics as diag
    assert callable(diag.memory_pool_stats) and callable(orthon.cuda_synchronize)
    assert not any(n.startswith(('_memory_pool_', '_matmul_', 'cuda_conv2d_profile_')) for n in dir(orthon))
    assert not any(hasattr(orthon._C, name) for name in RETIRED_BINDINGS)
    for name in ['matmul_custom', 'matmul_cublas', 'matmul_dispatch_path', 'matmul_guard_report']:
        assert callable(getattr(diag, name)) if orthon.cuda_enabled() else getattr(diag, name) is None


def test_native_profiler_removed_and_private_selectors_retained():
    code = read_cpp_source('src/backends/cuda/conv2d_cuda.cu')
    assert 'ORTHON_PROFILE_CONV2D' not in code and 'Conv2DProfileStats' not in code
    assert 'conv2d_profile_' not in code and 'chrono' not in code
    for selector in ['FORWARD', 'GRAD_INPUT', 'GRAD_WEIGHT', 'BIAS_GRAD', 'GRAD_BIAS']:
        assert 'ORTHON_CUDA_CONV2D_' + selector in code
    matmul = read_cpp_source('src/backends/cuda/matmul_kernels.cu')
    assert 'CudaTimingEvent' not in matmul and 'cublas_elapsed_ms' not in matmul


def test_no_performance_gates_in_tests_or_ci():
    # Restrict to assertion/comparison syntax, preserving mathematical tolerances.
    for path in (REPO_ROOT / 'tests').rglob('*.py'):
        if path.name == 'test_benchmark_architecture.py':
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
                names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
                assert not names & {'latency', 'speedup', 'throughput', 'samples_per_sec', 'median_ms'}, path
    for path in (REPO_ROOT / '.github/workflows').glob('*'):
        assert 'python -m benchmarks.' not in path.read_text(), path


def test_quick_cpu_cli_emits_strict_schema_without_performance_assertions(tmp_path, monkeypatch):
    from benchmarks import compare_pytorch as driver
    from benchmarks._support import comparison_cases as cases

    monkeypatch.setattr(cases, 'ORTHON_DEVICE', 'cpu')
    monkeypatch.setattr(cases, 'TORCH_DEVICE', 'cpu')
    backward = cases.build_cross_entropy_backward((2, 3))
    assert backward.check()['ok']
    # The missing-value branch uses infinity when selecting the worst check.
    assert cases.math.isinf(cases.math.inf)
    order = []

    def identity_compile(fn):
        order.append('compile')

        def compiled():
            order.append('compiled-call')
            return fn()

        return compiled

    monkeypatch.setattr(cases.torch, 'compile', identity_compile)
    fused = cases.build_fused_linear_relu((2, 3, 4))
    compiled = fused.torch_compile_factory()
    assert order == ['compile', 'compiled-call']
    cases.torch.testing.assert_close(compiled(), fused.torch_fn())
    assert fused.check()['ok']
    order.clear()

    stats = timing.Statistics((1., 1., 1.), 2)
    monkeypatch.setattr(driver, 'measure_pair', lambda *args: (stats, stats))
    monkeypatch.setattr(driver, 'torch_sync', lambda: order.append('sync'))

    def measure_compiled(fn, sync, cfg):
        assert order == ['compile', 'compiled-call', 'sync']
        assert cfg.fixed_inner == stats.inner
        order.append('measurement')
        fn()
        return stats

    monkeypatch.setattr(driver, 'measure', measure_compiled)
    result = driver.run_case(fused, timing.TimingConfig(), want_compile=True)
    assert result.compile_error is None
    assert result.compiled_samples == list(stats.samples)

    def fail_compile():
        raise RuntimeError('compile setup failed')

    failed = driver.run_case(
        replace(fused, torch_compile_factory=fail_compile),
        timing.TimingConfig(), want_compile=True,
    )
    assert failed.compile_error == 'compile setup failed'
    assert failed.compiled_samples is None

    import subprocess
    import sys
    output = tmp_path / 'quick.json'
    result = subprocess.run([sys.executable, '-m', 'benchmarks.bench_elementwise', '--quick',
                           '--device', 'cpu', '--json', str(output)], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text())
    assert data['schema'] == 'orthon-benchmark' and data['schema_version'] == 1
    assert data['config']['representative'] is False
    assert all(c['correctness'] == 'pass' and c['sample_count'] == 3 for c in data['cases'])
    assert 'representative=false' in result.stdout


def _r4_controlled_comparison(monkeypatch, outcomes):
    from types import SimpleNamespace
    from benchmarks import compare_pytorch as driver

    def forbidden(*args, **kwargs):
        pytest.fail('check-only entered timing or wrote a performance artifact')

    for name in ['global_warmup', 'run_case', 'measure', 'measure_pair',
                 'write_json', 'write_markdown']:
        monkeypatch.setattr(driver, name, forbidden)
    monkeypatch.setattr(driver, 'collect_environment', lambda: {})
    monkeypatch.setattr(driver, 'print_environment', lambda env: None)
    for module in [driver, driver.comparison_cases]:
        for name in ['ORTHON_DEVICE', 'TORCH_DEVICE']:
            monkeypatch.setattr(module, name, getattr(module, name))
    names = [f'controlled_{i}' for i in range(len(outcomes))]
    built = []

    def build(shape, index):
        built.append(index)
        return SimpleNamespace(label='controlled', check=lambda: {
            'ok': outcomes[index], 'rel': 0. if outcomes[index] else 1., 'tol': 1e-3,
        })
    monkeypatch.setattr(driver, 'BENCHMARK_ORDER', names)
    monkeypatch.setattr(driver, 'BENCHMARKS', {
        name: {'shapes': [(1,)], 'build': lambda shape, i=i: build(shape, i)}
        for i, name in enumerate(names)
    })
    return driver, built


@pytest.mark.parametrize('outcomes,expected', [
    ([True, True], 0), ([False], 1), ([False, False], 1),
    ([True, False], 1), ([False, True], 1),
], ids=['all-pass', 'one-fail', 'multiple-fail', 'pass-fail', 'fail-pass'])
def test_r4_check_only_returns_all_checker_outcomes(monkeypatch, capsys, tmp_path, outcomes, expected):
    driver, built = _r4_controlled_comparison(monkeypatch, outcomes)
    output, markdown = tmp_path / 'results.json', tmp_path / 'results.md'
    status = driver.main(['--device', 'cpu', '--check-only', '--json', str(output),
                          '--markdown', str(markdown)])
    captured = capsys.readouterr()
    labels = [line.split()[-1] for line in captured.out.splitlines() if 'controlled_' in line]
    assert status == expected and labels == ['PASS' if ok else 'FAIL' for ok in outcomes]
    assert built == list(range(len(outcomes))) and not captured.err
    assert not output.exists() and not markdown.exists()


def test_r4_check_only_empty_selection_is_usage_error(monkeypatch, capsys):
    driver, built = _r4_controlled_comparison(monkeypatch, [])
    with pytest.raises(SystemExit) as error:
        driver.main(['--device', 'cpu', '--check-only'])
    captured = capsys.readouterr()
    assert error.value.code == 2 and 'no numerical checks selected' in captured.err
    assert 'PASS' not in captured.out and 'FAIL' not in captured.out and built == []


def test_r4_check_only_checker_exception_propagates(monkeypatch, capsys):
    driver, _ = _r4_controlled_comparison(monkeypatch, [True])

    def fail():
        raise RuntimeError('controlled checker exception')
    from types import SimpleNamespace
    driver.BENCHMARKS['controlled_0']['build'] = lambda shape: SimpleNamespace(label='controlled', check=fail)
    with pytest.raises(RuntimeError, match='controlled checker exception'):
        driver.main(['--device', 'cpu', '--check-only'])
    captured = capsys.readouterr()
    assert 'PASS' not in captured.out and 'FAIL' not in captured.out


@pytest.mark.parametrize('arguments,message', [
    (['--benchmark', 'unknown'], 'unknown benchmark'),
    (['--shape', '0'], 'no case matches --shape'),
], ids=['unknown-benchmark', 'invalid-shape'])
def test_r4_check_only_invalid_selection_retains_cli_error(monkeypatch, capsys, arguments, message):
    driver, built = _r4_controlled_comparison(monkeypatch, [True])
    with pytest.raises(SystemExit) as error:
        driver.main(['--device', 'cpu', '--check-only', *arguments])
    captured = capsys.readouterr()
    assert message in str(error.value.code) and built == []
    assert 'PASS' not in captured.out and 'FAIL' not in captured.out


def test_r4_list_remains_successful_without_checks(monkeypatch, capsys):
    driver, built = _r4_controlled_comparison(monkeypatch, [])
    assert driver.main(['--list', '--check-only']) == 0
    assert 'Available benchmarks' in capsys.readouterr().out and built == []


@pytest.mark.parametrize('ok,expected', [(True, 0), (False, 1)], ids=['pass', 'fail'])
def test_r4_check_only_child_entrypoint_propagates_exit(monkeypatch, tmp_path, ok, expected):
    import subprocess
    import sys
    code = f"""import sys, runpy
from types import SimpleNamespace
from benchmarks._support import comparison_cases as cases
cases.BENCHMARK_ORDER = ['controlled']
cases.BENCHMARKS = {{'controlled': {{'shapes': [(1,)], 'build': lambda shape: SimpleNamespace(label='controlled', check=lambda: {{'ok': {ok!r}, 'rel': 0., 'tol': 1e-3}})}}}}
sys.argv = ['benchmarks.compare_pytorch', '--device', 'cpu', '--check-only']
runpy.run_module('benchmarks.compare_pytorch', run_name='__main__')
"""
    result = subprocess.run([sys.executable, '-c', code], cwd=REPO_ROOT,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == expected, result.stdout + result.stderr
    assert ('PASS' if ok else 'FAIL') in result.stdout and 'measuring' not in result.stdout

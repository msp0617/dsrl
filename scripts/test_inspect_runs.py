import os
import tempfile

try:
    from scripts import inspect_runs
except ImportError:  # also allow ``python scripts/test_inspect_runs.py``
    import inspect_runs


def summarize(text):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        return inspect_runs.out_summary(path)
    finally:
        os.unlink(path)


def test_done_survives_a_long_unmarked_shutdown_epilogue():
    marker, error, done = summarize(
        "[eval] env_steps=100000\n"
        "[done] 100000 env steps, checkpoint in /tmp/checkpoint\n"
        + "worker shutdown\n" * 12
    )
    assert marker.startswith("[done] 100000")
    assert error is None
    assert done


def test_successful_retry_supersedes_an_old_traceback():
    marker, error, done = summarize(
        "Traceback (most recent call last):\n"
        "  frame\n"
        "RuntimeError: old failure\n"
        "[resume] slot a at 25000 env steps\n"
        "[done] 100000 env steps\n"
    )
    assert marker == "[done] 100000 env steps"
    assert error is None
    assert done


def test_new_attempt_after_done_is_not_mistaken_for_done():
    marker, error, done = summarize(
        "[done] 25000 env steps\n"
        "[resume] slot a at 25000 env steps\n"
        "[budget] env steps done 25000, target 100000\n"
    )
    assert marker.startswith("[budget]")
    assert error is None
    assert not done


def test_resume_fallback_exception_name_is_not_a_fatal_error():
    marker, error, done = summarize(
        "[resume] slot a did not load (EOFError: truncated), falling back\n"
        "[resume] slot b at 20000 env steps\n"
    )
    assert marker.startswith("[resume] slot b")
    assert error is None
    assert not done


def test_uncaught_exception_after_progress_is_active_error():
    marker, error, done = summarize(
        "[budget] env steps done 0, target 100000\n"
        "Traceback (most recent call last):\n"
        "  frame\n"
        "RuntimeError: pretrain_path holds 'td' weights but this run is variant=cql\n"
    )
    assert marker.startswith("RuntimeError: pretrain_path")
    assert error == marker
    assert not done


def test_find_experiments_ignores_tensorboard_container_dirs():
    with tempfile.TemporaryDirectory() as logs:
        os.makedirs(os.path.join(logs, "robomimic-dsrl", "run", "timestamp"))
        os.makedirs(os.path.join(logs, "real_run"))
        open(os.path.join(logs, "real_run", "eval_log.csv"), "w").close()
        open(os.path.join(logs, "output_only.out"), "w").close()
        open(os.path.join(logs, "pretrain_old.out"), "w").close()

        assert inspect_runs.find_experiments(logs) == ["output_only", "real_run"]


def test_csv_tail_ignores_a_partially_written_last_row():
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        f.write("env_steps,success_rate,reward\n100,0.5,-10\n200,0.6")
        path = f.name
    try:
        count, last = inspect_runs.csv_tail(path)
        assert count == 1
        assert last == {"env_steps": "100", "success_rate": "0.5", "reward": "-10"}
    finally:
        os.unlink(path)


def test_checkpoint_summary_survives_a_partial_but_valid_state():
    with tempfile.TemporaryDirectory() as ckpt:
        with open(os.path.join(ckpt, "run_state.json"), "w", encoding="utf-8") as f:
            f.write('{"slot":"a","slots":{"a":{"env_steps":null},"b":null}}')
        text = inspect_runs.checkpoint_summary(ckpt)
        assert "a@-1[m!r!r!]" in text
        assert "b@-1[m!r!r!]" in text


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print("%d checks passed" % len(tests))

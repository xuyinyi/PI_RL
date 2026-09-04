from RL_PPO.envs.embedding import _checkpoint_fingerprint


def test_local_checkpoint_fingerprint_covers_all_nontransient_files(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    extra = tmp_path / "custom-tokenizer.asset"
    extra.write_bytes(b"first")
    first = _checkpoint_fingerprint(str(tmp_path))
    assert first == _checkpoint_fingerprint(str(tmp_path))

    extra.write_bytes(b"second")
    assert _checkpoint_fingerprint(str(tmp_path)) != first


def test_local_checkpoint_fingerprint_ignores_only_transient_trees(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    first = _checkpoint_fingerprint(str(tmp_path))
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "temporary.pyc").write_bytes(b"transient")
    assert _checkpoint_fingerprint(str(tmp_path)) == first

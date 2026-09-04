from RL_PPO.envs.rng import derive_seed, named_index, stable_uint64


def test_named_streams_are_replayable_and_order_independent():
    assert named_index(12, "dianhydride_product", 7) == named_index(
        12, "dianhydride_product", 7
    )
    before = named_index(12, "diamine_product", 11)
    _ = named_index(12, "dianhydride_product", 3)
    after = named_index(12, "diamine_product", 11)
    assert before == after


def test_seed_derivation_is_stable_and_semantically_separated():
    assert derive_seed(7, "episode", 3) == derive_seed(7, "episode", 3)
    assert derive_seed(7, "episode", 3) != derive_seed(7, "episode", 4)
    assert stable_uint64(7, "a") != stable_uint64(7, "b")
    assert stable_uint64(7, 1) != stable_uint64(7, "1")

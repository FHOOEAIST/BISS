from biss.dummy import dummy_world


def test_dummy() -> None:
    # given
    expected = "dummy"
    # when
    dummy = dummy_world()
    # then
    assert expected == dummy

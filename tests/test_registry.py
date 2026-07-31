from detector.registry import CategoryRegistry


def test_register_and_get():
    reg = CategoryRegistry()
    assert reg.get("apple") is not None
    assert reg.get("bread") is not None
    assert reg.get("beverage") is not None
    assert reg.get("unknown") is None
    assert len(reg.list()) == 3

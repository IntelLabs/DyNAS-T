from dynast.utils.nn import AverageMeter


def test_average_meter():
    am = AverageMeter()

    val = 1
    am.update(val=val)
    assert am.val == val
    assert am.count == 1
    assert am.avg == 1

    val = 2
    am.update(val=val, n=2)
    assert am.val == val
    assert am.count == 3
    assert am.avg == 5 / 3

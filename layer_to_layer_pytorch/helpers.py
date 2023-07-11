from typing import Iterable

from tqdm.auto import tqdm
from tqdm.contrib import tenumerate, tzip


def iterator(iterable: Iterable, verbose: bool, **kwargs):
    if not verbose:
        return iterable

    return tqdm(iterable, **kwargs)

# 定义了一个名为 enumerator 的函数，它用于根据给定的参数来选择合适的遍历方式。
# enumerator 函数接受两个参数：
# iterable：要遍历的可迭代对象。
# verbose：布尔值，指示是否启用详细模式的标志。
def enumerator(iterable: Iterable, verbose: bool, **kwargs):
    # 如果 verbose 的值为 False，则直接调用内置的 enumerate 函数对 iterable 进行遍历，并返回遍历结果。
    if not verbose:
        return enumerate(iterable)

    # tenumerate 函数是 tqdm.contrib.tenumerate 函数的别名，
    # 它是 tqdm 库提供的增强版遍历函数，用于在遍历过程中显示进度条和其他附加信息。
    return tenumerate(iterable, **kwargs)


def zipper(iterable1: Iterable, iterable2, verbose: bool, **kwargs):
    if not verbose:
        return zip(iterable1, iterable2)

    return tzip(iterable1, iterable2, **kwargs)


__all__ = ["iterator", "enumerator", "zipper"]

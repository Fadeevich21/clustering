from src.range import Range


class Area:
    __ranges = []

    def __init__(self) -> None:
        pass

    def add_range(self, range_: Range) -> None:
        self.__ranges.append(range_)

    def remove_range(self, index_range: int) -> None:
        self.__ranges.pop(index_range)

    def clear(self) -> None:
        self.__ranges.clear()

    def get_range(self, index_range: int) -> Range:
        return self.__ranges[index_range]

    def __len__(self) -> int:
        return len(self.__ranges)

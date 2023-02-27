class Dot:
    __dimensions: int = 0
    __values: list = []

    def __init__(self, dimensions: int) -> None:
        self.__dimensions = dimensions
        self.__values = [0] * self.__dimensions

    def set_value(self, dimension: int, value) -> None:
        self.__values[dimension] = value

    def get_value(self, dimension: int):
        return self.__values[dimension]

    def __len__(self) -> int:
        return self.__dimensions

    def __repr__(self):
        s = '['
        for i in range(self.__dimensions):
            s += str(self.__values[i])
            if i+1 != self.__dimensions:
                s += ", "
        s += "]"

        return s

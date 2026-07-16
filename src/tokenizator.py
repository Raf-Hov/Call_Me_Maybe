from src.jsonloader import ArgPars
from typing import Generator
from string import printable


class Tokenizator(ArgPars):
    def generator(self) -> Generator:
        self.container()
        for i in self.tokens:
            yield i.values()

    def tokeniz(self) -> None:
        gen = self.generator()
        new_list = list(next(gen))[0]
        new_list = [i if i != " " else "_" for i in new_list]
        print(new_list)
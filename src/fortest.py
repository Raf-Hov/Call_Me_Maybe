# from dataclasses import dataclass
# from pydantic.dataclasses import dataclass
from pydantic import BaseModel, Field
from enum import Enum


class Baba(str, Enum):
    poss = "aaaaa"


class User:
    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age

    def __repr__(self) -> str:
        return f"{__class__.__name__}()"

    def __eq__(self, value: object) -> bool:
        if isinstance(value, User):
            return self.name == value.name and self.age == value.age
        return False


class ServiceConfig(BaseModel):
    name: str = Field()
    port: int = Field()


us1 = User("as", 1)
us2 = User("ad", 2)
item1 = ServiceConfig(name="apee", port=500)

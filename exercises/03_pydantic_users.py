from pydantic import BaseModel

# Define a User model with name: str, age: int, email: str.


class User(BaseModel):
    name: str
    age: int
    email: str

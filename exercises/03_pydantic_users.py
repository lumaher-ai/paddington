from pydantic import BaseModel, EmailStr, ValidationError


# Define a User model with name: str, age: int, email: str.
class User(BaseModel):
    name: str
    age: int
    email: EmailStr


# Define a UserSummary model with total: int, adults: int, average_age: float.
class UserSummary(BaseModel):
    total: int
    adults: int
    average_age: float


# Rewrite parse_user, filter_adults, and summarize to take User instances and return User / UserSummary.


# Exercice 1: type hint using dict
def parse_user(raw: dict) -> User:
    return User.model_validate(raw)


# Exercise 2: list comprehension
def filter_adults(users: list[User]) -> list[User]:
    return [user for user in users if user.age >= 18]


# Exercise 3: summarize
def summarize(users: list[User]) -> UserSummary:
    if not users:
        return UserSummary(total=0, adults=0, average_age=0.0)

    total = len(users)
    adults = sum(1 for user in users if user.age >= 18)
    average_age = sum(user.age for user in users) / total if total > 0 else 0.0

    return UserSummary(total=total, adults=adults, average_age=average_age)


# Create 5 User instances, print the result with summary.model_dump_json(indent=2).

if __name__ == "__main__":
    users = [
        User(name="Luisa", age=26, email="lmhm@gmail.com"),
        User(name="David", age=25, email="dadom@gmail.com"),
        User(name="Kiara", age=16, email="kiaher@gmail.com"),
        User(name="Isabel", age=14, email="shopy@gmail.com"),
        User(name="Daniela", age=13, email="daher@gmail.com"),
    ]

print("Summary:")
print(summarize(users).model_dump_json(indent=2))

# Demonstrate Pydantic's validation by feeding it invalid data

invalid_data = {"name": "Maria", "age": "not a number", "email": "maria@gmail.com"}
print("\nAttempting to create User with invalid data:")
try:
    User.model_validate(invalid_data)
except ValidationError as e:
    print(f"Validation failed with {e.error_count()} error(s)")
    print(e)

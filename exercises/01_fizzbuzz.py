# Version 1: imperative with for loop and if/elif/else
print("--- Imperative version ---")
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)

# Version 2: idiomatic with list comprehension
print("--- List comprehension version ---")
results = [
    "FizzBuzz" if i % 3 == 0  and i % 5 == 0
    else "Fizz" if i % 3 == 0
    else "Buzz" if i % 5 == 0
    else i
    for i in range (1, 101)
]
for item in results:
    print(item)